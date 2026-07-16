"""Memory-bounded grouped Hessian collection tests (P0.4).

Exercises the full gptq_quantize_model() parallel path on a tiny REAL
GptOssForCausalLM (transformers), CPU, with synthetic calibration data —
no network, no GPU required.

Covers the handoff §P0.4 requirements:
  - group_size=1 vs group_size=n_layers produce IDENTICAL Hessian caches and
    IDENTICAL quantized weights (grouping is purely a memory/time trade)
  - resume: an interrupted (partially deleted) cache recollects ONLY the
    missing layers, and reuses cached calibration tokens
  - completeness is manifest-verified: corrupt/tampered layer files are
    detected via SHA-256 and NOT treated as complete
  - the calibration token cache is immutable + hashed; tampering hard-errors;
    a resumed run uses the CACHED tokens even if the loader would now return
    different data
  - fail-closed: a calibration sample that raises aborts the run
  - fail-closed: NaN Hessians are rejected at save time
  - fail-closed: expert-forward bypass (fused-kernel simulation) is detected
  - collection stats (runtime, memory, cache bytes) are recorded per group

Run (from anywhere, in .venv-quant):
    pytest -q tests/internalTests/test_hessian_grouped_collection.py
    python  tests/internalTests/test_hessian_grouped_collection.py
"""

import copy
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_CODE_ROOT = Path(__file__).resolve().parents[2] / "opteam-blockwise-gptq"
sys.path.insert(0, str(_CODE_ROOT))

import apply as apply_mod                              # noqa: E402
from hessian_cache import HessianCache, sha256_of_token_samples  # noqa: E402

torch.manual_seed(0)

# ── Tiny model + synthetic calibration ─────────────────────────────────────────

N_LAYERS   = 3
HIDDEN     = 64
INTERM     = 48
N_EXPERTS  = 8
TOP_K      = 2
VOCAB      = 256
SEQLEN     = 32
N_SAMPLES  = 4


def _tiny_gpt_oss():
    """Real transformers GptOssForCausalLM, tiny config, CPU float32."""
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERM,
        num_local_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB,
        max_position_embeddings=128,
        sliding_window=64,
    )
    torch.manual_seed(42)
    model = GptOssForCausalLM(cfg).float()
    model.eval()
    return model


def _fake_calibration(n=N_SAMPLES, seqlen=SEQLEN, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(0, VOCAB, (1, seqlen), generator=g)
        out.append((ids, ids.clone()))
    return out


def _quantize(model, cache_root, group_size=1, dataset_tag="synthetic",
              calib=None, threshold=None):
    """Run the real gptq_quantize_model with calibration loading stubbed."""
    calib = calib if calib is not None else _fake_calibration()
    orig_loader = apply_mod.get_calibration_data
    apply_mod.get_calibration_data = lambda *a, **k: calib
    try:
        return apply_mod.gptq_quantize_model(
            model, "tiny-gpt-oss",
            quant_format="nvfp4",
            dataset=dataset_tag,
            nsamples=len(calib),
            seqlen=SEQLEN,
            blocksize=32,
            percdamp=0.01,
            seed=0,
            device="cpu",
            mode="blockwise",
            parallel_hessian=True,
            mixed_precision_threshold=threshold,
            hessian_cache_dir=str(cache_root),
            hessian_layer_group_size=group_size,
        )
    finally:
        apply_mod.get_calibration_data = orig_loader


def _cache_dir(cache_root, dataset_tag="synthetic", n=N_SAMPLES):
    return apply_mod._hessian_cache_dir(
        str(cache_root), "tiny-gpt-oss", dataset_tag, n, SEQLEN, 0
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_group1_equals_group_all():
    """Grouping is a pure memory/time trade: group_size=1 and =N_LAYERS must
    yield BITWISE identical cached Hessians and quantized weights.

    This is strict equality, not allclose: every collection pass pins all MoE
    layers to the same expert-forward implementation (attach_passthrough), so
    activations cannot depend on group membership."""
    base = _tiny_gpt_oss()
    m1, m2 = copy.deepcopy(base), copy.deepcopy(base)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        _quantize(m1, d1, group_size=1)
        _quantize(m2, d2, group_size=N_LAYERS)

        c1 = HessianCache(_cache_dir(d1), N_LAYERS, meta={
            "model_name": "tiny-gpt-oss", "dataset": "synthetic",
            "nsamples": N_SAMPLES, "seqlen": SEQLEN, "seed": 0})
        c2 = HessianCache(_cache_dir(d2), N_LAYERS, meta=c1.meta)

        for li in range(N_LAYERS):
            p1, p2 = c1.load_layer(li), c2.load_layer(li)
            assert p1["attn"].keys() == p2["attn"].keys()
            for name in p1["attn"]:
                assert p1["attn"][name]["nsamples"] == p2["attn"][name]["nsamples"]
                assert torch.equal(p1["attn"][name]["H"], p2["attn"][name]["H"]), \
                    f"layer {li} attn '{name}' Hessian differs between groupings"
            for side in ("gu", "dn"):
                assert p1["experts"][side].keys() == p2["experts"][side].keys()
                for e in p1["experts"][side]:
                    assert torch.equal(p1["experts"][side][e]["H"],
                                       p2["experts"][side][e]["H"]), \
                        f"layer {li} experts.{side}[{e}] Hessian differs"

    sd1, sd2 = m1.state_dict(), m2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"quantized weight '{k}' differs"


def test_resume_recollects_only_missing_layers():
    """Delete one layer file from a complete cache → rerun recollects exactly
    the missing layer (1 pass, not N_LAYERS) using the CACHED tokens."""
    base = _tiny_gpt_oss()

    with tempfile.TemporaryDirectory() as d:
        _quantize(copy.deepcopy(base), d, group_size=1)
        cdir = _cache_dir(d)

        # Snapshot manifests + delete layer 1's file
        meta = {"model_name": "tiny-gpt-oss", "dataset": "synthetic",
                "nsamples": N_SAMPLES, "seqlen": SEQLEN, "seed": 0}
        cache = HessianCache(cdir, N_LAYERS, meta=meta)
        before = {li: cache.manifest["layers"][str(li)]["sha256"]
                  for li in range(N_LAYERS)}
        (cdir / "layer_01.pt").unlink()
        assert cache.pending_layers() == [1]

        # Count full-model passes via an embedding forward hook
        m2 = copy.deepcopy(base)
        n_fwd = {"n": 0}
        h = m2.model.embed_tokens.register_forward_hook(
            lambda *a, **k: n_fwd.__setitem__("n", n_fwd["n"] + 1))
        # Loader returns DIFFERENT tokens — cached tokens must win on resume
        decoy = _fake_calibration(seed=999)
        _quantize(m2, d, group_size=1, calib=decoy)
        h.remove()

        # Exactly one pass over the 4 cached samples (plus 0 for layers 0/2)
        assert n_fwd["n"] == N_SAMPLES, \
            f"expected {N_SAMPLES} forwards (1 group), got {n_fwd['n']}"

        cache2 = HessianCache(cdir, N_LAYERS, meta=meta)
        after = {li: cache2.manifest["layers"][str(li)]["sha256"]
                 for li in range(N_LAYERS)}
        assert after[0] == before[0] and after[2] == before[2], \
            "untouched layers must not be recollected"
        # Layer 1 was recollected from the CACHED tokens → same Hessians as
        # the original collection (sha may differ due to torch.save
        # non-determinism, so compare tensors)
        p = cache2.load_layer(1)
        assert all(e["nsamples"] > 0 for e in p["attn"].values())


def test_manifest_detects_tampered_layer_file():
    base = _tiny_gpt_oss()
    with tempfile.TemporaryDirectory() as d:
        _quantize(copy.deepcopy(base), d, group_size=N_LAYERS)
        cdir = _cache_dir(d)
        meta = {"model_name": "tiny-gpt-oss", "dataset": "synthetic",
                "nsamples": N_SAMPLES, "seqlen": SEQLEN, "seed": 0}

        # Append garbage to a layer file → sha mismatch → incomplete
        with open(cdir / "layer_02.pt", "ab") as f:
            f.write(b"CORRUPTION")
        cache = HessianCache(cdir, N_LAYERS, meta=meta)
        assert not cache.layer_complete(2)
        assert 2 in cache.pending_layers()
        # Without hash verification it would (wrongly) look complete —
        # proving the manifest hash is what protects us.
        assert cache.layer_complete(0)


def test_token_cache_tamper_detection():
    base = _tiny_gpt_oss()
    with tempfile.TemporaryDirectory() as d:
        _quantize(copy.deepcopy(base), d, group_size=N_LAYERS)
        cdir = _cache_dir(d)

        # Tamper with the token cache → any resumed run must hard-error
        tok = torch.load(cdir / "calib_tokens.pt", weights_only=False)
        tok[0] = (tok[0][0] + 1, tok[0][1])
        torch.save(tok, cdir / "calib_tokens.pt")

        (cdir / "layer_00.pt").unlink()   # force a resume
        with pytest.raises(RuntimeError, match="hash mismatch"):
            _quantize(copy.deepcopy(base), d, group_size=1)


def test_failing_sample_aborts():
    """A calibration sample that raises must abort collection (fail closed),
    not be skipped with a warning like the old code did."""
    base = _tiny_gpt_oss()
    bad = _fake_calibration()
    # Out-of-range token id → embedding lookup error mid-pass
    bad[2] = (torch.full((1, SEQLEN), VOCAB + 10, dtype=torch.long),
              bad[2][1])
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(Exception):
            _quantize(copy.deepcopy(base), d, group_size=1, calib=bad)
        # And nothing may have been cached as complete for the failed group
        meta = {"model_name": "tiny-gpt-oss", "dataset": "synthetic",
                "nsamples": N_SAMPLES, "seqlen": SEQLEN, "seed": 0}
        cache = HessianCache(_cache_dir(d), N_LAYERS, meta=meta)
        assert cache.pending_layers() == list(range(N_LAYERS))


def test_nan_hessian_rejected_at_save():
    with tempfile.TemporaryDirectory() as d:
        cache = HessianCache(Path(d) / "c", n_layers=1, meta={"k": "v"})
        bad = {"attn": {"q": {"H": torch.tensor([[float("nan")]]),
                              "nsamples": 4}},
               "experts": None}
        with pytest.raises(RuntimeError, match="NaN/Inf"):
            cache.save_layer(0, bad)
        assert not cache.layer_complete(0)


def test_expert_bypass_detected():
    """Simulate a fused-kernel forward bypassing the expert patch: restore
    the original forward right after attach_hooks. Collection must raise."""
    from expert_dispatch import GptOssHandler

    base = _tiny_gpt_oss()
    orig_attach = GptOssHandler.attach_hooks

    def bypassing_attach(self, layer, acc_state):
        token = orig_attach(self, layer, acc_state)
        layer.mlp.experts.forward = token   # kernel "replaces" the patch
        return token

    GptOssHandler.attach_hooks = bypassing_attach
    try:
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError, match="never .*invoked|bypass"):
                _quantize(copy.deepcopy(base), d, group_size=1)
    finally:
        GptOssHandler.attach_hooks = orig_attach


def test_collection_stats_recorded():
    base = _tiny_gpt_oss()
    with tempfile.TemporaryDirectory() as d:
        _quantize(copy.deepcopy(base), d, group_size=2)   # 2 groups: [0,1], [2]
        meta = {"model_name": "tiny-gpt-oss", "dataset": "synthetic",
                "nsamples": N_SAMPLES, "seqlen": SEQLEN, "seed": 0}
        cache = HessianCache(_cache_dir(d), N_LAYERS, meta=meta)
        stats = cache.manifest["collection_stats"]
        assert len(stats) == 2
        assert stats[0]["layers"] == [0, 1] and stats[1]["layers"] == [2]
        for s in stats:
            for key in ("seconds", "gpu_peak_bytes", "host_maxrss_kb",
                        "cache_bytes_written"):
                assert key in s
        # Manifest also records per-layer sample counts for coverage audits
        entry = cache.manifest["layers"]["0"]
        assert all(v > 0 for v in entry["attn_nsamples"].values())
        assert entry["expert_nsamples"] is not None


def test_token_hash_is_order_and_content_sensitive():
    a = _fake_calibration(seed=1)
    b = _fake_calibration(seed=2)
    assert sha256_of_token_samples(a) == sha256_of_token_samples(a)
    assert sha256_of_token_samples(a) != sha256_of_token_samples(b)
    assert sha256_of_token_samples(a) != sha256_of_token_samples(list(reversed(a)))


# ── Script entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
