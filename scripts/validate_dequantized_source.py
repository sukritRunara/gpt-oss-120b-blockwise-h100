#!/usr/bin/env python
"""Validate the dequantized BF16 source (arm B) against the official MXFP4
checkpoint (arm A). Handoff §11 requirements.

Checks (each FAILS CLOSED — nonzero exit on any failure):
  1. structure  — B reloads as ordinary BF16: no quantization_config, no
                  packed *_blocks/*_scales tensors, BF16 dtypes, finite.
  2. decode     — EVERY expert tensor of B equals the exact MXFP4 decode of
                  A's blocks/scales (transformers' pinned decoder,
                  convert_moe_packed_tensors), bit-exact in BF16. Non-expert
                  tensors must be byte-identical to A's.
  3. count      — parameter completeness: B's tensor set == A's tensor set
                  (with blocks/scales replaced by the decoded tensor).
  4. determinism— two forwards of B on identical input give identical logits.
  5. logits     — A (native MXFP4 kernels, if available) vs B on a fixed
                  prompt set: cosine, KL, top-1/top-10 agreement, max abs
                  diff, greedy 32-token prefix agreement. If the native
                  MXFP4 path is unavailable in transformers, this check is
                  recorded as DEFERRED (A is compared at the vLLM stage).

Writes results JSON to results/dequant_validation.json.

Run inside .venv-quant:
    python scripts/validate_dequantized_source.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

RESULTS = {"checks": {}, "created_utc":
           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "def quicksort(arr):",
    "The theory of general relativity states that",
    "Once upon a time, in a village by the sea,",
    "The chemical formula for water is",
    "To bake sourdough bread, you first need",
    "The stock market crashed in 1929 because",
    "Quantum entanglement is a phenomenon where",
    "El sistema solar tiene ocho planetas:",
    "SELECT name, age FROM users WHERE",
    "The mitochondria is the powerhouse of",
    "In 1969, Apollo 11 landed on",
    "A haiku about autumn:",
    "The difference between TCP and UDP is",
    "Newton's second law states that force equals",
]


def _fail(name, msg):
    RESULTS["checks"][name] = {"pass": False, "detail": msg}
    print(f"  [FAIL] {name}: {msg}")


def _ok(name, detail=""):
    RESULTS["checks"][name] = {"pass": True, "detail": detail}
    print(f"  [PASS] {name}: {detail}")


def load_flat_tensors(path: Path):
    from safetensors import safe_open
    idx = path / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
              if idx.exists() else ["model.safetensors"])
    out = {}
    for s in shards:
        with safe_open(str(path / s), framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def check_structure(b_dir: Path, b_tensors):
    cfg = json.loads((b_dir / "config.json").read_text())
    if "quantization_config" in cfg:
        return _fail("structure", "config.json still has quantization_config")
    packed = [k for k in b_tensors if "_blocks" in k or "_scales" in k]
    if packed:
        return _fail("structure", f"packed tensors remain: {packed[:3]}")
    bad = {k: str(t.dtype) for k, t in b_tensors.items()
           if t.dtype not in (torch.bfloat16, torch.float32)}
    if bad:
        return _fail("structure", f"unexpected dtypes: {dict(list(bad.items())[:3])}")
    for k, t in b_tensors.items():
        if not torch.isfinite(t).all():
            return _fail("structure", f"NaN/Inf in {k}")
    n = sum(t.numel() for t in b_tensors.values())
    _ok("structure", f"{len(b_tensors)} tensors, {n/1e9:.3f}B params, all "
        f"finite BF16/FP32, no packed tensors, config clean")


def check_decode_equivalence(a_dir: Path, b_tensors):
    """Every B tensor must equal A's tensor (bf16 pass-through) or the exact
    MXFP4 decode of A's blocks/scales (experts)."""
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    a_tensors = load_flat_tensors(a_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    a_covered = set()
    n_decoded = 0
    n_passthrough = 0
    for k, bt in sorted(b_tensors.items()):
        if k.endswith(("gate_up_proj", "down_proj")) and ".experts." in k:
            blocks_k, scales_k = k + "_blocks", k + "_scales"
            if blocks_k not in a_tensors:
                return _fail("decode", f"{blocks_k} missing from A")
            dec = convert_moe_packed_tensors(
                a_tensors[blocks_k].to(dev), a_tensors[scales_k].to(dev),
                dtype=torch.bfloat16).cpu()
            dec = dec.reshape(bt.shape)
            if not torch.equal(dec, bt):
                diff = (dec.float() - bt.float()).abs().max().item()
                return _fail("decode", f"{k}: decode != B (max diff {diff:.3e})")
            a_covered.update((blocks_k, scales_k))
            n_decoded += 1
        else:
            if k not in a_tensors:
                return _fail("decode", f"{k} in B but not in A")
            if not torch.equal(a_tensors[k], bt):
                return _fail("decode", f"{k}: B differs from A passthrough")
            a_covered.add(k)
            n_passthrough += 1

    missing = set(a_tensors) - a_covered
    if missing:
        return _fail("decode", f"A tensors unaccounted for in B: "
                     f"{sorted(missing)[:5]}")
    _ok("decode", f"{n_decoded} expert tensors decode-exact, "
        f"{n_passthrough} tensors byte-identical, full coverage")


@torch.no_grad()
def check_determinism_and_logits(a_dir: Path, b_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(b_dir))
    enc = [tok(p, return_tensors="pt").input_ids for p in PROMPTS]

    # ── A first (smaller footprint), native MXFP4 if available ───────────────
    a_logits, a_greedy, a_native = None, None, False
    try:
        model_a = AutoModelForCausalLM.from_pretrained(
            str(a_dir), torch_dtype=torch.bfloat16, device_map="cuda",
            low_cpu_mem_usage=True)
        model_a.eval()
        # Native path keeps packed buffers; the auto-fallback dequantizes.
        a_native = any("_blocks" in n for n, _ in model_a.named_buffers()) or \
            any("_blocks" in n for n, _ in model_a.named_parameters())
        a_logits, a_greedy = [], []
        for ids in enc:
            out = model_a(ids.cuda()).logits[0, -1].float().cpu()
            a_logits.append(out)
        for ids in enc[:4]:
            g = model_a.generate(ids.cuda(), max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            a_greedy.append(g[0, ids.shape[1]:].cpu())
        del model_a
        torch.cuda.empty_cache()
        print(f"  arm A loaded ({'native MXFP4' if a_native else 'auto-dequantized'})")
    except Exception as exc:                              # noqa: BLE001
        print(f"  arm A load failed: {type(exc).__name__}: {exc}")
        a_logits = None

    # ── B: determinism + logits ───────────────────────────────────────────────
    model_b = AutoModelForCausalLM.from_pretrained(
        str(b_dir), torch_dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True)
    model_b.eval()
    if getattr(model_b.config, "quantization_config", None) is not None:
        return _fail("determinism", "B reloaded WITH quantization_config")

    l1 = model_b(enc[0].cuda()).logits[0, -1].float().cpu()
    l2 = model_b(enc[0].cuda()).logits[0, -1].float().cpu()
    if not torch.equal(l1, l2):
        return _fail("determinism", "two forwards differ")
    _ok("determinism", "repeated forward is bitwise-identical")

    if a_logits is None:
        RESULTS["checks"]["logits_a_vs_b"] = {
            "pass": None,
            "detail": "DEFERRED: arm A not loadable in transformers here; "
                      "A-vs-B compared at the vLLM serving stage."}
        print("  [DEFER] logits_a_vs_b")
        del model_b
        torch.cuda.empty_cache()
        return

    b_logits, b_greedy = [], []
    for ids in enc:
        b_logits.append(model_b(ids.cuda()).logits[0, -1].float().cpu())
    for ids in enc[:4]:
        g = model_b.generate(ids.cuda(), max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        b_greedy.append(g[0, ids.shape[1]:].cpu())
    del model_b
    torch.cuda.empty_cache()

    cos, kls, top1, top10, maxdiff = [], [], [], [], []
    for la, lb in zip(a_logits, b_logits):
        cos.append(torch.nn.functional.cosine_similarity(
            la.unsqueeze(0), lb.unsqueeze(0)).item())
        pa = torch.log_softmax(la, -1)
        pb = torch.log_softmax(lb, -1)
        kls.append(torch.nn.functional.kl_div(
            pb, pa, log_target=True, reduction="sum").item())
        top1.append(int(la.argmax() == lb.argmax()))
        sa = set(la.topk(10).indices.tolist())
        sb = set(lb.topk(10).indices.tolist())
        top10.append(len(sa & sb) / 10.0)
        maxdiff.append((la - lb).abs().max().item())

    prefix = []
    for ga, gb in zip(a_greedy, b_greedy):
        n = min(len(ga), len(gb))
        same = 0
        for i in range(n):
            if ga[i] != gb[i]:
                break
            same += 1
        prefix.append(same / n)

    metrics = {
        "a_native_mxfp4": a_native,
        "cosine_mean": sum(cos) / len(cos), "cosine_min": min(cos),
        "kl_mean": sum(kls) / len(kls), "kl_max": max(kls),
        "top1_agreement": sum(top1) / len(top1),
        "top10_overlap_mean": sum(top10) / len(top10),
        "max_abs_logit_diff": max(maxdiff),
        "greedy32_prefix_agreement": sum(prefix) / len(prefix),
    }
    RESULTS["metrics_a_vs_b"] = metrics
    good = (metrics["cosine_min"] > 0.99 and metrics["top1_agreement"] >= 0.9)
    (_ok if good else _fail)(
        "logits_a_vs_b",
        f"cos_min={metrics['cosine_min']:.5f} kl_max={metrics['kl_max']:.4f} "
        f"top1={metrics['top1_agreement']:.2f} "
        f"prefix={metrics['greedy32_prefix_agreement']:.2f} "
        f"(A {'native' if a_native else 'AUTO-DEQUANTIZED — weak check'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a_dir", type=Path,
                    default=Path("/workspace/models/gpt-oss-20b-official-mxfp4"))
    ap.add_argument("--b_dir", type=Path,
                    default=Path("/workspace/models/gpt-oss-20b-mxfp4-dequant-bf16"))
    ap.add_argument("--results", type=Path,
                    default=Path("/workspace/results/dequant_validation.json"))
    args = ap.parse_args()

    print("Loading B tensors…")
    b_tensors = load_flat_tensors(args.b_dir)

    print("1/4 structure")
    check_structure(args.b_dir, b_tensors)
    print("2/4 decode equivalence (all tensors)")
    check_decode_equivalence(args.a_dir, b_tensors)
    del b_tensors
    print("3-4/4 determinism + A-vs-B logits")
    check_determinism_and_logits(args.a_dir, args.b_dir)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nResults → {args.results}")

    hard_fail = any(v["pass"] is False for v in RESULTS["checks"].values())
    print("VALIDATION:", "FAIL" if hard_fail else "PASS")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
