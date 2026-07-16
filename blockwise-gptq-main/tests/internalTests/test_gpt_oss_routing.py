"""GPT-OSS expert-routing tests (P0.2).

Validates patch_expert_forward() against the REAL GptOssExperts implementation
installed in .venv-quant (transformers 5.14.0 top-k routing contract):

    router_indices  [num_tokens, top_k]   expert IDs
    routing_weights [num_tokens, top_k]   softmax over top-k logits
    weight lookup   routing_weights[token_idx, top_k_pos]  (by POSITION)

Covers the handoff §12 battery:
  - 32 experts with top-4 routing
  - expert IDs greater than top_k (the old patch crashed at one_hot here)
  - repeated expert IDs across tokens
  - an expert selected at different top-k positions (positional weight lookup)
  - tokens routed to multiple experts
  - unselected experts receive no Hessian samples
  - patched forward vs the reference implementation (numerical equivalence)
  - add_batch receives exactly the right token subsets, in expert order
  - dense-variant (routing_weights[tokens, num_experts]) compatibility branch

Run (from anywhere, in .venv-quant):
    pytest -q tests/internalTests/test_gpt_oss_routing.py
    python  tests/internalTests/test_gpt_oss_routing.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
# Repo-relative code root (P0.1): library at <repo>/opteam-blockwise-gptq.
_CODE_ROOT = Path(__file__).resolve().parents[2] / "opteam-blockwise-gptq"
sys.path.insert(0, str(_CODE_ROOT))

from gpt_oss_expert_gptq import patch_expert_forward  # noqa: E402

torch.manual_seed(0)

# ── Fixtures ───────────────────────────────────────────────────────────────────

NUM_EXPERTS  = 32
TOP_K        = 4
HIDDEN       = 64
INTERMEDIATE = 48
NUM_TOKENS   = 96


class _Recorder:
    """Stands in for GPTQ/_GptqH: records every add_batch input."""

    def __init__(self):
        self.batches = []

    def add_batch(self, inp, out=None):
        self.batches.append(inp.detach().clone())

    @property
    def nsamples(self):
        return sum(b.shape[0] for b in self.batches)


def _make_experts():
    """Build a real transformers GptOssExperts with random weights (float32)."""
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    cfg = GptOssConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    experts = GptOssExperts(cfg).float()
    with torch.no_grad():
        for p in experts.parameters():
            p.normal_(0.0, 0.5)
    experts.eval()
    return experts


def _make_routing(num_tokens=NUM_TOKENS, seed=1):
    """Random top-k routing in the pinned (5.14.0) contract."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=g)
    top_val, router_indices = torch.topk(logits, TOP_K, dim=-1)
    routing_weights = F.softmax(top_val, dim=1)
    return router_indices, routing_weights


def _patched_run(experts, hidden, router_indices, routing_weights):
    """Patch, run one forward, restore. Returns (output, gu_recs, dn_recs, calls)."""
    gu = {e: _Recorder() for e in range(NUM_EXPERTS)}
    dn = {e: _Recorder() for e in range(NUM_EXPERTS)}
    counter = {}
    original = patch_expert_forward(experts, gu, dn, call_counter=counter)
    try:
        with torch.no_grad():
            out = experts.forward(hidden, router_indices=router_indices,
                                  routing_weights=routing_weights)
    finally:
        experts.forward = original
    return out, gu, dn, counter


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_reference_contract_is_topk():
    """Sanity: the installed transformers really uses the top-k contract.

    If this fails, the pinned routing contract changed — re-verify
    patch_expert_forward against the new GptOssExperts.forward.
    """
    import inspect
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssTopKRouter
    src = inspect.getsource(GptOssExperts.forward)
    assert "routing_weights[token_idx, top_k_pos" in src.replace("\n", ""), (
        "GptOssExperts.forward no longer indexes routing_weights by top-k "
        "position — the pinned contract changed; update patch_expert_forward."
    )
    rsrc = inspect.getsource(GptOssTopKRouter.forward)
    assert "topk" in rsrc


def test_patched_matches_reference_forward():
    """Patched forward ≡ real GptOssExperts.forward on identical inputs."""
    experts = _make_experts()
    hidden = torch.randn(NUM_TOKENS, HIDDEN)
    router_indices, routing_weights = _make_routing()

    with torch.no_grad():
        ref = experts.forward(hidden, router_indices=router_indices,
                              routing_weights=routing_weights)

    out, _, _, counter = _patched_run(experts, hidden, router_indices, routing_weights)

    assert counter["n"] == 1, "patched forward was not invoked"
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_expert_ids_above_topk_do_not_crash():
    """The old patch crashed one_hot for expert IDs >= top_k+1. Force IDs 28-31."""
    experts = _make_experts()
    num_tokens = 8
    hidden = torch.randn(num_tokens, HIDDEN)
    router_indices = torch.tensor([[28, 29, 30, 31]] * num_tokens)
    routing_weights = F.softmax(torch.randn(num_tokens, TOP_K), dim=1)

    with torch.no_grad():
        ref = experts.forward(hidden, router_indices=router_indices,
                              routing_weights=routing_weights)
    out, gu, _, _ = _patched_run(experts, hidden, router_indices, routing_weights)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    for e in (28, 29, 30, 31):
        assert gu[e].nsamples == num_tokens
    for e in range(28):
        assert gu[e].nsamples == 0


def test_weight_lookup_is_by_topk_position():
    """Same expert at different top-k positions must use each token's own
    positional weight. Construct routing where expert 7 is at position 0 for
    token 0 and position 3 for token 1, with very different weights."""
    experts = _make_experts()
    hidden = torch.randn(2, HIDDEN)
    router_indices = torch.tensor([[7, 1, 2, 3],
                                   [4, 5, 6, 7]])
    # Distinct, hand-picked weights (rows sum to 1)
    routing_weights = torch.tensor([[0.70, 0.10, 0.10, 0.10],
                                    [0.05, 0.05, 0.05, 0.85]])

    with torch.no_grad():
        ref = experts.forward(hidden, router_indices=router_indices,
                              routing_weights=routing_weights)
    out, _, _, _ = _patched_run(experts, hidden, router_indices, routing_weights)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    # Directly verify expert 7's contribution uses 0.70 for token 0 and 0.85
    # for token 1 (positional), not a single expert-indexed weight.
    with torch.no_grad():
        gate_up = hidden @ experts.gate_up_proj[7] + experts.gate_up_proj_bias[7]
        gated   = experts._apply_gate(gate_up)
        contrib = gated @ experts.down_proj[7] + experts.down_proj_bias[7]
        expected_0 = contrib[0] * 0.70
        expected_1 = contrib[1] * 0.85

        # Zero every other expert's weights → isolate expert 7 in the output
        experts2 = _make_experts()
        with torch.no_grad():
            experts2.gate_up_proj.copy_(experts.gate_up_proj)
            experts2.gate_up_proj_bias.copy_(experts.gate_up_proj_bias)
            experts2.down_proj.copy_(experts.down_proj)
            experts2.down_proj_bias.copy_(experts.down_proj_bias)
            for e in range(NUM_EXPERTS):
                if e != 7:
                    experts2.gate_up_proj[e].zero_()
                    experts2.gate_up_proj_bias[e].zero_()
                    experts2.down_proj[e].zero_()
                    experts2.down_proj_bias[e].zero_()

    iso, _, _, _ = _patched_run(experts2, hidden, router_indices, routing_weights)
    # Other experts still contribute constant bias-only terms == 0 after zeroing
    # (bias zeroed too), except SwiGLU of zeros: gate=0,up=0 → glu=0 → (0+1)*0=0. OK.
    torch.testing.assert_close(iso[0], expected_0, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(iso[1], expected_1, rtol=1e-4, atol=1e-4)


def test_add_batch_receives_exact_token_subsets():
    """Per-expert Hessian inputs must be exactly the tokens routed to that
    expert (gate_up) and the recomputed gated activations (down)."""
    experts = _make_experts()
    hidden = torch.randn(NUM_TOKENS, HIDDEN)
    router_indices, routing_weights = _make_routing(seed=3)

    _, gu, dn, _ = _patched_run(experts, hidden, router_indices, routing_weights)

    mask = F.one_hot(router_indices, num_classes=NUM_EXPERTS).permute(2, 1, 0)
    for e in range(NUM_EXPERTS):
        _, token_idx = torch.where(mask[e])
        if token_idx.numel() == 0:
            assert gu[e].nsamples == 0 and dn[e].nsamples == 0
            continue
        assert len(gu[e].batches) == 1
        torch.testing.assert_close(gu[e].batches[0], hidden[token_idx].float())

        with torch.no_grad():
            gate_up = hidden[token_idx] @ experts.gate_up_proj[e] \
                      + experts.gate_up_proj_bias[e]
            gated   = experts._apply_gate(gate_up)
        torch.testing.assert_close(dn[e].batches[0], gated.float(),
                                   rtol=1e-5, atol=1e-5)


def test_repeated_and_multi_expert_tokens():
    """Repeated expert IDs across tokens + every token hits TOP_K experts."""
    experts = _make_experts()
    num_tokens = 16
    hidden = torch.randn(num_tokens, HIDDEN)
    # All tokens pick the SAME experts (heavy repetition)
    router_indices = torch.tensor([[3, 11, 19, 27]] * num_tokens)
    routing_weights = F.softmax(torch.randn(num_tokens, TOP_K), dim=1)

    with torch.no_grad():
        ref = experts.forward(hidden, router_indices=router_indices,
                              routing_weights=routing_weights)
    out, gu, _, _ = _patched_run(experts, hidden, router_indices, routing_weights)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    for e in (3, 11, 19, 27):
        assert gu[e].nsamples == num_tokens          # every token, once
    assert sum(r.nsamples for r in gu.values()) == num_tokens * TOP_K


def test_unselected_experts_get_zero_samples():
    experts = _make_experts()
    hidden = torch.randn(8, HIDDEN)
    router_indices = torch.tensor([[0, 1, 2, 3]] * 8)   # only experts 0-3
    routing_weights = F.softmax(torch.randn(8, TOP_K), dim=1)

    _, gu, dn, _ = _patched_run(experts, hidden, router_indices, routing_weights)
    for e in range(4, NUM_EXPERTS):
        assert gu[e].nsamples == 0
        assert dn[e].nsamples == 0


def test_dense_variant_branch():
    """Compatibility branch: routing_weights given as a dense
    [tokens, num_experts] scatter (early-4.55-style) must produce the same
    output as the top-k contract on equivalent inputs."""
    experts = _make_experts()
    hidden = torch.randn(NUM_TOKENS, HIDDEN)
    router_indices, topk_weights = _make_routing(seed=7)

    dense = torch.zeros(NUM_TOKENS, NUM_EXPERTS)
    dense.scatter_(1, router_indices, topk_weights)

    with torch.no_grad():
        ref = experts.forward(hidden, router_indices=router_indices,
                              routing_weights=topk_weights)

    out, _, _, _ = _patched_run(experts, hidden, router_indices, dense)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_3d_input_shape_preserved():
    """If a caller passes [batch, seq, hidden], output shape must match."""
    experts = _make_experts()
    b, s = 4, 24
    hidden = torch.randn(b, s, HIDDEN)
    router_indices, routing_weights = _make_routing(num_tokens=b * s, seed=9)

    with torch.no_grad():
        ref = experts.forward(hidden.reshape(-1, HIDDEN),
                              router_indices=router_indices,
                              routing_weights=routing_weights)
    out, _, _, _ = _patched_run(experts, hidden, router_indices, routing_weights)
    assert out.shape == (b, s, HIDDEN)
    torch.testing.assert_close(out.reshape(-1, HIDDEN), ref, rtol=1e-5, atol=1e-5)


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
