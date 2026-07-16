"""
Stage 3 — DeepSeek V2 Lite Shape Tests

Validates fasterquant_blockwise + NVFP4 at the exact linear-layer dimensions
of DeepSeek V2 Lite using synthetic (random) weights. No model download required.
Passes here mean Stage 5 (full quantization) will not hit shape/memory errors.

DeepSeek V2 Lite architecture constants (from config.json):
    hidden_size           = 2048
    moe_intermediate_size = 1408
    dense_intermediate    = 10944  (layer 0 only)
    n_routed_experts      = 64
    n_shared_experts      = 2  →  shared MLP intermediate = 2 × 1408 = 2816
    num_attention_heads   = 16
    qk_nope_head_dim      = 128
    qk_rope_head_dim      = 64
    v_head_dim            = 128
    kv_lora_rank          = 512

MLA attention projections (all layers):
    q_proj              [2048 → 3072]   16 × (128+64)
    kv_a_proj_with_mqa  [2048 → 576]    kv_lora_rank + qk_rope_head_dim
    kv_b_proj           [512  → 4096]   kv_lora_rank → 16 × (128+128)
    o_proj              [2048 → 2048]

Dense MLP (layer 0 only):
    gate_proj / up_proj [2048 → 10944]
    down_proj           [10944 → 2048]

MoE experts (layers 1-26):
    64 × routed: gate_proj/up_proj [2048→1408],  down_proj [1408→2048]
    1  × shared: gate_proj/up_proj [2048→2816],  down_proj [2816→2048]
                 (shared has 2× intermediate because n_shared_experts=2)

Tests:
  1. MLA attention shapes     — q/kv_a/kv_b/o projections at actual dimensions
  2. Dense layer 0 MLP        — gate/up/down at dense intermediate (10944)
  3. is_deepseek_v2_moe       — False for dense layer, True for MoE layers
  4. filter_standard_layers   — expert linears excluded from standard GPTQ subset
  5. Expert instance creation — 64×3 + 1×3 = 195 GPTQ instances; no OOM; shapes OK
  6. Routed expert shapes     — gate/up [2048→1408] and down [1408→2048] quantize
  7. Shared expert shapes     — gate/up [2048→2816] and down [2816→2048] quantize
  8. Full MoE layer smoke     — handler setup → forward → quantize all projections

Runtime: ~20-40 min on DGX Spark GB10
         (Tests 1-5 fast; Tests 6-8 proportional to expert count)
Usage:   python stage3_deepseek_v2_lite_shape_tests.py
Exit:    0 = all passed, 1 = one or more failed
"""

import sys
import math
import time
import traceback

import torch
import torch.nn as nn
from pathlib import Path

# Repo-relative code root (P0.1 fix): the library lives at
# <repo>/opteam-blockwise-gptq regardless of where the repo is checked out.
_CODE_ROOT = Path(__file__).resolve().parents[1] / "opteam-blockwise-gptq"

if not _CODE_ROOT.exists():
    raise RuntimeError(
        f"Code root not found: {_CODE_ROOT}\n"
        "Update _CODE_ROOT at the top of this script."
    )

sys.path.insert(0, str(_CODE_ROOT))
print(f"[path] {_CODE_ROOT}")


from gptq import GPTQ
from quantizer import NVFP4Quantizer
from model_utils import find_layers
from expert_dispatch import get_handler, DeepSeekV2Handler
from deepseek_v2_lite_expert_gptq import (
    is_deepseek_v2_moe,
    setup_expert_gptq_instances,
    register_expert_hooks,
    quantize_and_writeback_experts,
)

# ── Architecture constants ─────────────────────────────────────────────────────

HIDDEN       = 2048
MOE_INTER    = 1408
DENSE_INTER  = 10944
N_ROUTED     = 64
N_SHARED_EXP = 2
SHARED_INTER = N_SHARED_EXP * MOE_INTER  # 2816

N_HEADS  = 16
QK_NOPE  = 128
QK_ROPE  = 64
V_HEAD   = 128
KV_LORA  = 512

# Derived MLA dimensions
Q_DIM    = N_HEADS * (QK_NOPE + QK_ROPE)   # 3072
KV_A_DIM = KV_LORA + QK_ROPE               # 576
KV_B_DIM = N_HEADS * (QK_NOPE + V_HEAD)    # 4096
O_DIM    = N_HEADS * V_HEAD                 # 2048

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BLOCKSIZE = 128
PERCDAMP  = 0.01
NSAMPLES  = 8    # calibration samples per test
SEQLEN    = 32   # tokens per sample (short for speed)

# For the full handler integration test (test 8), use a small expert count so
# it completes in reasonable time.  The shape is still correct — only the
# count differs from production (64).
N_ROUTED_SMOKE = 8

QUANT_FORMAT   = "nvfp4"
BLOCK_SIZE_FP4 = 16


# ── Mock DeepSeek V2 Lite module hierarchy ────────────────────────────────────
#
# Class names must match what is_deepseek_v2_moe() checks:
#     type(layer.mlp).__name__ == "DeepseekV2MoE"
# so we name the class exactly that.

class DeepseekV2MLP(nn.Module):
    """Mock of DeepseekV2MLP (dense MLP — used for layer 0 and shared experts)."""
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden,       intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden,       intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden,       bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV2MoE(nn.Module):
    """Mock MoE MLP whose type name triggers is_deepseek_v2_moe == True."""
    def __init__(self, hidden: int, moe_inter: int, n_routed: int,
                 shared_inter: int):
        super().__init__()
        self.experts = nn.ModuleList([
            DeepseekV2MLP(hidden, moe_inter) for _ in range(n_routed)
        ])
        self.shared_experts = DeepseekV2MLP(hidden, shared_inter)

    def forward(self, x):
        # Non-routing forward — all experts see all tokens — sufficient for
        # populating hooks in tests. Routing logic irrelevant for shape tests.
        h = torch.zeros_like(x)
        for expert in self.experts:
            h = h + expert(x)
        return h + self.shared_experts(x)


class _MockMLA(nn.Module):
    """Stub MLA attention with the correct Linear shapes for all layers."""
    def __init__(self):
        super().__init__()
        self.q_proj             = nn.Linear(HIDDEN,   Q_DIM,    bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN,   KV_A_DIM, bias=False)
        self.kv_b_proj          = nn.Linear(KV_LORA,  KV_B_DIM, bias=False)
        self.o_proj             = nn.Linear(O_DIM,    HIDDEN,   bias=False)


class MockDenseLayer(nn.Module):
    """Layer 0: standard dense MLP, no MoE."""
    def __init__(self):
        super().__init__()
        self.self_attn = _MockMLA()
        self.mlp       = DeepseekV2MLP(HIDDEN, DENSE_INTER)


class MockMoELayer(nn.Module):
    """Layers 1-26: MLA attention + DeepseekV2MoE."""
    def __init__(self, n_routed: int = N_ROUTED):
        super().__init__()
        self.self_attn = _MockMLA()
        self.mlp       = DeepseekV2MoE(HIDDEN, MOE_INTER, n_routed, SHARED_INTER)


# ── Test utilities ─────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global _PASS, _FAIL
    tag = "PASS" if cond else "FAIL"
    msg = f"  {tag}  {name}"
    if not cond and detail:
        msg += f"\n         {detail}"
    print(msg)
    if cond:
        _PASS += 1
    else:
        _FAIL += 1


def _make_quantizer():
    return NVFP4Quantizer(block_size=BLOCK_SIZE_FP4, device=DEVICE)


def _quantize_linear(linear: nn.Linear,
                     nsamples: int = NSAMPLES,
                     seqlen: int = SEQLEN) -> float:
    """GPTQ-quantize a single nn.Linear with synthetic calibration. Returns loss."""
    linear = linear.to(DEVICE)
    g = GPTQ(linear)
    g.quantizer = _make_quantizer()

    in_features = linear.weight.shape[1]
    for _ in range(nsamples):
        x = torch.randn(seqlen, in_features, device=DEVICE)
        g.add_batch(x, None)

    loss = g.fasterquant_blockwise(blocksize=BLOCKSIZE, percdamp=PERCDAMP)
    g.free()
    linear.cpu()
    return loss


def _populate_gptq(g: GPTQ, nsamples: int = NSAMPLES, seqlen: int = SEQLEN):
    """Add synthetic calibration batches directly to a GPTQ instance."""
    in_features = g.layer.weight.shape[1]
    for _ in range(nsamples):
        x = torch.randn(seqlen, in_features, device=g.layer.weight.device)
        g.add_batch(x, None)


# ── Test 1: MLA attention shapes ──────────────────────────────────────────────

def test_mla_attention_shapes():
    print("\nTest 1 — MLA attention shapes")
    t0 = time.time()
    attn = _MockMLA().to(DEVICE)

    cases = [
        ("q_proj",             attn.q_proj,             HIDDEN,   Q_DIM),
        ("kv_a_proj_with_mqa", attn.kv_a_proj_with_mqa, HIDDEN,   KV_A_DIM),
        ("kv_b_proj",          attn.kv_b_proj,          KV_LORA,  KV_B_DIM),
        ("o_proj",             attn.o_proj,              O_DIM,    HIDDEN),
    ]

    for name, linear, in_f, out_f in cases:
        try:
            check(f"  shape  {name} [{in_f}→{out_f}]",
                  linear.weight.shape == (out_f, in_f),
                  f"got {tuple(linear.weight.shape)}")
            loss = _quantize_linear(linear)
            check(f"  GPTQ   {name} loss finite",
                  math.isfinite(loss), f"loss={loss}")
        except Exception as e:
            check(f"  GPTQ   {name}", False, traceback.format_exc(limit=1))

    attn.cpu()
    print(f"  [{time.time()-t0:.1f}s]")


# ── Test 2: Dense layer 0 MLP shapes ─────────────────────────────────────────

def test_dense_layer0_shapes():
    print("\nTest 2 — Dense layer 0 MLP shapes")
    t0 = time.time()
    mlp = DeepseekV2MLP(HIDDEN, DENSE_INTER).to(DEVICE)

    cases = [
        ("gate_proj", mlp.gate_proj, HIDDEN,       DENSE_INTER),
        ("up_proj",   mlp.up_proj,   HIDDEN,       DENSE_INTER),
        ("down_proj", mlp.down_proj, DENSE_INTER,  HIDDEN),
    ]

    for name, linear, in_f, out_f in cases:
        try:
            check(f"  shape  {name} [{in_f}→{out_f}]",
                  linear.weight.shape == (out_f, in_f),
                  f"got {tuple(linear.weight.shape)}")
            loss = _quantize_linear(linear)
            check(f"  GPTQ   {name} loss finite",
                  math.isfinite(loss), f"loss={loss}")
        except Exception as e:
            check(f"  GPTQ   {name}", False, traceback.format_exc(limit=1))

    mlp.cpu()
    print(f"  [{time.time()-t0:.1f}s]")


# ── Test 3: is_deepseek_v2_moe detection ─────────────────────────────────────

def test_moe_detection():
    print("\nTest 3 — is_deepseek_v2_moe detection")

    dense_layer = MockDenseLayer()
    moe_layer   = MockMoELayer()

    check("  dense layer (layer 0): is_deepseek_v2_moe == False",
          is_deepseek_v2_moe(dense_layer) == False)

    check("  MoE layer (layers 1-26): is_deepseek_v2_moe == True",
          is_deepseek_v2_moe(moe_layer) == True)

    check("  get_handler('deepseek_v2') returns DeepSeekV2Handler",
          isinstance(get_handler("deepseek_v2"), DeepSeekV2Handler))

    check("  get_handler('deepseek_v2').has_moe(moe_layer) == True",
          get_handler("deepseek_v2").has_moe(moe_layer) == True)

    check("  get_handler('deepseek_v2').has_moe(dense_layer) == False",
          get_handler("deepseek_v2").has_moe(dense_layer) == False)

    check("  get_handler('deepseek_v2').num_experts(moe_layer) == N_ROUTED",
          get_handler("deepseek_v2").num_experts(moe_layer) == N_ROUTED)


# ── Test 4: filter_standard_layers ───────────────────────────────────────────

def test_filter_standard_layers():
    print("\nTest 4 — filter_standard_layers")
    handler   = get_handler("deepseek_v2")
    moe_layer = MockMoELayer()

    raw_subset = find_layers(moe_layer)
    filtered   = handler.filter_standard_layers(moe_layer, raw_subset)

    # Raw subset should contain expert linears
    expert_names_raw = [n for n in raw_subset if "mlp.experts." in n
                        or "mlp.shared_experts." in n]
    check("  raw find_layers() finds expert linears",
          len(expert_names_raw) > 0,
          f"found {len(expert_names_raw)} expert names in raw subset")

    # Filtered subset must contain NO expert linears
    expert_names_filtered = [n for n in filtered if "mlp.experts." in n
                              or "mlp.shared_experts." in n]
    check("  filtered subset has 0 expert linears",
          len(expert_names_filtered) == 0,
          f"still found: {expert_names_filtered[:3]}")

    # Filtered subset must still contain attention linears
    attn_names = [n for n in filtered if "self_attn" in n]
    check("  filtered subset retains attention linears",
          len(attn_names) > 0,
          f"attention names found: {attn_names}")

    # Expected raw count: N_ROUTED experts × 3 + 1 shared × 3 + 4 attn = 195 + 4 = 199
    expected_expert_linears = (N_ROUTED + 1) * 3
    check(f"  raw expert linear count == {expected_expert_linears}",
          len(expert_names_raw) == expected_expert_linears,
          f"got {len(expert_names_raw)}")

    print(f"  raw subset: {len(raw_subset)} linears  "
          f"→  filtered: {len(filtered)} linears  "
          f"(removed {len(raw_subset) - len(filtered)} expert linears)")


# ── Test 5: Expert instance creation (OOM check) ─────────────────────────────

def test_expert_instance_creation():
    print("\nTest 5 — Expert instance creation (OOM / shape check)")
    t0    = time.time()
    layer = MockMoELayer().to(DEVICE)

    try:
        routed_gptq, shared_gptq = setup_expert_gptq_instances(
            layer, quant_format=QUANT_FORMAT, device=DEVICE,
            nvfp4_block_size=BLOCK_SIZE_FP4,
        )

        # Count
        n_routed = len(routed_gptq)
        check(f"  routed_gptq has {N_ROUTED} entries",
              n_routed == N_ROUTED, f"got {n_routed}")
        check("  shared_gptq is a 3-tuple",
              isinstance(shared_gptq, tuple) and len(shared_gptq) == 3)

        # Each routed entry is a 3-tuple
        check("  routed_gptq[0] is a 3-tuple of GPTQ",
              isinstance(routed_gptq[0], tuple) and len(routed_gptq[0]) == 3
              and all(isinstance(g, GPTQ) for g in routed_gptq[0]))

        # Verify shapes for expert 0 (gate/up: [2048,1408], down: [1408,2048])
        g_gate, g_up, g_dn = routed_gptq[0]
        check(f"  routed[0].gate_proj shape [MOE_INTER, HIDDEN] = [{MOE_INTER},{HIDDEN}]",
              g_gate.layer.weight.shape == (MOE_INTER, HIDDEN),
              f"got {tuple(g_gate.layer.weight.shape)}")
        check(f"  routed[0].up_proj   shape [MOE_INTER, HIDDEN] = [{MOE_INTER},{HIDDEN}]",
              g_up.layer.weight.shape == (MOE_INTER, HIDDEN),
              f"got {tuple(g_up.layer.weight.shape)}")
        check(f"  routed[0].down_proj shape [HIDDEN, MOE_INTER] = [{HIDDEN},{MOE_INTER}]",
              g_dn.layer.weight.shape == (HIDDEN, MOE_INTER),
              f"got {tuple(g_dn.layer.weight.shape)}")

        # Shared expert uses SHARED_INTER = 2 × MOE_INTER
        sg_gate, sg_up, sg_dn = shared_gptq
        check(f"  shared.gate_proj shape [SHARED_INTER, HIDDEN] = [{SHARED_INTER},{HIDDEN}]",
              sg_gate.layer.weight.shape == (SHARED_INTER, HIDDEN),
              f"got {tuple(sg_gate.layer.weight.shape)}")
        check(f"  shared.down_proj shape [HIDDEN, SHARED_INTER] = [{HIDDEN},{SHARED_INTER}]",
              sg_dn.layer.weight.shape == (HIDDEN, SHARED_INTER),
              f"got {tuple(sg_dn.layer.weight.shape)}")

        # Total GPTQ instances = N_ROUTED × 3 + 1 × 3
        total_instances = N_ROUTED * 3 + 3
        check(f"  total GPTQ instances = {total_instances} (no OOM)",
              True)   # reaching here means no OOM

        # Free all GPTQ instances
        for g_gate, g_up, g_dn in routed_gptq.values():
            for g in (g_gate, g_up, g_dn):
                g.free()
        for g in shared_gptq:
            g.free()

    except Exception:
        check("  setup_expert_gptq_instances", False, traceback.format_exc(limit=2))

    layer.cpu()
    torch.cuda.empty_cache()
    print(f"  [{time.time()-t0:.1f}s]")


# ── Test 6: Routed expert quantization shapes ─────────────────────────────────

def test_routed_expert_shapes():
    """Quantize a small subset of routed experts to verify gate/up/down shapes."""
    print(f"\nTest 6 — Routed expert quantization shapes (testing 4 of {N_ROUTED})")
    t0    = time.time()
    layer = MockMoELayer().to(DEVICE)

    # Only test 4 experts — enough to validate shapes, much faster than 64
    N_TEST = 4
    try:
        routed_gptq, _ = setup_expert_gptq_instances(
            layer, quant_format=QUANT_FORMAT, device=DEVICE,
            nvfp4_block_size=BLOCK_SIZE_FP4,
        )

        for e in range(N_TEST):
            g_gate, g_up, g_dn = routed_gptq[e]
            _populate_gptq(g_gate)
            _populate_gptq(g_up)
            _populate_gptq(g_dn)

            loss_gate = g_gate.fasterquant_blockwise(blocksize=BLOCKSIZE,
                                                      percdamp=PERCDAMP)
            loss_up   = g_up.fasterquant_blockwise(blocksize=BLOCKSIZE,
                                                    percdamp=PERCDAMP)
            loss_dn   = g_dn.fasterquant_blockwise(blocksize=BLOCKSIZE,
                                                    percdamp=PERCDAMP)

            check(f"  expert[{e:02d}] gate_proj loss finite",
                  math.isfinite(loss_gate), f"loss={loss_gate}")
            check(f"  expert[{e:02d}] up_proj   loss finite",
                  math.isfinite(loss_up), f"loss={loss_up}")
            check(f"  expert[{e:02d}] down_proj loss finite",
                  math.isfinite(loss_dn), f"loss={loss_dn}")

            for g in (g_gate, g_up, g_dn):
                g.free()

        # Free remaining instances
        for e in range(N_TEST, N_ROUTED):
            for g in routed_gptq[e]:
                g.free()

    except Exception:
        check("  routed expert quantization", False, traceback.format_exc(limit=2))

    layer.cpu()
    torch.cuda.empty_cache()
    print(f"  [{time.time()-t0:.1f}s]")


# ── Test 7: Shared expert quantization shapes ─────────────────────────────────

def test_shared_expert_shapes():
    """Shared expert uses 2× intermediate — verify shape and loss."""
    print(f"\nTest 7 — Shared expert shapes  "
          f"(gate/up [{HIDDEN}→{SHARED_INTER}], down [{SHARED_INTER}→{HIDDEN}])")
    t0    = time.time()
    layer = MockMoELayer().to(DEVICE)

    try:
        _, shared_gptq = setup_expert_gptq_instances(
            layer, quant_format=QUANT_FORMAT, device=DEVICE,
            nvfp4_block_size=BLOCK_SIZE_FP4,
        )

        names = ["gate_proj", "up_proj", "down_proj"]
        for name, g in zip(names, shared_gptq):
            in_f  = g.layer.weight.shape[1]
            out_f = g.layer.weight.shape[0]
            _populate_gptq(g)
            loss = g.fasterquant_blockwise(blocksize=BLOCKSIZE, percdamp=PERCDAMP)
            check(f"  shared.{name} [{in_f}→{out_f}] loss finite",
                  math.isfinite(loss), f"loss={loss}")
            g.free()

    except Exception:
        check("  shared expert quantization", False, traceback.format_exc(limit=2))

    layer.cpu()
    torch.cuda.empty_cache()
    print(f"  [{time.time()-t0:.1f}s]")


# ── Test 8: Full MoE layer integration smoke ──────────────────────────────────

def test_full_moe_layer_smoke():
    """Full handler lifecycle: setup → hooks → synthetic forward → quantize.

    Uses N_ROUTED_SMOKE experts (< N_ROUTED) for speed. All code paths are
    exercised; only the expert count differs from production.
    """
    print(f"\nTest 8 — Full MoE layer smoke  "
          f"({N_ROUTED_SMOKE} routed experts + shared)")
    t0      = time.time()
    handler = get_handler("deepseek_v2")
    layer   = MockMoELayer(n_routed=N_ROUTED_SMOKE).to(DEVICE)

    try:
        # Phase 1: setup accumulators
        acc_state = handler.setup_accumulators(
            layer, DEVICE, quant_format=QUANT_FORMAT,
            nvfp4_block_size=BLOCK_SIZE_FP4,
        )
        check("  setup_accumulators OK",
              "routed_gptq" in acc_state and "shared_gptq" in acc_state)

        # Phase 2: attach hooks
        hook_token = handler.attach_hooks(layer, acc_state)
        check("  attach_hooks returns list of handles",
              isinstance(hook_token, list) and len(hook_token) > 0,
              f"got {len(hook_token)} handles")

        # Phase 3: synthetic forward pass to populate Hessians
        with torch.no_grad():
            for _ in range(NSAMPLES):
                x = torch.randn(SEQLEN, HIDDEN, device=DEVICE)
                layer.mlp(x)

        # Check at least some Hessians were populated
        n_populated = sum(
            1 for g_gate, g_up, g_dn in acc_state["routed_gptq"].values()
            for g in (g_gate, g_up, g_dn)
            if g.nsamples > 0
        )
        check(f"  hooks collected data for ≥1 projection (got {n_populated})",
              n_populated > 0)

        # Phase 4: detach hooks
        handler.detach_hooks(layer, hook_token)
        check("  detach_hooks runs without error", True)

        # Phase 5: quantize
        expert_losses = handler.quantize(
            layer, acc_state,
            quant_format=QUANT_FORMAT, device=DEVICE,
            nvfp4_block_size=BLOCK_SIZE_FP4,
            blocksize=BLOCKSIZE, percdamp=PERCDAMP,
            threshold=None,
        )
        check("  quantize returns dict with 'routed' and 'shared' keys",
              "routed" in expert_losses and "shared" in expert_losses)

        # Every routed expert should have 3 projection losses
        n_routed_losses = len(expert_losses["routed"])
        check(f"  losses for {N_ROUTED_SMOKE} routed experts",
              n_routed_losses == N_ROUTED_SMOKE,
              f"got {n_routed_losses}")

        n_shared_losses = len(expert_losses["shared"])
        check("  losses for shared expert (3 projections)",
              n_shared_losses == 3, f"got {n_shared_losses}")

        # summarize_losses should run cleanly
        n_q, n_bf16, n_rtn, avg_a, avg_b = handler.summarize_losses(expert_losses)
        check(f"  summarize_losses: n_q={n_q}, n_bf16={n_bf16}, n_rtn={n_rtn}",
              n_q + n_bf16 + n_rtn > 0)

    except Exception:
        check("  full MoE smoke", False, traceback.format_exc(limit=3))

    layer.cpu()
    torch.cuda.empty_cache()
    print(f"  [{time.time()-t0:.1f}s]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("Stage 3 — DeepSeek V2 Lite Shape Tests")
    print(f"Device: {DEVICE}")
    print(f"Architecture: hidden={HIDDEN}, moe_inter={MOE_INTER}, "
          f"dense_inter={DENSE_INTER}")
    print(f"Experts: {N_ROUTED} routed × 3 proj + 1 shared × 3 proj "
          f"(shared_inter={SHARED_INTER})")
    print("=" * 68)

    if DEVICE == "cpu":
        print("WARNING: Running on CPU — tests will be very slow. "
              "Set CUDA device for production use.\n")

    t_start = time.time()

    test_mla_attention_shapes()
    test_dense_layer0_shapes()
    test_moe_detection()
    test_filter_standard_layers()
    test_expert_instance_creation()
    test_routed_expert_shapes()
    test_shared_expert_shapes()
    test_full_moe_layer_smoke()

    elapsed = time.time() - t_start

    print()
    print("=" * 68)
    print(f"Results: {_PASS} passed, {_FAIL} failed  "
          f"(total {elapsed:.1f}s)")
    print("=" * 68)

    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()