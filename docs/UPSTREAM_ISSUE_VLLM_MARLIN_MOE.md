# UPSTREAM ISSUE — READY TO FILE (blocked only on token scope)

Target: https://github.com/vllm-project/vllm/issues/new (Bug Report template)

**Status 2026-07-17:** body finalized below (verbatim); repro validated on this pod;
kernel confirmed byte-identical v0.25.1 ↔ main; duplicate search clean (#46641 is the
FlashInfer/Blackwell path, distinct). Filing FAILED with the session's fine-grained PAT
(cannot create issues on repos it isn't granted). To file: paste title+body below in the
browser, or supply a classic PAT with public_repo scope.

## Title

```
[Bug]: moe_wna16_marlin_gemm applies wrong per-row topk weights (mul_topk_weights=True) at gpt-oss NVFP4 MoE shapes — corrupt output
```

## Body (paste verbatim)

### Your current environment

- vLLM **0.25.1** (pip wheel), 1× **H100 80GB (SM90)**, torch 2.9.0, CUDA 12.8, Python 3.12
- The affected kernel source (`csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h`) is **byte-identical between v0.25.1 and current `main`**, so this should still reproduce on main.
- Model: `openai/gpt-oss-20b` requantized to ModelOpt-style **NVFP4 W4A16** (`quant_algo: NVFP4`, group 16), served on Hopper via the Marlin FP4 MoE fallback (`fused_marlin_moe` → `ops.moe_wna16_marlin_gemm`). MoE dims: E=32 experts, top_k=4, hidden=2880, intermediate=2880.

### 🐛 Describe the bug

**When `mul_topk_weights=True`, `moe_wna16_marlin_gemm` applies the WRONG per-row routing weight at these shapes** — each output row is multiplied by *a different row's* `topk_weights` entry, by 0.0, or (in-engine) by garbage read from shared memory. End-to-end this makes gpt-oss-20b NVFP4 on Hopper generate garbage (huge/NaN logits → repeated token; we observed intermediate values ~1e33 = output × a garbage bf16 multiplier).

The same call with `mul_topk_weights=False` (and the multiply applied externally on the output rows) is correct to kernel noise — with bit-identical weights, activations, and scheduling metadata.

#### Isolation trail (standalone replay of a captured failing `fused_marlin_moe` call, exact fp32 reference)

| variant | result |
|---|---|
| gemm1 (w13), as captured | all rows match (maxrel ≤ 3e-3) |
| gemm2 (w2), as captured (`top_k=1, mul_topk_weights=True` — the config `fused_marlin_moe` uses when `apply_router_weight_on_input=False`) | **128/128 rows corrupt, out_absmax ≈ 4e+32** |
| gemm2, same args, `mul_topk_weights=False` + external multiply | all rows match (maxrel 2.4e-3) |
| sweeps: thread configs, `use_atomic_add`, `use_fp32_reduce`, moe_block_size | failure unaffected |
| random weight values instead of real checkpoint | still fails → value-independent |
| small shapes (e.g. E=8, N=K=512) | pass → layout/shape-dependent |

#### Forensic: the wrong multiplier is another row's topk weight

In the minimal repro below, dividing each corrupted output row by its (bias-free) unweighted reference reveals the factor that was actually applied. It is **another row's routing weight, bf16-exact, or 0.0**:

```
row | applied_ratio | own_tw   | ratio matches tw of row#
  0 | +0.146432     | 0.192106 | row 1   (tw=0.146532, err=1.0e-04)
 11 | +0.067504     | 0.065274 | row 66  (tw=0.067503, err=1.6e-06)
 22 | +0.083851     | 0.093152 | row 114 (tw=0.083904, err=5.2e-05)
 44 | +0.081211     | 0.170854 | row 93  (tw=0.081133, err=7.9e-05)
 66..121 | +0.000000 | (varies) | unmatched — zero/stale
```

This points at the epilogue's shared-memory read `topk_weight_score = sh_block_topk_weights[row]` (marlin_template.h ~L1879, populated at ~L514-527 indexed by `threadIdx.x` over the moe block's sorted slots): at these shapes the output-row index used in the epilogue does not correspond to the slot the weight was stored under. Inside the engine the same wrong read lands on clobbered shared memory (the epilogue reuses `sh_red`/adjacent buffers), which is where the ~1e33 values come from. Possibly related to the static-analysis OOB report in this kernel family: #27915.

#### Minimal repro (self-contained, random weights, no checkpoint needed)

Requires only vLLM 0.25.1 + a Hopper GPU:

```python
#!/usr/bin/env python
"""Self-contained repro: moe_wna16_marlin_gemm corrupts output when
mul_topk_weights=True at gpt-oss-20b MoE shapes (NVFP4 W4A16, group 16).

Random weights (failure is value-independent). Mirrors vLLM 0.25.1's own
fused_marlin_moe gemm2 configuration for gpt-oss-20b served as ModelOpt
NVFP4: E=32 experts, top_k=4, N(out)=2880, K(reduce)=2880 padded to 2944.

Expected on H100 / vLLM 0.25.1 (kernel source identical on main):
    mul_topk_weights=True : most rows corrupt, |out| up to ~1e33
    mul_topk_weights=False + external multiply: all rows match reference

Run: python upstream_repro_marlin_moe_topk.py
"""

import sys

import torch

E, N_OUT, K_RAW, K_PAD = 32, 2880, 2880, 2944   # gemm2: w2 [E, N_OUT, K]
M, TOPK, BLOCK_M = 32, 4, 16
GROUP = 16

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


@torch.no_grad()
def main():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale,
        _nvfp4_compute_scale_factor)
    from vllm.scalar_type import scalar_types

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16

    # ── Random NVFP4 weights in ModelOpt checkpoint layout ────────────────
    # codes: two E2M1 nibbles per byte along K (low nibble = even index)
    codes = torch.randint(0, 256, (E, N_OUT, K_PAD // 2),
                          dtype=torch.uint8, device=dev)
    codes[..., K_RAW // 2:] = 0                       # zero-pad K tail
    scale_f = torch.rand(E, N_OUT, K_PAD // GROUP, device=dev) * 3 + 0.5
    scale_f[..., K_RAW // GROUP:] = 0
    scales_fp8 = scale_f.to(torch.float8_e4m3fn)
    gscale = torch.rand(E, device=dev) * 0.002 + 1e-4  # per-expert global

    # Dequantized reference weights [E, N_OUT, K_PAD] in fp32
    lo = E2M1_LUT.to(dev)[(codes & 0xF).long()]
    hi = E2M1_LUT.to(dev)[(codes >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).reshape(E, N_OUT, K_PAD)
    w = w * scales_fp8.float().repeat_interleave(GROUP, dim=2) \
          * gscale.view(E, 1, 1)

    # ── Marlin repack (the exact prepare-for-marlin path for MoE w2) ─────
    perm = torch.empty(0, dtype=torch.int, device=dev)
    qw, ms = [], []
    csf = _nvfp4_compute_scale_factor(scales_fp8.to(dt), dt)
    for e in range(E):
        # kernel wants int32 [size_k/8, size_n]; codes[e] is [N, K/2] uint8
        q = codes[e].view(torch.int32).T.contiguous()  # -> [K/8, N]
        qw.append(ops.gptq_marlin_repack(
            b_q_weight=q, perm=perm, size_k=K_PAD, size_n=N_OUT,
            num_bits=4, is_a_8bit=False))
        s = marlin_permute_scales(s=scales_fp8[e].to(dt).T, size_k=K_PAD,
                                  size_n=N_OUT, group_size=GROUP,
                                  is_a_8bit=False)
        s, _ = nvfp4_marlin_process_scales(s, scale_factor=csf, a_dtype=dt)
        ms.append(s)
    w_marlin, s_marlin = torch.stack(qw), torch.stack(ms)
    g_marlin = nvfp4_marlin_process_global_scale(gscale.float(), dt) / csf

    # ── Routing: M tokens × top-4 experts, rows pre-expanded (gemm2 style) ─
    router = torch.randn(M, E, device=dev)
    topk_w, topk_ids = torch.topk(torch.softmax(router, -1), TOPK, dim=-1)
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids.to(torch.int32), BLOCK_M, E)
    act = torch.randn(M * TOPK, K_PAD, device=dev, dtype=dt) * 0.5
    act[:, K_RAW:] = 0
    tw_flat = topk_w.reshape(-1).float()

    # fp32 reference: out[row] = (act[row] @ w[e].T + 0) * topk_w[row]
    row_expert = torch.full((M * TOPK,), -1, dtype=torch.long, device=dev)
    for b in range(len(expert_ids)):
        if expert_ids[b] < 0:
            continue
        blk = sorted_ids[b * BLOCK_M:(b + 1) * BLOCK_M]
        for sid in blk[blk < M * TOPK]:
            row_expert[sid] = expert_ids[b]
    ref = torch.zeros(M * TOPK, N_OUT, device=dev)
    for r in range(M * TOPK):
        e = row_expert[r].item()
        if e >= 0:
            ref[r] = (act[r].float() @ w[e].T) * tw_flat[r]

    def gemm(mul_in_kernel):
        c = torch.zeros(M * TOPK, N_OUT, device=dev, dtype=dt)
        out = ops.moe_wna16_marlin_gemm(
            act, c, w_marlin, None, s_marlin, None, g_marlin,
            None, None, None,
            torch.zeros(1024, dtype=torch.int, device=dev),
            sorted_ids, expert_ids, num_post_pad,
            topk_w.to(dt), moe_block_size=BLOCK_M, top_k=1,
            mul_topk_weights=mul_in_kernel,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=M * TOPK, size_n=N_OUT, size_k=K_PAD, is_k_full=True,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        if not mul_in_kernel:
            out = out * tw_flat.view(-1, 1).to(out.dtype)
        return out.float()

    for label, mul in (("mul_topk_weights=True (vLLM's gemm2 config)", True),
                       ("mul_topk_weights=False + external multiply", False)):
        o = gemm(mul)
        bad = maxrel = 0
        for r in range(M * TOPK):
            if row_expert[r] < 0:
                continue
            rel = ((o[r] - ref[r]).norm() / (ref[r].norm() + 1e-9)).item()
            maxrel = max(maxrel, rel)
            bad += rel > 0.05
        print(f"{label}: bad_rows(>5% rel err)={bad}/{M*TOPK} "
              f"maxrel={maxrel:.3e} out_absmax={o.abs().max():.3e}")
    return 0


```

Output on H100, vLLM 0.25.1:

```
mul_topk_weights=True (vLLM's gemm2 config): bad_rows(>5% rel err)=117/128 maxrel=1.377e+00 out_absmax=2.812e-01
mul_topk_weights=False + external multiply:  bad_rows(>5% rel err)=0/128   maxrel=4.163e-03 out_absmax=4.258e-01
```

(In-engine the corruption is far larger — ~1e33 — because the misread shared-memory slot holds garbage rather than a neighbor's weight; the flag-flip contrast is identical.)

#### Workaround we're shipping

Wrap `fused_marlin_moe`'s gemm2 call: run the kernel with `mul_topk_weights=False` and apply routing weights externally —

```python
out = ops.moe_wna16_marlin_gemm(..., mul_topk_weights=False, ...)
out.mul_(topk_weights.reshape(-1, 1).to(out.dtype))
```

With this workaround, our full gpt-oss-20b NVFP4 W4A16 pack serves correctly on H100 (deterministic, greedy-64 agreement 0.869 vs its QDQ reference, coherent chat, zero failures across benchmark suites).

#### Related issues

- #46641 — same model+format on Blackwell via the FlashInfer CUTLASS/TRTLLM path (different backend, different gaps); this issue is the Hopper/Marlin fallback path.
- #27915 — static-analysis report of potential OOB in `moe_wna16`/`marlin_moe_wna16` (closed stale); the forensics above look like a concrete instance of a mis-indexed shared read in that family.
- #39549 — borderline tolerance failure in `test_fused_marlin_moe` at odd m; possibly the same row-index issue at sub-corruption magnitude.

Full isolation trail, capture/replay harness, and a 2-layer real-weight repro checkpoint are available in https://github.com/sukritRunara/gpt-oss-120b-blockwise-h100 (see `KNOWN_ISSUES.md` P0.10 and `scripts/marlin_replay*.py`).

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.
