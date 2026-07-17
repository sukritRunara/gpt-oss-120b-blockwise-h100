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


if __name__ == "__main__":
    sys.exit(main())
