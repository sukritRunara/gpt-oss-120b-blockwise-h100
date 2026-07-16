#!/usr/bin/env python
"""P0.10 microscope v2: GEMM1-only replay via raw ops.moe_wna16_marlin_gemm
with exact per-row reference. Env knobs:
    R_FP32_REDUCE=0|1 (default 1, kernel's hardcoded value)
    R_ATOMIC=0|1      (default 0)
Run inside .venv-serve."""

import json
import os
import sys

import torch

sys.path.insert(0, "/workspace/blockwise-gptq-main/opteam-blockwise-gptq")


@torch.no_grad()
def main():
    from vllm import _custom_ops as ops
    from vllm.scalar_type import ScalarType
    from safetensors import safe_open

    blob = torch.load("/workspace/results/pilot/marlin_call_capture.pt",
                      map_location="cuda", weights_only=False)
    hs = blob["hidden_states"]
    M, K = hs.shape
    topk = blob["num_topk"]
    kw = blob["kw"]
    w1 = blob["w1"]
    quant_type = ScalarType.from_id(blob["quant_type_id"])
    N2 = w1.size(2) * 16 // K if False else None
    # size_n from capture context: w13 shards → derive from scales
    size_n = blob["w1_scale"].shape[-1] * 0 + 2 * (w1.shape[2] * 8 // K) * 0
    # simpler: marlin w1 [E, K/16, size_n*2]: size_n = w1.shape[2] // 2
    size_n = w1.shape[2] // 2

    fill = os.environ.get("R_CACHE_FILL", "zeros")
    if fill == "zeros":
        cache1 = torch.zeros(M * topk, size_n, device="cuda", dtype=hs.dtype)
    elif fill == "poison":
        cache1 = torch.full((M * topk, size_n), 12345.0, device="cuda", dtype=hs.dtype)
    else:
        cache1 = torch.empty(M * topk, size_n, device="cuda", dtype=hs.dtype)

    out = ops.moe_wna16_marlin_gemm(
        hs, cache1, w1, blob["bias1"], blob["w1_scale"], None,
        kw.get("global_scale1"), kw.get("w1_zeros"), kw.get("g_idx1"),
        kw.get("sort_indices1"),
        torch.zeros(1024, dtype=torch.int, device="cuda"),
        blob["sorted_token_ids"], blob["expert_ids"],
        blob["num_tokens_post_padded"], blob["topk_weights"],
        moe_block_size=blob["block_size_m"], top_k=topk,
        mul_topk_weights=blob["apply_router_weight_on_input"],
        b_q_type=quant_type, size_m=M, size_n=size_n, size_k=K,
        is_k_full=True,
        use_atomic_add=os.environ.get("R_ATOMIC", "0") == "1",
        use_fp32_reduce=os.environ.get("R_FP32_REDUCE", "1") == "1",
        is_zp_float=False,
    )
    o = out.float().cpu()

    # Exact reference for gemm1 rows: x_t @ gate_up_deint[e] + bias_deint[e]
    qdq = "/workspace/models/fixture-real2l-qdq"
    idx = json.load(open(f"{qdq}/model.safetensors.index.json"))

    def load(key):
        with safe_open(f"{qdq}/{idx['weight_map'][key]}", framework="pt",
                       device="cpu") as f:
            return f.get_tensor(key).float()

    gu = load("model.layers.0.mlp.experts.gate_up_proj")        # [E, K, 2I]
    gub = load("model.layers.0.mlp.experts.gate_up_proj_bias")  # [E, 2I]
    twoI = gu.shape[2]
    deint = torch.cat([torch.arange(0, twoI, 2), torch.arange(1, twoI, 2)])
    gu = gu[:, :, deint]
    gub = gub[:, deint]
    # padded size_n vs logical 2I: kernel output has padded_N per shard
    I = twoI // 2
    padN = size_n // 2

    st = blob["sorted_token_ids"].cpu()
    ei = blob["expert_ids"].cpu()
    bm = blob["block_size_m"]
    hsc = hs.float().cpu()
    n_valid = M * topk

    checked, bad, maxrel = 0, 0, 0.0
    for b in range(len(ei)):
        e = ei[b].item()
        if e < 0:
            continue
        for s in range(bm):
            stid = st[b * bm + s].item()
            if stid >= n_valid:
                continue
            t = stid // topk
            ref_row = hsc[t] @ gu[e] + gub[e]                  # [2I] deint
            got = o[stid]                                       # [2*padN]
            got_logical = torch.cat([got[:I], got[padN:padN + I]])
            rel = (got_logical - ref_row).norm() / (ref_row.norm() + 1e-9)
            maxrel = max(maxrel, rel.item())
            checked += 1
            if rel > 0.05:
                bad += 1
                if bad <= 3:
                    print(f"  BAD row stid={stid} t={t} e={e} rel={rel:.3f} "
                          f"got_absmax={got_logical.abs().max():.3e} "
                          f"ref_absmax={ref_row.abs().max():.3e}")
    print(f"GEMM1 rows checked={checked} bad(>5%)={bad} maxrel={maxrel:.4f} "
          f"out_absmax={o.abs().max():.3e} "
          f"fp32_reduce={os.environ.get('R_FP32_REDUCE', '1')} "
          f"atomic={os.environ.get('R_ATOMIC', '0')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
