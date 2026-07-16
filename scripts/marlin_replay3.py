#!/usr/bin/env python
"""P0.10 microscope v3: GEMM2-only replay. Feed an EXACT activation input
(computed in torch from ground truth) through the w2 marlin gemm and compare
per-token outputs against the exact reference. Run inside .venv-serve."""

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
    w2 = blob["w2"]                       # [E, K, padded_N/2] marlin-packed
    quant_type = ScalarType.from_id(blob["quant_type_id"])
    padded_N = w2.shape[2] * 16 // K if False else None
    # marlin w2 layout [E, size_k/16, size_n*2]: size_k=padded_N, size_n=K
    size_k2 = w2.shape[1] * 16
    size_n2 = w2.shape[2] // 2
    print(f"w2 marlin shape {tuple(w2.shape)} → size_k={size_k2} size_n={size_n2}")

    qdq = "/workspace/models/fixture-real2l-qdq"
    idx = json.load(open(f"{qdq}/model.safetensors.index.json"))

    def load(key):
        with safe_open(f"{qdq}/{idx['weight_map'][key]}", framework="pt",
                       device="cpu") as f:
            return f.get_tensor(key).float()

    gu = load("model.layers.0.mlp.experts.gate_up_proj")
    gub = load("model.layers.0.mlp.experts.gate_up_proj_bias")
    dn = load("model.layers.0.mlp.experts.down_proj")           # [E, I, K]
    dnb = load("model.layers.0.mlp.experts.down_proj_bias")     # [E, K]
    twoI = gu.shape[2]
    I = twoI // 2

    st = blob["sorted_token_ids"].cpu()
    ei = blob["expert_ids"].cpu()
    bm = blob["block_size_m"]
    hsc = hs.float().cpu()
    n_valid = M * topk
    tw = blob["topk_weights"].float().cpu().reshape(-1)

    # Build EXACT activation rows (cache2) and the per-row reference output
    padN_in = size_k2                      # gemm2 reduction width
    cache2 = torch.zeros(M * topk, padN_in, dtype=torch.float32)
    ref_rows = {}
    for b in range(len(ei)):
        e = ei[b].item()
        if e < 0:
            continue
        for s in range(bm):
            stid = st[b * bm + s].item()
            if stid >= n_valid:
                continue
            t = stid // topk
            g_u = hsc[t] @ gu[e] + gub[e]
            gate, up = g_u[::2], g_u[1::2]
            gate = gate.clamp(max=7.0)
            up = up.clamp(min=-7.0, max=7.0)
            act = (up + 1) * (gate * torch.sigmoid(gate * 1.702))   # [I]
            cache2[stid, :I] = act                                  # zero-pad
            ref_rows[stid] = (act @ dn[e] + dnb[e]) * tw[stid]

    cache2_gpu = cache2.to(dtype=hs.dtype, device="cuda")
    cache3 = torch.zeros(M * topk, size_n2, device="cuda", dtype=hs.dtype)

    out = ops.moe_wna16_marlin_gemm(
        cache2_gpu, cache3, w2, blob["bias2"], blob["w2_scale"], None,
        kw.get("global_scale2"), kw.get("w2_zeros"), kw.get("g_idx2"),
        kw.get("sort_indices2"),
        torch.zeros(1024, dtype=torch.int, device="cuda"),
        blob["sorted_token_ids"], blob["expert_ids"],
        blob["num_tokens_post_padded"], blob["topk_weights"],
        moe_block_size=blob["block_size_m"], top_k=1,
        mul_topk_weights=not blob["apply_router_weight_on_input"],
        b_q_type=quant_type, size_m=M * topk, size_n=size_n2, size_k=size_k2,
        is_k_full=True,
        use_atomic_add=os.environ.get("R_ATOMIC", "0") == "1",
        use_fp32_reduce=os.environ.get("R_FP32_REDUCE", "1") == "1",
        is_zp_float=False,
    )
    o = out.float().cpu()

    # Per-expert accuracy pattern
    stid_expert = {}
    for b in range(len(ei)):
        e = ei[b].item()
        if e < 0:
            continue
        for s in range(bm):
            stid = st[b * bm + s].item()
            if stid < n_valid:
                stid_expert[stid] = e
    by_e = {}
    checked, bad, maxrel = 0, 0, 0.0
    for stid, ref in ref_rows.items():
        got = o[stid][:K]
        rel = ((got - ref).norm() / (ref.norm() + 1e-9)).item()
        maxrel = max(maxrel, rel)
        checked += 1
        bad += int(rel > 0.05)
        e = stid_expert[stid]
        by_e.setdefault(e, []).append(rel)
    for e in sorted(by_e):
        rels = by_e[e]
        print(f"  expert {e:2d}: n={len(rels)} rel_mean={sum(rels)/len(rels):.4g} rel_max={max(rels):.4g}")
    print(f"GEMM2 rows checked={checked} bad(>5%)={bad} maxrel={maxrel:.4f} "
          f"out_absmax={o.abs().max():.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
