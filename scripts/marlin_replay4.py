#!/usr/bin/env python
"""P0.10 fix validation: re-repack w2 with output dim padded 2880→3072
(multiple of 256, so marlin dispatch avoids the broken {128,64} config)
and check gemm2 rows against the exact reference. Run inside .venv-serve."""

import json
import sys

import torch

sys.path.insert(0, "/workspace/blockwise-gptq-main/opteam-blockwise-gptq")


@torch.no_grad()
def main():
    from vllm import _custom_ops as ops
    from vllm.scalar_type import ScalarType
    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales, marlin_permute_bias,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale,
        _nvfp4_compute_scale_factor,
    )

    blob = torch.load("/workspace/results/pilot/marlin_call_capture.pt",
                      map_location="cuda", weights_only=False)
    hs = blob["hidden_states"]
    M, K = hs.shape
    topk = blob["num_topk"]
    quant_type = ScalarType.from_id(blob["quant_type_id"])
    PAD_TO = 256
    padded_K = (K + PAD_TO - 1) // PAD_TO * PAD_TO      # 3072

    # ── Load checkpoint-layout w2 (HF orientation, pre-marlin) ────────────────
    with safe_open("/workspace/models/fixture-real2l-packed/model.safetensors",
                   framework="pt", device="cpu") as f:
        w2_ck = f.get_tensor("model.layers.0.mlp.experts.w2_weight")        # [E, N/2? , K]?? -> checkpoint [E, I/2, K]
        w2s_ck = f.get_tensor("model.layers.0.mlp.experts.w2_weight_scale") # [E, I/16, K]
        g2_ck = f.get_tensor("model.layers.0.mlp.experts.w2_weight_scale_2")  # [E]
        b2_ck = f.get_tensor("model.layers.0.mlp.experts.down_proj_bias")   # [E, K]
    # vLLM loader permutes (0,2,1): params become [E, K, I/2] etc.
    w2_p = w2_ck.permute(0, 2, 1).contiguous().cuda()      # [E, K, I/2]
    w2s_p = w2s_ck.permute(0, 2, 1).contiguous().cuda()    # [E, K, I/16]
    b2 = b2_ck.cuda()
    E = w2_p.shape[0]
    I = w2_p.shape[2] * 2
    print(f"checkpoint w2 param layout {tuple(w2_p.shape)} E={E} I={I} K={K} → padded_K {padded_K}")

    # ── Pad output dim K → padded_K with zero rows ────────────────────────────
    def padK(t):
        pad = torch.zeros(E, padded_K - K, t.shape[2], dtype=t.dtype,
                          device=t.device)
        return torch.cat([t, pad], dim=1)

    w2_p = padK(w2_p)
    w2s_p = padK(w2s_p)
    b2 = torch.cat([b2, torch.zeros(E, padded_K - K, dtype=b2.dtype,
                                    device=b2.device)], dim=1)

    # ── Marlin repack (mirrors prepare_nvfp4_moe_layer_for_marlin's w2 branch,
    #    with size_n = padded_K) ────────────────────────────────────────────────
    N_pad = 2944          # gemm2 reduction dim (w13 padded_N from the pipeline)
    # pad packed N dim 1440 → 1472 (N 2880 → 2944), packing=2
    padN2 = torch.zeros(E, padded_K, (N_pad - I) // 2, dtype=w2_p.dtype,
                        device=w2_p.device)
    w2_p = torch.cat([w2_p, padN2], dim=2)                 # [E, padK, N_pad/2]
    padS = torch.zeros(E, padded_K, (N_pad - I) // 16, dtype=w2s_p.dtype,
                       device=w2s_p.device)
    w2s_p = torch.cat([w2s_p, padS], dim=2)                # [E, padK, N_pad/16]

    perm = torch.empty(0, dtype=torch.int, device="cuda")
    tl = []
    for i in range(E):
        qw = w2_p[i].view(torch.int32).T.contiguous()
        tl.append(ops.gptq_marlin_repack(
            b_q_weight=qw, perm=perm, size_k=N_pad, size_n=padded_K,
            num_bits=4, is_a_8bit=False))
    w2_marlin = torch.stack(tl)

    param_dtype = torch.bfloat16
    scales = w2s_p.to(param_dtype)
    csf = _nvfp4_compute_scale_factor(scales, param_dtype)
    tl = []
    for i in range(E):
        ms = marlin_permute_scales(s=scales[i].T, size_k=N_pad,
                                   size_n=padded_K, group_size=16,
                                   is_a_8bit=False)
        ms, _ = nvfp4_marlin_process_scales(ms, scale_factor=csf,
                                            a_dtype=param_dtype)
        tl.append(ms)
    w2s_marlin = torch.stack(tl)
    g2 = nvfp4_marlin_process_global_scale(g2_ck.cuda().float(), param_dtype)
    g2 = g2 / csf
    b2_marlin = torch.stack([marlin_permute_bias(b2[i]) for i in range(E)])

    # ── Exact activation input + reference (same as replay3) ─────────────────
    qdq = "/workspace/models/fixture-real2l-qdq"
    idx = json.load(open(f"{qdq}/model.safetensors.index.json"))

    def load(key):
        with safe_open(f"{qdq}/{idx['weight_map'][key]}", framework="pt",
                       device="cpu") as f:
            return f.get_tensor(key).float()

    gu = load("model.layers.0.mlp.experts.gate_up_proj")
    gub = load("model.layers.0.mlp.experts.gate_up_proj_bias")
    dn = load("model.layers.0.mlp.experts.down_proj")
    dnb = load("model.layers.0.mlp.experts.down_proj_bias")
    Ii = gu.shape[2] // 2

    st = blob["sorted_token_ids"].cpu()
    ei = blob["expert_ids"].cpu()
    bm = blob["block_size_m"]
    hsc = hs.float().cpu()
    n_valid = M * topk
    tw = blob["topk_weights"].float().cpu().reshape(-1)

    cache2 = torch.zeros(M * topk, N_pad, dtype=torch.float32)
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
            act = (up.clamp(-7, 7) + 1) * (gate.clamp(max=7)
                                           * torch.sigmoid(gate.clamp(max=7) * 1.702))
            cache2[stid, :Ii] = act
            ref_rows[stid] = (act @ dn[e] + dnb[e]) * tw[stid]
    cache2 = cache2.to(dtype=hs.dtype, device="cuda")
    cache3 = torch.zeros(M * topk, padded_K, device="cuda", dtype=hs.dtype)

    out = ops.moe_wna16_marlin_gemm(
        cache2, cache3, w2_marlin, b2_marlin, w2s_marlin, None,
        g2, None, None, None,
        torch.zeros(1024, dtype=torch.int, device="cuda"),
        blob["sorted_token_ids"], blob["expert_ids"],
        blob["num_tokens_post_padded"], blob["topk_weights"],
        moe_block_size=bm, top_k=1, mul_topk_weights=True,
        b_q_type=quant_type, size_m=M * topk, size_n=padded_K, size_k=N_pad,
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
        is_zp_float=False)
    o = out.float().cpu()

    checked, bad, maxrel = 0, 0, 0.0
    for stid, ref in ref_rows.items():
        got = o[stid][:K]
        rel = ((got - ref).norm() / (ref.norm() + 1e-9)).item()
        maxrel = max(maxrel, rel)
        checked += 1
        bad += int(rel > 0.05)
    print(f"GEMM2-PADDED rows checked={checked} bad(>5%)={bad} "
          f"maxrel={maxrel:.4f} out_absmax={o.abs().max():.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
