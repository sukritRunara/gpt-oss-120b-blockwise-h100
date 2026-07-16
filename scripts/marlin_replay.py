#!/usr/bin/env python
"""Standalone replay of a captured _fused_marlin_moe call (P0.10 microscope).

Loads the captured arguments, replays the exact kernel call outside the
engine, and compares against an exact ground-truth MoE computed in plain
torch from the QDQ expert weights. Knobs can be toggled via env:
    REPLAY_FP32_REDUCE=0|1   (default: captured behavior, 1)
    REPLAY_ATOMIC_ADD=0|1    (default 0)
    REPLAY_EXPERT_SLICE=N    (use only first N experts; remap topk ids)
Run inside .venv-serve.
"""

import os
import sys

import torch

sys.path.insert(0, "/workspace/blockwise-gptq-main/opteam-blockwise-gptq")


def reconstruct_assignments(sorted_token_ids, expert_ids, block_size_m,
                            n_valid):
    """block-format schedule → list of (token, slot_index_in_topk_flat, expert)."""
    out = []
    st = sorted_token_ids.cpu()
    ei = expert_ids.cpu()
    for b in range(len(ei)):
        e = ei[b].item()
        if e < 0:
            continue
        for s in range(block_size_m):
            stid = st[b * block_size_m + s].item()
            if stid < n_valid:
                out.append((stid, e))
    return out


@torch.no_grad()
def ground_truth(blob, qdq_dir):
    """Exact MoE output from QDQ weights (fp32), matching the fused call."""
    import json
    from safetensors import safe_open

    hs = blob["hidden_states"].float().cpu()
    M, K = hs.shape
    topk = blob["num_topk"]
    n_valid = M * topk
    tw = blob["topk_weights"].float().cpu().reshape(-1)

    idx = json.load(open(f"{qdq_dir}/model.safetensors.index.json")) \
        if os.path.exists(f"{qdq_dir}/model.safetensors.index.json") else None

    def load(key):
        shard = idx["weight_map"][key] if idx else "model.safetensors"
        with safe_open(f"{qdq_dir}/{shard}", framework="pt", device="cpu") as f:
            return f.get_tensor(key).float()

    # Layer 0 of the 2-layer repro (first MoE the engine runs)
    gu = load("model.layers.0.mlp.experts.gate_up_proj")     # [E, K, 2I]
    dn = load("model.layers.0.mlp.experts.down_proj")        # [E, I, K]
    gub = load("model.layers.0.mlp.experts.gate_up_proj_bias")  # [E, 2I]
    dnb = load("model.layers.0.mlp.experts.down_proj_bias")     # [E, K]

    assigns = reconstruct_assignments(blob["sorted_token_ids"],
                                      blob["expert_ids"],
                                      blob["block_size_m"], n_valid)
    out = torch.zeros(M, K)
    for stid, e in assigns:
        t = stid // topk
        x = hs[t]
        g_u = x @ gu[e] + gub[e]
        gate, up = g_u[::2], g_u[1::2]
        gate = gate.clamp(max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        act = (up + 1) * (gate * torch.sigmoid(gate * 1.702))
        y = act @ dn[e] + dnb[e]
        out[t] += y * tw[stid]
    return out


def main():
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
    from vllm.scalar_type import ScalarType

    blob = torch.load("/workspace/results/pilot/marlin_call_capture.pt",
                      map_location="cuda", weights_only=False)
    kw = dict(blob["kw"])
    quant_type = ScalarType.from_id(blob["quant_type_id"])

    n_slice = int(os.environ.get("REPLAY_EXPERT_SLICE", "0"))
    expert_ids = blob["expert_ids"]
    w1, w2 = blob["w1"], blob["w2"]
    w1s, w2s = blob["w1_scale"], blob["w2_scale"]
    bias1, bias2 = blob["bias1"], blob["bias2"]
    for k in ("global_scale1", "global_scale2"):
        pass
    if n_slice:
        keep = expert_ids < n_slice
        # remap out-of-range experts in the schedule to expert 0 with zero
        # weight? simpler: drop blocks (mask to -1) — kernel skips them
        expert_ids = torch.where(keep, expert_ids,
                                 torch.full_like(expert_ids, -1))

    if os.environ.get("REPLAY_FP32_REDUCE") is not None:
        kw["use_fp32_reduce"] = os.environ["REPLAY_FP32_REDUCE"] == "1"
    if os.environ.get("REPLAY_ATOMIC_ADD") is not None:
        kw["use_atomic_add"] = os.environ["REPLAY_ATOMIC_ADD"] == "1"

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    act = MoEActivation(blob["activation"].split(".")[-1].lower()) \
        if "." in blob["activation"] else MoEActivation(blob["activation"])

    out = mm._fused_marlin_moe(
        hidden_states=blob["hidden_states"], w1=w1, w2=w2,
        bias1=bias1, bias2=bias2, w1_scale=w1s, w2_scale=w2s,
        topk_weights=blob["topk_weights"], num_topk=blob["num_topk"],
        quant_type=quant_type,
        apply_router_weight_on_input=blob["apply_router_weight_on_input"],
        expert_map=None, block_size_m=blob["block_size_m"],
        sorted_token_ids=blob["sorted_token_ids"], expert_ids=expert_ids,
        num_tokens_post_padded=blob["num_tokens_post_padded"],
        activation=act, **kw)

    ref = ground_truth(blob, "/workspace/models/fixture-real2l-qdq")
    o = out.float().cpu()
    # fused output is [M*topk? or M,K]: _fused_marlin_moe returns per-token
    # summed [M, K] when mul_topk_weights — handle [M*topk, K] too
    if o.shape[0] != ref.shape[0]:
        o = o.view(ref.shape[0], -1, ref.shape[1]).sum(1)
    rel = (o - ref).norm() / (ref.norm() + 1e-9)
    print(f"REPLAY out_absmax={o.abs().max():.4e} ref_absmax={ref.abs().max():.4e} "
          f"rel_err={rel:.4f} "
          f"fp32_reduce={kw.get('use_fp32_reduce')} atomic={kw.get('use_atomic_add')} "
          f"slice={n_slice or 'all'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
