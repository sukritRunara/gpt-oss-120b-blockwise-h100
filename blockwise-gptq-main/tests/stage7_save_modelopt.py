"""
Stage 7 — Save GPTQ-quantized model in a packed quantized checkpoint.

Reads the BF16-dequantized model written by Stage 5 and repacks the
quantized layers into a format readable by vLLM, SGLang, and TensorRT-LLM.

NOTE on serving-engine flags
─────────────────────────────
There are NO --vllm / --sglang / --trt flags.  For all formats currently
supported (NVFP4, FP8, INT8) the three engines read the exact same
HuggingFace-compatible modelopt safetensors layout, so a single output
directory works for all of them.  Engine-specific packing only becomes
relevant for INT4/GPTQ (deferred — see TODO below).

Supported formats (auto-detected from Stage 5 results JSON, or set
explicitly with --quant_format):

  nvfp4   → uint8 packed FP4-E2M1 + fp8_e4m3 block scales  (Blackwell, W4A16)
  fp8     → float8_e4m3fn weights + float32 per-tensor scale
  int8    → int8 weights + float32 per-channel scale
  int4 / int4_perchannel / mxint4  → NOT YET IMPLEMENTED (kept BF16 + warning)

Per quantized linear layer the output contains:

  NVFP4 (W4A16_NVFP4):
    {layer}.weight          uint8        [out, in//2]     packed FP4-E2M1
    {layer}.weight_scale    float8_e4m3  [out, in//16]    per-block FP8 scale
                                                           (normalised by weight_scale_2)
    {layer}.weight_scale_2  bfloat16     [1]              shared global scale for
                                                           parallel-layer groups;
                                                           1.0 for singleton layers

  FP8:
    {layer}.weight          float8_e4m3  [out, in]        quantized weights
    {layer}.weight_scale    float32      [1]              per-tensor scale

  INT8:
    {layer}.weight          int8         [out, in]        quantized weights
    {layer}.weight_scale    float32      [out]            per-channel scale

Parallel-scale sharing (NVFP4)
────────────────────────────────
Serving engines (vLLM, TRT-LLM, SGLang) fuse parallel projection layers
(e.g. q_proj + k_proj + v_proj, gate_proj + up_proj) into a single GEMM.
The fused kernel expects all layers in the group to share the same global
scale (weight_scale_2).

Stage 7 automatically detects parallel groups: any set of quantized Linear
layers that share the same parent module AND the same input dimension
(in_features) are treated as parallel.  Their weight_scale_fp8 tensors are
normalised by a shared weight_scale_2 = max raw scale across the group,
so the global scale is identical for all members.

Usage:
    # Auto-detect format from Stage 5 results (recommended)
    python stage7_save_modelopt.py \\
        --model_path  models/DeepSeek-V2-Lite-NVFP4 \\
        --output_dir  models/DeepSeek-V2-Lite-NVFP4-modelopt \\
        --stage5_results results/stage5_DeepSeek-V2-Lite_nvfp4_quantize.json

    # Override format explicitly
    python stage7_save_modelopt.py \\
        --model_path  models/DeepSeek-V2-Lite-FP8 \\
        --output_dir  models/DeepSeek-V2-Lite-FP8-modelopt \\
        --quant_format fp8

    # Fallback: no results JSON (packs all nn.Linear except lm_head)
    python stage7_save_modelopt.py \\
        --model_path  models/DeepSeek-V2-Lite-NVFP4 \\
        --output_dir  models/DeepSeek-V2-Lite-NVFP4-modelopt \\
        --quant_format nvfp4
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Repo-relative code root (P0.1 fix): the library lives at
# <repo>/opteam-blockwise-gptq regardless of where the repo is checked out.
_CODE_ROOT = Path(__file__).resolve().parents[1] / "opteam-blockwise-gptq"
if not _CODE_ROOT.exists():
    raise RuntimeError(f"Code root not found: {_CODE_ROOT}")
sys.path.insert(0, str(_CODE_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Per-format packing functions
# Each takes a float32 weight tensor and returns (main_weight, *scales).
# ══════════════════════════════════════════════════════════════════════════════

# ── NVFP4 ────────────────────────────────────────────────────────────────────

_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MAX  = 6.0


def _to_e2m1_nibbles(x_scaled: torch.Tensor) -> torch.Tensor:
    """Snap scale-normalised values to the nearest E2M1 grid point (0–15)."""
    grid  = _E2M1_GRID.to(x_scaled.device)
    sign  = (x_scaled < 0).long()
    dists = (x_scaled.abs().unsqueeze(-1) - grid).abs()
    idx   = dists.argmin(dim=-1).long()
    return idx | (sign << 3)


def pack_nvfp4(
    W: torch.Tensor,
    block_size: int = 16,
    global_scale: float | None = None,
):
    """NVFP4: FP4-E2M1 packed as uint8 + FP8 per-block scales.

    Args:
        W             float tensor [out, in]
        block_size    microscaling block size (default 16)
        global_scale  shared global scale for parallel-layer groups.
                      When provided:
                        weight_scale_2   = global_scale  (bfloat16)
                        weight_scale_fp8 = raw_per_block / global_scale
                                           (normalised to ≤ 1.0, stored as fp8)
                      When None (singleton layer):
                        weight_scale_2   = 1.0
                        weight_scale_fp8 = raw_per_block  (stored as fp8)
                      Dequantisation is always:
                        W_float = E2M1_value * weight_scale_fp8 * weight_scale_2

    Returns:
        weight_u8        [out, in//2]          torch.uint8
        weight_scale_fp8 [out, in//block_size] torch.float8_e4m3fn
        weight_scale_2   [1]                   torch.bfloat16
    """
    W = W.float()
    out_features, in_features = W.shape
    bs = block_size

    pad = (bs - in_features % bs) % bs
    W_p = torch.nn.functional.pad(W, (0, pad)) if pad else W
    n_blocks = W_p.shape[1] // bs

    W_blocked = W_p.reshape(out_features, n_blocks, bs)

    amax      = W_blocked.abs().amax(dim=2).clamp(min=1e-12)  # [out, n_blocks]
    raw_scale = amax / _E2M1_MAX                               # [out, n_blocks]

    # NOTE: weight_scale_2 is always 1.0 for W4A16_NVFP4 (Marlin kernel).
    # Marlin reads weight_scale_fp8 directly as the full per-block scale and
    # does not multiply by weight_scale_2 in its computation.  Normalising
    # the FP8 scales by a global_scale and storing that in weight_scale_2
    # would be correct for W4A4 (CUTLASS kernel), but breaks W4A16 (Marlin).
    # The global_scale parameter is accepted for API compatibility and reserved
    # for future W4A4 support, but is unused here.
    _ = global_scale  # reserved — see note above
    weight_scale_2 = torch.ones(1, dtype=torch.bfloat16)
    scale_fp8      = raw_scale.to(torch.float8_e4m3fn)
    scale_f32      = scale_fp8.to(torch.float32)

    nibbles  = _to_e2m1_nibbles(W_blocked / scale_f32.unsqueeze(2))
    fp4_flat = nibbles.reshape(out_features, n_blocks * bs)[:, :in_features]

    if in_features % 2 != 0:
        raise ValueError(f"in_features={in_features} must be even for FP4 packing.")
    fp4_pairs = fp4_flat.reshape(out_features, in_features // 2, 2)
    weight_u8 = (fp4_pairs[..., 0] | (fp4_pairs[..., 1] << 4)).to(torch.uint8)

    n_scale_blocks   = (in_features + bs - 1) // bs
    weight_scale_fp8 = scale_fp8[:, :n_scale_blocks]

    return weight_u8, weight_scale_fp8, weight_scale_2


def _compute_shared_scales(
    model: nn.Module,
    quantized_fullnames: set,
    block_size: int = 16,
    raw_weights: dict | None = None,
) -> dict:
    """Pre-compute shared global scales for groups of parallel linear layers.

    Parallel layers are detected by grouping quantized Linear modules that share
    the same parent module path AND the same in_features dimension.  Typical
    groups: (q_proj, k_proj, v_proj), (gate_proj, up_proj).

    raw_weights: optional {mod_name: tensor} from _build_raw_weight_map.
    When provided, scale computation uses the raw on-disk weights rather than
    the potentially-transformed model object weights.

    Returns a dict {layer_name: shared_global_scale} covering only layers that
    belong to a group of 2+.  Singleton layers are NOT included (they get
    global_scale=None → weight_scale_2=1.0 in pack_nvfp4).
    """
    if raw_weights is None:
        raw_weights = {}

    # Group by (parent_module_path, in_features)
    groups: dict = defaultdict(list)
    for mod_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if mod_name not in quantized_fullnames:
            continue
        parent = mod_name.rsplit(".", 1)[0] if "." in mod_name else ""
        groups[(parent, module.in_features)].append((mod_name, module))

    shared_scales: dict = {}

    for (parent, in_feats), members in groups.items():
        if len(members) < 2:
            continue  # singleton — independent scale, no sharing needed

        # Max raw per-block scale across all members in this parallel group
        group_max = 0.0
        for _name, module in members:
            # Prefer raw on-disk weight over model-object weight
            W_raw = raw_weights.get(_name)
            W    = W_raw.float() if W_raw is not None else module.weight.data.float()
            out_f, in_f = W.shape
            pad  = (block_size - in_f % block_size) % block_size
            W_p  = torch.nn.functional.pad(W, (0, pad)) if pad else W
            n_bl = W_p.shape[1] // block_size
            amax = W_p.reshape(out_f, n_bl, block_size).abs().amax(dim=2).clamp(min=1e-12)
            raw_max = (amax / _E2M1_MAX).max().item()
            if raw_max > group_max:
                group_max = raw_max

        for name, _ in members:
            shared_scales[name] = group_max

    # Summary
    if shared_scales:
        # Count distinct groups (unique scale values are approximate; just count
        # distinct (parent, in_feats) keys that contributed 2+ members)
        n_groups  = sum(1 for members in groups.values() if len(members) >= 2)
        n_covered = len(shared_scales)
        print(
            f"[Stage 7] Parallel scale sharing: "
            f"{n_covered} layers across {n_groups} groups "
            f"(shared weight_scale_2 per group)"
        )
    else:
        print("[Stage 7] No parallel layer groups found — all layers use independent scales.")

    return shared_scales


# ── FP8 ──────────────────────────────────────────────────────────────────────

_FP8_E4M3_MAX = 448.0   # torch.finfo(torch.float8_e4m3fn).max


def pack_fp8(W: torch.Tensor):
    """FP8 E4M3: per-tensor symmetric quantisation.

    Returns:
        weight_fp8   [out, in]  torch.float8_e4m3fn
        weight_scale [1]        torch.float32
    """
    W = W.float()
    amax        = W.abs().max().clamp(min=1e-12)
    scale       = (amax / _FP8_E4M3_MAX).to(torch.float32)
    weight_fp8  = (W / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return weight_fp8, scale.reshape(1)


# ── INT8 ─────────────────────────────────────────────────────────────────────

_INT8_MAX = 127.0


def pack_int8(W: torch.Tensor):
    """INT8 symmetric: per-channel quantisation (one scale per output row).

    Returns:
        weight_int8  [out, in]  torch.int8
        weight_scale [out]      torch.float32
    """
    W = W.float()
    amax        = W.abs().amax(dim=1).clamp(min=1e-12)   # [out]
    scale       = (amax / _INT8_MAX).to(torch.float32)
    weight_int8 = (W / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return weight_int8, scale


# ── Dispatch ─────────────────────────────────────────────────────────────────

# Maps quant_format → (packer, tensor_names, quant_algo_for_config)
#   packer(*) returns a tuple of tensors; tensor_names lists their suffixes
_FORMAT_REGISTRY = {
    "nvfp4": {
        "packer":      pack_nvfp4,
        "suffixes":    ["weight", "weight_scale", "weight_scale_2"],
        "quant_algo":  "W4A16_NVFP4",   # W4A16: FP4 weights + BF16 activations
        "num_bits":    4,
        "weight_type": "float",
        "group_size":  16,
    },
    "fp8": {
        "packer":      pack_fp8,
        "suffixes":    ["weight", "weight_scale"],
        "quant_algo":  "FP8",
        "num_bits":    8,
        "weight_type": "float",
        "group_size":  None,
    },
    "int8": {
        "packer":      pack_int8,
        "suffixes":    ["weight", "weight_scale"],
        "quant_algo":  "W8A8_SQ",
        "num_bits":    8,
        "weight_type": "int",
        "group_size":  None,
    },
}

# Formats not yet implemented — saved as BF16 with a warning
_UNIMPLEMENTED_FORMATS = {"int4", "int4_perchannel", "mxint4"}


# ══════════════════════════════════════════════════════════════════════════════
# Layer-name resolution (maps apply.py keys → full model paths)
# ══════════════════════════════════════════════════════════════════════════════

def resolve_quantized_fullnames(model, quantized_attn_keys, layer_losses, handler):
    """Map 'layer.{idx}.{sublayer}' → full module paths in the model.

    Returns:
        quantized_fullnames : set of full module paths to pack
        bf16_fullnames      : list of module paths kept BF16 (for config ignore list)
    """
    from model_utils import get_model_layers, find_layers

    layers, _          = get_model_layers(model)
    id_to_fullname     = {id(m): n for n, m in model.named_modules()}
    quantized_fullnames = set()
    bf16_fullnames      = []

    for layer_idx, layer in enumerate(layers):
        raw_subset  = find_layers(layer)
        attn_subset = (
            handler.filter_standard_layers(layer, raw_subset)
            if handler is not None else raw_subset
        )

        for sub_name, sub_module in attn_subset.items():
            key       = f"layer.{layer_idx}.{sub_name}"
            full_name = id_to_fullname.get(id(sub_module))
            if full_name is None:
                continue
            if key in quantized_attn_keys:
                quantized_fullnames.add(full_name)
            else:
                bf16_fullnames.append(full_name)

        if handler is not None and handler.has_moe(layer):
            expert_subset    = {k: v for k, v in raw_subset.items() if k not in attn_subset}
            layer_loss       = layer_losses.get(str(layer_idx), {})
            expert_quantized = (layer_loss.get("experts.gate_up") != "BF16")

            for sub_name, sub_module in expert_subset.items():
                full_name = id_to_fullname.get(id(sub_module))
                if full_name is None:
                    continue
                if expert_quantized:
                    quantized_fullnames.add(full_name)
                else:
                    bf16_fullnames.append(full_name)

    return quantized_fullnames, bf16_fullnames


# ══════════════════════════════════════════════════════════════════════════════
# Raw-weight loader (bypasses model-init transformations)
# ══════════════════════════════════════════════════════════════════════════════

def _load_all_safetensors(model_path: Path) -> dict[str, torch.Tensor]:
    """Load ALL tensors directly from the on-disk safetensors shards.

    AutoModelForCausalLM.from_pretrained with trust_remote_code=True can apply
    model-specific weight transformations (e.g. absorbing RoPE scale factors into
    projection weights).  vLLM re-applies these same transformations when it loads
    a quantized checkpoint, so both the BF16 and the packed-FP4 tensors stored in
    the checkpoint must be the RAW on-disk values — not the post-transform ones.

    Using this as the base state-dict ensures full consistency, matching the
    behaviour of loading without trust_remote_code (which is what happened when
    v1 was generated before the modeling_deepseek.py patch was applied).
    """
    from safetensors import safe_open

    idx_file = model_path / "model.safetensors.index.json"
    if idx_file.exists():
        shard_files = sorted(set(json.loads(idx_file.read_text())["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    tensors: dict[str, torch.Tensor] = {}
    for shard_fname in shard_files:
        with safe_open(str(model_path / shard_fname), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    print(f"[Stage 7] Loaded {len(tensors)} raw tensors from safetensors "
          f"({len(shard_files)} shard(s))")
    return tensors


# ══════════════════════════════════════════════════════════════════════════════
# State-dict builder
# ══════════════════════════════════════════════════════════════════════════════

def build_quantized_state_dict(
    model: nn.Module,
    quantized_fullnames: set,
    quant_format: str,
    model_path: Path,
    block_size: int = 16,
) -> dict:
    """Pack quantised layers and return a flat state dict for safetensors."""

    if quant_format in _UNIMPLEMENTED_FORMATS:
        print(
            f"[Stage 7] WARNING: {quant_format} packing is not yet implemented.\n"
            f"          Layers will be saved as BF16.  The resulting checkpoint\n"
            f"          will NOT be directly loadable by vLLM/SGLang/TRT-LLM as\n"
            f"          a quantized model."
        )
        return {name: param.data.cpu() for name, param in model.named_parameters()}

    fmt      = _FORMAT_REGISTRY[quant_format]
    packer   = fmt["packer"]
    suffixes = fmt["suffixes"]

    # ── Use raw on-disk tensors as the base state dict ───────────────────────
    # This bypasses ALL model-init transformations for ALL parameters (BF16
    # non-quantized layers AND the weights we'll pack as FP4/FP8/INT8).
    # vLLM re-applies the same trust_remote_code transformations when it loads
    # the checkpoint, so the checkpoint must contain the pre-transform values.
    state_dict = _load_all_safetensors(model_path)

    # Fall back to model object for any tensor missing from safetensors
    for name, param in model.named_parameters():
        if name not in state_dict:
            state_dict[name] = param.data.cpu()
            print(f"[Stage 7] Fallback to model object for: {name}")

    # Extract raw weights for quantized layers (used for packing and shared scales)
    raw_weights = {
        mod_name: state_dict[f"{mod_name}.weight"]
        for mod_name in quantized_fullnames
        if f"{mod_name}.weight" in state_dict
    }
    n_miss = len(quantized_fullnames) - len(raw_weights)
    if n_miss:
        print(f"[Stage 7] WARNING: {n_miss} quantized layers missing from safetensors "
              f"— will fall back to (possibly transformed) model object weights")

    # ── NVFP4: pre-compute shared global scales for parallel layer groups ──────
    # Use raw weights so scales reflect the true on-disk values
    if quant_format == "nvfp4":
        shared_scales = _compute_shared_scales(model, quantized_fullnames, block_size,
                                               raw_weights=raw_weights)
    else:
        shared_scales = {}

    n_packed = 0
    n_total  = len(quantized_fullnames)

    for mod_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if mod_name not in quantized_fullnames:
            continue

        # Use raw safetensors weight; fall back to model object
        W = raw_weights.get(mod_name)
        if W is None:
            W = module.weight.data.cpu()

        if quant_format == "nvfp4":
            # Pass the shared global scale (None for singleton layers)
            g_scale = shared_scales.get(mod_name)
            tensors = packer(W, block_size=block_size, global_scale=g_scale)
        else:
            tensors = packer(W)

        for suffix, tensor in zip(suffixes, tensors):
            state_dict[f"{mod_name}.{suffix}"] = tensor

        n_packed += 1
        if n_packed % 100 == 0 or n_packed == n_total:
            print(f"  [{n_packed}/{n_total}] layers packed …", flush=True)

    n_linear = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    print(
        f"[Stage 7] {n_packed} layers packed ({quant_format.upper()}), "
        f"{n_linear - n_packed} layers kept BF16"
    )
    return state_dict


# ══════════════════════════════════════════════════════════════════════════════
# Shard + save
# ══════════════════════════════════════════════════════════════════════════════

def save_sharded(state_dict: dict, output_dir: Path, max_shard_gb: float = 4.0):
    from safetensors.torch import save_file

    max_bytes = int(max_shard_gb * 1024 ** 3)
    shards: list[dict] = []
    cur: dict = {}
    cur_bytes = 0

    for key, tensor in state_dict.items():
        sz = tensor.numel() * tensor.element_size()
        if cur_bytes + sz > max_bytes and cur:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[key] = tensor
        cur_bytes += sz
    if cur:
        shards.append(cur)

    if len(shards) == 1:
        path = output_dir / "model.safetensors"
        save_file(shards[0], str(path))
        print(f"[Stage 7] Saved {path.name}  ({path.stat().st_size / 1024**3:.2f} GB)")
    else:
        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        for i, shard in enumerate(shards):
            fname = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            path  = output_dir / fname
            save_file(shard, str(path))
            sz = path.stat().st_size
            index["metadata"]["total_size"] += sz
            for k in shard:
                index["weight_map"][k] = fname
            print(f"[Stage 7] Saved {fname}  ({sz / 1024**3:.2f} GB)")
        idx_path = output_dir / "model.safetensors.index.json"
        idx_path.write_text(json.dumps(index, indent=2))
        print(f"[Stage 7] Shard index → {idx_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# config.json
# ══════════════════════════════════════════════════════════════════════════════

def write_quantization_config(
    model_path: Path,
    output_dir: Path,
    quant_format: str,
    ignored_names: list,
    block_size: int,
):
    config = json.loads((model_path / "config.json").read_text())

    ignore_list = ["lm_head"] + [n for n in ignored_names if n != "lm_head"]
    seen, deduped = set(), []
    for n in ignore_list:
        if n not in seen:
            deduped.append(n)
            seen.add(n)

    if quant_format in _UNIMPLEMENTED_FORMATS:
        # Don't add a quantization_config for unimplemented formats — the
        # checkpoint is still BF16 and engines should treat it as such.
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
        return

    fmt = _FORMAT_REGISTRY[quant_format]
    weights_cfg: dict = {"num_bits": fmt["num_bits"], "type": fmt["weight_type"]}
    if fmt["group_size"] is not None:
        weights_cfg["group_size"] = block_size if quant_format == "nvfp4" else fmt["group_size"]

    config["quantization_config"] = {
        "quant_method":  "modelopt",
        "quant_algo":    fmt["quant_algo"],
        "config_groups": {
            "group_0": {
                "weights": weights_cfg,
                "targets": ["Linear"],
            }
        },
        "ignore": deduped,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[Stage 7] config.json written  (quant_algo={fmt['quant_algo']})")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

_ALL_FORMATS = list(_FORMAT_REGISTRY.keys()) + list(_UNIMPLEMENTED_FORMATS)


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_path",     type=Path, required=True,
                   help="Stage 5 BF16 output directory (dequantised model)")
    p.add_argument("--output_dir",     type=Path, required=True,
                   help="Output directory for the packed checkpoint")
    p.add_argument("--stage5_results", type=Path, default=None,
                   help="Stage 5 results JSON.  Provides the quant_format and "
                        "the exact list of NVFP4 vs BF16 layers.  If omitted, "
                        "--quant_format is required and all nn.Linear are packed.")
    p.add_argument("--quant_format",   default=None, choices=_ALL_FORMATS,
                   help="Quantization format.  Auto-detected from stage5_results "
                        "if not set.  Required when --stage5_results is omitted.")
    p.add_argument("--block_size",     type=int, default=16,
                   help="Microscaling block size for NVFP4 (default: 16)")
    return p.parse_args()


def main():
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = _parse_args()

    # ── Resolve quant_format ──────────────────────────────────────────────────
    quant_format = args.quant_format

    if args.stage5_results is not None:
        if not args.stage5_results.exists():
            print(f"ERROR: stage5_results not found: {args.stage5_results}")
            sys.exit(1)
        results = json.loads(args.stage5_results.read_text())
        if quant_format is None:
            quant_format = results.get("quant_format")
            if quant_format is None:
                print("ERROR: 'quant_format' not in results JSON and --quant_format not set.")
                sys.exit(1)
    else:
        results = None
        if quant_format is None:
            print("ERROR: --quant_format is required when --stage5_results is omitted.")
            sys.exit(1)

    print("=" * 68)
    print(f"Stage 7 — Save {quant_format.upper()} checkpoint")
    print("=" * 68)
    print(f"  Model path    : {args.model_path}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Quant format  : {quant_format}")
    print(f"  Results JSON  : {args.stage5_results or '(none)'}")
    if quant_format == "nvfp4":
        print(f"  Block size    : {args.block_size}")
        print(f"  Scale sharing : enabled (parallel groups get shared weight_scale_2)")

    if not args.model_path.exists():
        print(f"ERROR: model not found: {args.model_path}")
        sys.exit(1)

    # ── Load model on CPU ─────────────────────────────────────────────────────
    print("\nLoading model on CPU …", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    print(f"  Loaded in {time.perf_counter() - t0:.1f} s  "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f} B params)")

    # ── Resolve which layers to pack ──────────────────────────────────────────
    if results is not None and "quantized_attn_keys" in results:
        from model_utils import get_model_layers
        from expert_dispatch import get_handler

        quantized_attn_keys = set(results["quantized_attn_keys"])
        layer_losses        = results.get("layer_losses", {})
        _, arch_type        = get_model_layers(model)
        handler             = get_handler(arch_type)

        quantized_fullnames, bf16_fullnames = resolve_quantized_fullnames(
            model, quantized_attn_keys, layer_losses, handler
        )
        print(f"\n  {len(quantized_fullnames)} {quant_format.upper()} layers  "
              f"({len(bf16_fullnames)} BF16/fallback layers from results JSON)")
    else:
        # Fallback: pack all nn.Linear except lm_head
        quantized_fullnames = {
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name != "lm_head"
        }
        bf16_fullnames = ["lm_head"]
        print(f"\n  Packing all {len(quantized_fullnames)} nn.Linear (excl. lm_head)")

    # ── Pack ──────────────────────────────────────────────────────────────────
    print(f"\nPacking weights to {quant_format.upper()} …", flush=True)
    t_pack = time.perf_counter()
    state_dict = build_quantized_state_dict(
        model, quantized_fullnames, quant_format,
        model_path=args.model_path, block_size=args.block_size
    )
    print(f"  Packing done in {time.perf_counter() - t_pack:.1f} s")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\nSaving safetensors shards …", flush=True)
    t_save = time.perf_counter()
    save_sharded(state_dict, args.output_dir)
    print(f"  Saved in {time.perf_counter() - t_save:.1f} s")

    print("Saving tokenizer and support files …")
    tokenizer.save_pretrained(str(args.output_dir))
    for f in args.model_path.iterdir():
        if not f.is_file():
            continue
        if f.suffix in {".safetensors", ".bin"}:
            continue
        if f.name in {"config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.json", "merges.txt"}:
            continue
        dst = args.output_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    write_quantization_config(
        args.model_path, args.output_dir, quant_format, bf16_fullnames, args.block_size
    )

    print(f"\n{'='*68}")
    print(f"Stage 7 complete — {quant_format.upper()} checkpoint at {args.output_dir}")
    print(f"{'='*68}")
    shard = args.output_dir / "model.safetensors"
    if shard.exists():
        print(f"\nSanity check:")
        print(f"  python3 -c \"from safetensors import safe_open; "
              f"f=safe_open('{shard}',framework='pt',device='cpu'); "
              f"print([k for k in f.keys() if 'weight' in k][:8])\"")


if __name__ == "__main__":
    main()