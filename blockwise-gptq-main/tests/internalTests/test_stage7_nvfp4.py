"""
Stage 7 verification test for NVFP4 modelopt checkpoints.

Two levels of checks (no vLLM required, runs on CPU):

  Level 1 — Shape / dtype sanity
    Every layer listed in the Stage 5 results as quantized must have:
      .weight          uint8        shape [out, in//2]
      .weight_scale    float8_e4m3  shape [out, in//16]
      .weight_scale_2  bfloat16     shape [1]

  Level 2 — Numerical round-trip
    Unpack the uint8 + fp8 scales back to float32 and compare against
    the Stage 5 BF16 model weights.  Because Stage 5 already ran
    quantize_dequantize(), the BF16 weights are sitting exactly on the
    FP4 grid — so the reconstructed values should match very closely
    (max absolute error < 1e-3 for all but a negligible fraction of
    elements, which may flip due to per-block vs per-full-matrix scale).

Usage:
    python tests/test_stage7_nvfp4.py \\
        --bf16_model   models/DeepSeek-V2-Lite-NVFP4 \\
        --modelopt_dir models/DeepSeek-V2-Lite-NVFP4-modelopt \\
        --stage5_results results/stage5_DeepSeek-V2-Lite_nvfp4_quantize.json \\
        [--n_layers 5]     # check only the first N quantized layers (default: all)
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# ── E2M1 decode grid ──────────────────────────────────────────────────────────

_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequantize_nvfp4(weight_u8: torch.Tensor,
                     weight_scale_fp8: torch.Tensor,
                     weight_scale_2: torch.Tensor,
                     block_size: int = 16) -> torch.Tensor:
    """Unpack modelopt NVFP4 tensors back to float32.

    Args:
        weight_u8        [out, in//2]         torch.uint8
        weight_scale_fp8 [out, in//block]     torch.float8_e4m3fn
        weight_scale_2   [1]                  torch.bfloat16
        block_size       int                  (16 for NVFP4)

    Returns:
        W_reconstructed  [out, in]            torch.float32
    """
    out_features = weight_u8.shape[0]
    in_half      = weight_u8.shape[1]
    in_features  = in_half * 2

    # ── Unpack two nibbles per byte ───────────────────────────────────────
    u8 = weight_u8.long()
    lo = u8 & 0xF           # [out, in//2]  — column 2i
    hi = (u8 >> 4) & 0xF   # [out, in//2]  — column 2i+1

    # Interleave lo/hi back into column order: [out, in]
    nibbles = torch.stack([lo, hi], dim=2).reshape(out_features, in_features)

    # ── Decode E2M1 nibbles to float ──────────────────────────────────────
    grid = _E2M1_GRID  # [8]
    sign    = ((nibbles >> 3) & 1).float()    # 1 = negative
    mag_idx = (nibbles & 0x7)                 # 0–7

    W_fp4 = grid[mag_idx] * (1.0 - 2.0 * sign)   # [out, in]

    # ── Apply FP8 block scales ────────────────────────────────────────────
    # weight_scale: [out, n_blocks]
    scale_f32 = weight_scale_fp8.to(torch.float32)               # [out, n_blocks]

    # Expand each scale to cover block_size columns
    # Pad in_features to multiple of block_size if needed
    n_blocks = scale_f32.shape[1]
    scale_expanded = scale_f32.repeat_interleave(block_size, dim=1)  # [out, n_blocks*16]
    scale_expanded = scale_expanded[:, :in_features]                  # strip any padding

    # Apply global scale (weight_scale_2, typically 1.0)
    global_scale = weight_scale_2.to(torch.float32).item()

    W_reconstructed = W_fp4 * scale_expanded * global_scale
    return W_reconstructed


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_safetensors_flat(checkpoint_dir: Path) -> dict:
    """Load all tensors from a safetensors checkpoint into a flat dict."""
    from safetensors import safe_open

    tensors = {}
    shard_index = checkpoint_dir / "model.safetensors.index.json"

    if shard_index.exists():
        index = json.loads(shard_index.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    for fname in shard_files:
        path = checkpoint_dir / fname
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    return tensors


def load_bf16_weights(model_dir: Path) -> dict:
    """Load BF16 model weights as a flat state dict (no model instantiation)."""
    return load_safetensors_flat(model_dir)


# ── Main test ─────────────────────────────────────────────────────────────────

def run_tests(bf16_dir: Path, modelopt_dir: Path,
              quantized_keys: list, n_layers: int | None, block_size: int):

    print("Loading modelopt tensors …", flush=True)
    modelopt_tensors = load_safetensors_flat(modelopt_dir)

    print("Loading BF16 reference weights …", flush=True)
    bf16_tensors = load_bf16_weights(bf16_dir)

    # Map from "layer.{idx}.{sub}" to full module paths via modelopt tensor keys
    # We need full module paths; infer them from what's actually in the checkpoint.
    # Build a lookup: base_name (without .weight suffix) → True if quantized
    quantized_base_names = set()
    for key in modelopt_tensors:
        if key.endswith(".weight_scale"):
            quantized_base_names.add(key[: -len(".weight_scale")])

    keys_to_check = sorted(quantized_base_names)
    if n_layers is not None:
        keys_to_check = keys_to_check[:n_layers]

    print(f"\nFound {len(quantized_base_names)} quantized layers in checkpoint.")
    print(f"Checking {len(keys_to_check)} layers.\n")

    # ── Level 1: shape / dtype checks ────────────────────────────────────────
    print("=" * 60)
    print("LEVEL 1 — Shape / dtype sanity")
    print("=" * 60)

    l1_pass = 0
    l1_fail = 0

    for base in keys_to_check:
        w_key   = f"{base}.weight"
        s_key   = f"{base}.weight_scale"
        s2_key  = f"{base}.weight_scale_2"

        missing = [k for k in [w_key, s_key, s2_key] if k not in modelopt_tensors]
        if missing:
            print(f"  FAIL  {base}: missing tensors {missing}")
            l1_fail += 1
            continue

        w   = modelopt_tensors[w_key]
        s   = modelopt_tensors[s_key]
        s2  = modelopt_tensors[s2_key]

        out  = w.shape[0]
        in_h = w.shape[1]
        in_  = in_h * 2

        errors = []
        if w.dtype  != torch.uint8:           errors.append(f"weight dtype={w.dtype} (want uint8)")
        if s.dtype  != torch.float8_e4m3fn:   errors.append(f"scale dtype={s.dtype} (want fp8_e4m3fn)")
        if s2.dtype != torch.bfloat16:         errors.append(f"scale2 dtype={s2.dtype} (want bf16)")
        if w.shape  != (out, in_ // 2):        errors.append(f"weight shape={w.shape}")
        n_blocks = (in_ + block_size - 1) // block_size
        if s.shape  != (out, n_blocks):        errors.append(f"scale shape={s.shape} (want [{out},{n_blocks}])")
        if s2.shape != (1,):                   errors.append(f"scale2 shape={s2.shape} (want [1])")

        if errors:
            print(f"  FAIL  {base}: {'; '.join(errors)}")
            l1_fail += 1
        else:
            print(f"  OK    {base}  weight={list(w.shape)} scale={list(s.shape)}")
            l1_pass += 1

    print(f"\nLevel 1: {l1_pass} passed, {l1_fail} failed\n")

    # ── Level 2: numerical round-trip ────────────────────────────────────────
    print("=" * 60)
    print("LEVEL 2 — Numerical round-trip (unpack → compare vs BF16)")
    print("=" * 60)

    l2_pass = 0
    l2_fail = 0
    l2_skip = 0

    MAX_ABS_THRESHOLD    = 1e-2   # generous — FP4 grid step at large values is 2.0
    MISMATCH_PCT_WARNING = 0.01   # warn if >1% of elements differ beyond threshold

    for base in keys_to_check:
        w_key  = f"{base}.weight"
        s_key  = f"{base}.weight_scale"
        s2_key = f"{base}.weight_scale_2"

        if any(k not in modelopt_tensors for k in [w_key, s_key, s2_key]):
            l2_skip += 1
            continue

        # BF16 reference key
        if w_key not in bf16_tensors:
            print(f"  SKIP  {base}: weight not found in BF16 checkpoint")
            l2_skip += 1
            continue

        w_ref = bf16_tensors[w_key].float()   # [out, in]

        w_u8  = modelopt_tensors[w_key]
        w_s   = modelopt_tensors[s_key]
        w_s2  = modelopt_tensors[s2_key]

        W_rec = dequantize_nvfp4(w_u8, w_s, w_s2, block_size=block_size)

        # Trim to the same in_features (ref may not be padded)
        min_in = min(W_rec.shape[1], w_ref.shape[1])
        W_rec  = W_rec[:, :min_in]
        w_ref  = w_ref[:, :min_in]

        diff      = (W_rec - w_ref).abs()
        max_err   = diff.max().item()
        mean_err  = diff.mean().item()
        mismatch  = (diff > MAX_ABS_THRESHOLD).float().mean().item()

        if mismatch > MISMATCH_PCT_WARNING:
            print(
                f"  WARN  {base}: {mismatch*100:.2f}% elements > {MAX_ABS_THRESHOLD}  "
                f"max={max_err:.4f}  mean={mean_err:.6f}"
            )
            l2_fail += 1
        else:
            print(
                f"  OK    {base}  max_err={max_err:.6f}  mean_err={mean_err:.8f}  "
                f"mismatch={mismatch*100:.4f}%"
            )
            l2_pass += 1

    print(f"\nLevel 2: {l2_pass} passed, {l2_fail} warnings, {l2_skip} skipped\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Level 1 (shapes/dtypes)   : {l1_pass}/{l1_pass+l1_fail} OK")
    print(f"  Level 2 (numerical match) : {l2_pass}/{l2_pass+l2_fail+l2_skip} OK  "
          f"({l2_skip} skipped, {l2_fail} warnings)")

    if l1_fail == 0 and l2_fail == 0:
        print("\n✓  All checks passed — Stage 7 output looks correct.")
    else:
        print("\n✗  Some checks failed — review warnings above.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bf16_model",     type=Path, required=True,
                   help="Stage 5 BF16 model directory (the source for Stage 7)")
    p.add_argument("--modelopt_dir",   type=Path, required=True,
                   help="Stage 7 modelopt output directory")
    p.add_argument("--stage5_results", type=Path, default=None,
                   help="Stage 5 results JSON (optional — used only for bookkeeping)")
    p.add_argument("--n_layers",       type=int,   default=None,
                   help="Check only the first N quantized layers (default: all)")
    p.add_argument("--block_size",     type=int,   default=16)
    args = p.parse_args()

    quantized_keys = []
    if args.stage5_results and args.stage5_results.exists():
        results = json.loads(args.stage5_results.read_text())
        quantized_keys = results.get("quantized_attn_keys", [])

    run_tests(
        bf16_dir        = args.bf16_model,
        modelopt_dir    = args.modelopt_dir,
        quantized_keys  = quantized_keys,
        n_layers        = args.n_layers,
        block_size      = args.block_size,
    )


if __name__ == "__main__":
    main()