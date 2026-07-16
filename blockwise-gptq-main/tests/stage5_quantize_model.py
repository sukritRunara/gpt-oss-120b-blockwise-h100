"""
Stage 5 — Quantize a model to a target precision

Thin orchestration wrapper around gptq_quantize_model() from quantize_model.py.
That function handles calibration data loading, layer-by-layer GPTQ, and
Hessian accumulation. Stage 5 adds:
  - C4 calibration by default (more diverse than WikiText-2)
  - Optional pre-flight blocksize sweep on two representative layers before
    committing to a full-model run (~60-90 min)
  - JSON results file consumed by Stage 6

Output filenames are derived from the model folder name and quant format so
multiple model × format combinations can coexist without overwriting each other.

Supported formats (--quant_format):
  nvfp4            NVIDIA FP4, block_size=16  (default)
  mxint4           MX INT4,    block_size=16
  int4             INT4 symmetric, group quantization (groupsize=128)
  int4_perchannel  INT4 symmetric, per-channel
  int8             INT8 symmetric
  fp8              FP8 E4M3

Calibration:  C4 validation, 512 samples × 2048 tokens  (default)
GPTQ params:  blocksize=128, percdamp=0.01, mode=blockwise

Cascade fix:  --parallel_hessian (default ON) collects all Hessians from the
              original BF16 model before any quantization, eliminating the
              layer-by-layer error cascade that caused NaN losses.

Usage:
    # Default (NVFP4):
    python stage5_quantize_model.py --model_path models/<model-name>

    # Different quantization format:
    python stage5_quantize_model.py --model_path models/<model-name> --quant_format int8

    # Sequential mode (cascade-prone, for debugging only):
    python stage5_quantize_model.py --model_path models/<model-name> --no_parallel_hessian

    # Fast blocksize sweep on two representative layers, then full quantization:
    python stage5_quantize_model.py --model_path models/<model-name> --blocksize_search

    # WikiText-2 calibration (compare vs C4):
    python stage5_quantize_model.py --model_path models/<model-name> --dataset wikitext2

Exit:  0 = success, 1 = failure
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT  = Path(__file__).resolve().parents[1]
_CODE_ROOT  = Path(
    "/home/runara_dgx_spark_1/Itamar/projects"
    "/Block-wise-GPTQ-GPT-OSS-20B-NVFP4"
    "/opteam-blockwise-gptq"
)

if not _CODE_ROOT.exists():
    raise RuntimeError(f"Code root not found: {_CODE_ROOT}")
sys.path.insert(0, str(_CODE_ROOT))

# ── Quantizer registry ────────────────────────────────────────────────────────
# Maps CLI format names to (class_name, constructor_kwargs).
# The actual classes are imported lazily inside _make_quantizer() so this
# module can be imported without torch being present.

QUANTIZER_REGISTRY = {
    "nvfp4":           ("NVFP4Quantizer",          {"block_size": 16}),
    "mxint4":          ("MXInt4Quantizer",          {"block_size": 16}),
    "int4":            ("Int4SymGroupQuantizer",     {"groupsize": 128}),
    "int4_perchannel": ("Int4SymQuantizer",          {}),
    "int8":            ("Int8SymQuantizer",          {}),
    "fp8":             ("FP8E4M3Quantizer",          {}),
}

_QUANT_FORMAT_CHOICES = list(QUANTIZER_REGISTRY.keys())


def _make_quantizer(quant_format: str, device=None):
    """Instantiate the quantizer for *quant_format*.

    NVFP4 and MXInt4 require a device argument; others do not.
    """
    from quantizer import (
        NVFP4Quantizer, MXInt4Quantizer,
        Int4SymGroupQuantizer, Int4SymQuantizer,
        Int8SymQuantizer, FP8E4M3Quantizer,
    )
    _cls_map = {
        "NVFP4Quantizer":         NVFP4Quantizer,
        "MXInt4Quantizer":        MXInt4Quantizer,
        "Int4SymGroupQuantizer":  Int4SymGroupQuantizer,
        "Int4SymQuantizer":       Int4SymQuantizer,
        "Int8SymQuantizer":       Int8SymQuantizer,
        "FP8E4M3Quantizer":       FP8E4M3Quantizer,
    }
    cls_name, kwargs = QUANTIZER_REGISTRY[quant_format]
    cls = _cls_map[cls_name]
    kw = dict(kwargs)
    if quant_format in ("nvfp4", "mxint4") and device is not None:
        kw["device"] = device
    return cls(**kw)


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=Path,
                   default=_REPO_ROOT / "models" / "gpt-oss-20b-BF16",
                   help="Path to the BF16 model weights")
    p.add_argument("--quant_format", default="nvfp4",
                   choices=_QUANT_FORMAT_CHOICES,
                   help="Quantization format (default: nvfp4). "
                        f"Choices: {', '.join(_QUANT_FORMAT_CHOICES)}")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to save the quantized model "
                        "(default: models/<model_stem>-<FORMAT>)")
    p.add_argument("--dataset",   default="c4",
                   choices=["c4", "wikitext2"],
                   help="Calibration dataset (default: c4)")
    p.add_argument("--n_calib",   type=int,   default=512,
                   help="Calibration samples (default: 512)")
    p.add_argument("--seq_len",   type=int,   default=2048,
                   help="Sequence length (default: 2048)")
    p.add_argument("--blocksize", type=int,   default=128,
                   help="GPTQ block width; must be multiple of 16 (default: 128)")
    p.add_argument("--percdamp",  type=float, default=0.01,
                   help="Hessian damping factor (default: 0.01)")
    p.add_argument("--blocksize_search", action="store_true",
                   help="Run B ∈ {64,128,256} sweep on two representative layers "
                        "before full quantization and pick the best B automatically.")
    p.add_argument("--no_parallel_hessian", action="store_true",
                   help="Disable parallel Hessian collection (sequential mode, "
                        "cascade-prone). Default: parallel mode ON.")
    p.add_argument("--mixed_precision_threshold", type=float, default=100.0,
                   help="Sublayers with GPTQ loss above this threshold are kept in "
                        "BF16 instead of being quantized. Set to 0 to quantize "
                        "everything. (default: 100.0)")
    p.add_argument("--results", type=Path, default=None,
                   help="Path for the JSON results file "
                        "(default: results/stage5_<model_stem>_quantize.json)")
    args = p.parse_args()

    # Derive model_stem-based defaults
    model_stem   = args.model_path.name
    fmt_tag      = args.quant_format.upper()   # e.g. "NVFP4", "INT8"
    if args.output_dir is None:
        args.output_dir = _REPO_ROOT / "models" / f"{model_stem}-{fmt_tag}"
    if args.results is None:
        args.results = _REPO_ROOT / "results" / f"stage5_{model_stem}_{args.quant_format}_quantize.json"

    return args


# ── Blocksize sweep on representative layers ──────────────────────────────────

def blocksize_sweep(model, model_name, device, percdamp, quant_format="nvfp4"):
    """Sweep B ∈ {64, 128, 256} on two representative attention projections.

    Uses a forward pre-hook on layer 0 to capture real hidden-state inputs
    (architecture-agnostic — no hardcoded layer paths or positional embedding
    kwargs). Probes prefer q_proj + o_proj; falls back to the first two linear
    layers found by find_layers() if those names aren't present.

    For each B reports GPTQ loss and Hessian condition numbers.
    Returns the B with the lowest combined loss.
    """
    import torch
    from model_utils import get_model_layers, find_layers
    from gptq import GPTQ

    N_PROBE      = 16
    SWEEP_SEQLEN = 512

    layers, arch_type = get_model_layers(model)
    layer0 = layers[0]

    # ── Capture real layer-0 inputs via forward pre-hook ─────────────────────
    all_hidden   = []   # list of [1, SWEEP_SEQLEN, hidden] tensors
    first_kwargs = {}   # position embeddings etc., architecture-specific

    def _pre_hook(module, args, kwargs):
        if len(all_hidden) < N_PROBE:
            all_hidden.append(args[0].detach())
            if not first_kwargs:
                for k, v in kwargs.items():
                    if isinstance(v, (torch.Tensor, tuple)):
                        first_kwargs[k] = v

    handle = layer0.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(N_PROBE):
            ids = torch.randint(0, model.config.vocab_size,
                                (1, SWEEP_SEQLEN), device=device)
            model(ids)
    handle.remove()

    if not all_hidden:
        print("  WARNING: could not capture layer-0 inputs; skipping sweep.")
        return 128, {}

    # ── Select two probe projections from layer 0 ─────────────────────────────
    subset = find_layers(layer0)
    preferred = ["self_attn.q_proj", "self_attn.o_proj"]
    probe_names = [n for n in preferred if n in subset]
    if len(probe_names) < 2:
        probe_names = list(subset.keys())[:2]

    print("\n── Blocksize sweep (representative layers) ──────────────────────────")
    print(f"  Projections: {', '.join(probe_names)}")
    print(f"  {'B':>5s}  {'proj':<24s}  {'loss':>12s}  "
          f"{'mean_cond':>12s}  {'max_cond':>12s}")

    sweep_results = {}   # B → total_loss

    for B in [64, 128, 256]:
        b_total_loss = 0.0
        for proj_name in probe_names:
            linear = subset[proj_name]
            g = GPTQ(linear)
            g.quantizer = _make_quantizer(quant_format, device)

            handles = []
            def _hook(m, inp, out, _g=g):
                _g.add_batch(inp[0], out)
            handles.append(linear.register_forward_hook(_hook))

            with torch.no_grad():
                for h in all_hidden:
                    layer0(h, **first_kwargs)

            for h in handles:
                h.remove()

            loss, cond_nums = g.fasterquant_blockwise(
                blocksize=B, percdamp=percdamp, log_condition=True
            )
            g.free()

            mean_cond = sum(cond_nums) / len(cond_nums) if cond_nums else float("nan")
            max_cond  = max(cond_nums) if cond_nums else float("nan")

            print(f"  {B:>5d}  {proj_name:<24s}  {loss:>12.4f}  "
                  f"{mean_cond:>12.2e}  {max_cond:>12.2e}")

            b_total_loss += loss

        sweep_results[B] = b_total_loss

    # Restore original weights (sweep modified them in-place)
    # Caller will reload from disk before the full quantization run.
    best_B = min(sweep_results, key=sweep_results.get)
    print(f"\n  Losses by B: " +
          "  ".join(f"B={b}:{v:.2f}" for b, v in sweep_results.items()))
    print(f"  → Recommended blocksize: {best_B}")
    return best_B, sweep_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    model_stem = args.model_path.name   # e.g. "gpt-oss-20b-BF16" or "DeepSeek-V2-Lite"
    fmt_tag    = args.quant_format.upper()

    print("=" * 68)
    print(f"Stage 5 — Quantize {model_stem} → {fmt_tag}")
    print("=" * 68)
    print(f"Model path      : {args.model_path}")
    print(f"Quant format    : {args.quant_format}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Dataset         : {args.dataset}  (calibration)")
    print(f"n_calib         : {args.n_calib} samples × {args.seq_len} tokens")
    print(f"Blocksize       : {args.blocksize}")
    print(f"Percdamp        : {args.percdamp}")
    print(f"Blocksize search    : {args.blocksize_search}")
    print(f"Parallel Hessian    : {not args.no_parallel_hessian}  "
          f"({'cascade fix ON' if not args.no_parallel_hessian else 'sequential, cascade-prone'})")
    thresh_str = str(args.mixed_precision_threshold) if args.mixed_precision_threshold > 0 else "disabled (quantize all)"
    print(f"Mixed-prec thresh   : {thresh_str}")

    # ── Validation ────────────────────────────────────────────────────────────
    if args.blocksize % 16 != 0:
        print(f"\nERROR: --blocksize {args.blocksize} is not a multiple of 16.")
        sys.exit(1)

    if not args.model_path.exists():
        print(f"\nERROR: model not found: {args.model_path}")
        print("Make sure the model weights are present at that path.")
        sys.exit(1)

    # ── Imports ───────────────────────────────────────────────────────────────
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from apply import gptq_quantize_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU   : {props.name}, {props.total_memory / 1024**3:.0f} GB")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer and model {model_stem!r} (BF16)...", flush=True)
    tokenizer  = AutoTokenizer.from_pretrained(str(args.model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f} s  "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f} B params)")

    model_name = str(args.model_path)   # full path — calibration.py uses this for tokenizer

    # ── Optional: blocksize sweep ─────────────────────────────────────────────
    sweep_results = None
    chosen_blocksize = args.blocksize

    if args.blocksize_search:
        best_B, sweep_results = blocksize_sweep(
            model, model_name, device, args.percdamp, quant_format=args.quant_format
        )
        print(f"\nUsing blocksize={best_B} (from sweep; "
              f"override with --blocksize N --no-blocksize_search)")
        chosen_blocksize = best_B

        # Reload model — sweep modified weights in-place
        print("Reloading model after sweep (weights were modified)...", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

    # ── Full-model quantization ───────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"Running gptq_quantize_model(")
    print(f"    quant_format = '{args.quant_format}'")
    print(f"    dataset      = '{args.dataset}'")
    print(f"    nsamples     = {args.n_calib}")
    print(f"    blocksize    = {chosen_blocksize}")
    print(f"    mode         = 'blockwise'")
    print(f"    mixed_precision_threshold = {args.mixed_precision_threshold}")
    print(f")")
    print(f"{'─'*68}\n", flush=True)

    t_quant = time.perf_counter()

    mixed_threshold = args.mixed_precision_threshold if args.mixed_precision_threshold > 0 else None

    try:
        all_quantizers, all_layer_losses, _ = gptq_quantize_model(
            model,
            model_name,
            quant_format               = args.quant_format,
            dataset                    = args.dataset,
            nsamples                   = args.n_calib,
            seqlen                     = args.seq_len,
            mode                       = "blockwise",
            blocksize                  = chosen_blocksize,
            percdamp                   = args.percdamp,
            parallel_hessian           = not args.no_parallel_hessian,
            mixed_precision_threshold  = mixed_threshold,
        )
    except Exception as e:
        print(f"\n[ERROR] gptq_quantize_model failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    quant_time = time.perf_counter() - t_quant

    # Compute total GPTQ loss (sum of all per-sublayer losses, skipping RTN layers)
    import math as _math
    total_loss = sum(
        v
        for layer_losses in all_layer_losses.values()
        for v in layer_losses.values()
        if isinstance(v, float) and not _math.isnan(v)
    )

    print(f"\n  Total GPTQ loss      : {total_loss:.4f}")
    print(f"  Quantization time    : {quant_time / 60:.1f} min")

    # ── Save quantized model ──────────────────────────────────────────────────
    print(f"\nSaving to {args.output_dir} ...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_save = time.perf_counter()
    model.save_pretrained(str(args.output_dir))   # model is quantized in-place
    tokenizer.save_pretrained(str(args.output_dir))
    # Copy custom model code files (e.g. configuration_deepseek.py)
    # that save_pretrained() does not copy automatically.
    import shutil as _shutil
    for _f in args.model_path.iterdir():
        if _f.suffix == ".py" and not (args.output_dir / _f.name).exists():
            _shutil.copy2(_f, args.output_dir / _f.name)
    
    print(f"  Saved in {time.perf_counter() - t_save:.1f} s")

    # ── Write results JSON ────────────────────────────────────────────────────
    from datetime import datetime, timezone

    args.results.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "stage":           "5_quantize",
        "model":           model_stem,
        "model_path":      str(args.model_path),
        "output_dir":      str(args.output_dir),
        "quant_format":    args.quant_format,
        "dataset":         args.dataset,
        "n_calib":         args.n_calib,
        "seq_len":         args.seq_len,
        "blocksize":       chosen_blocksize,
        "percdamp":        args.percdamp,
        "parallel_hessian":          not args.no_parallel_hessian,
        "mixed_precision_threshold": mixed_threshold,
        "total_gptq_loss":           round(total_loss, 6),
        "quant_time_s":    round(quant_time, 2),
        "blocksize_sweep": sweep_results,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }
    args.results.write_text(json.dumps(results, indent=2))
    print(f"  Results written to {args.results}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"Stage 5 summary — {model_stem} → {fmt_tag}")
    print("=" * 68)
    print(f"  GPTQ loss (total)  : {total_loss:.4f}")
    print(f"  Blocksize used     : {chosen_blocksize}")
    print(f"  Calibration        : {args.dataset}, {args.n_calib} samples")
    print(f"  Quantization time  : {quant_time / 60:.1f} min")
    print(f"  Quantized model    : {args.output_dir}")
    print("\n✓  Model quantized. Run Stage 6 to evaluate perplexity.")
    sys.exit(0)


if __name__ == "__main__":
    main()