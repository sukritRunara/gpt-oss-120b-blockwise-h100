"""
Stage 6 — Perplexity Evaluation of the Quantized NVFP4 Model

Loads the NVFP4-quantized model saved by Stage 5, evaluates perplexity on
WikiText-2 and C4 using the same cached samples as Stage 4, then compares
against the Stage 4 BF16 baseline and reports Δppl.

Output filenames are derived from the quantized model folder name, mirroring
the Stage 4/5 naming convention. Stripping the "-NVFP4" suffix from the
quantized model stem locates the matching Stage 4 files automatically.

  Quantized model  models/<stem>-<FORMAT>   (e.g. models/MyModel-NVFP4)
  BF16 stem        <stem>          (= model folder name minus "-<FORMAT>")
  Baseline         results/stage4_<stem>_baseline.json
  WT2 cache        results/stage4_<stem>_wikitext2_samples.json
  C4  cache        results/stage4_<stem>_c4_samples.json
  Output           results/stage6_<stem>-<FORMAT>_eval.json

Expected outcome:
  WikiText-2 Δppl ≲ 1.5   (GPTQ NVFP4 typically < 1.0 ppl degradation)
  C4         Δppl ≲ 2.0

IMPORTANT — evaluation method:
  Uses the same non-overlapping-window method as Stage 4: all texts are
  concatenated and evaluated in fixed seq_len windows. Per-sample tokenization
  would give systematically different numbers and make Δppl meaningless.

Usage:
    python stage6_eval_perplexity.py --model_path models/<stem>-NVFP4

    # Override cached file paths explicitly:
    python stage6_eval_perplexity.py \\
        --model_path   models/<stem>-NVFP4 \\
        --baseline     results/stage4_<stem>_baseline.json \\
        --seq_len      2048

    # Skip WikiText-2 (C4 only):
    python stage6_eval_perplexity.py --model_path models/<stem>-NVFP4 --skip_wikitext2

Exit: 0 = success, 1 = failure or error
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]   # tests/ → repo root
# Repo-relative code root (P0.1 fix): the library lives at
# <repo>/opteam-blockwise-gptq regardless of where the repo is checked out.
_CODE_ROOT = Path(__file__).resolve().parents[1] / "opteam-blockwise-gptq"

if not _CODE_ROOT.exists():
    raise RuntimeError(f"Code root not found: {_CODE_ROOT}")
sys.path.insert(0, str(_CODE_ROOT))

# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model_path", type=Path,
        required=True,
        help="Path to the quantized model saved by Stage 5 "
             "(e.g. models/gpt-oss-20b-BF16-NVFP4, models/MyModel-INT8)",
    )
    p.add_argument(
        "--quant_format", default=None,
        help="Quantization format tag used to strip the suffix from the model "
             "folder name (e.g. 'nvfp4' → strips '-NVFP4'). "
             "Auto-detected from folder name if omitted.",
    )
    p.add_argument(
        "--baseline", type=Path, default=None,
        help="Stage 4 JSON baseline "
             "(default: results/stage4_<bf16_stem>_baseline.json)",
    )
    p.add_argument(
        "--wt2_samples", type=Path, default=None,
        help="Stage 4 cached WikiText-2 samples "
             "(default: results/stage4_<bf16_stem>_wikitext2_samples.json)",
    )
    p.add_argument(
        "--c4_samples", type=Path, default=None,
        help="Stage 4 cached C4 samples "
             "(default: results/stage4_<bf16_stem>_c4_samples.json)",
    )
    p.add_argument(
        "--seq_len", type=int, default=2048,
        help="Evaluation window size in tokens (default: 2048)",
    )
    p.add_argument(
        "--c4_tokens", type=int, default=131_072,
        help="Max tokens to evaluate on C4 (default: 131072 = 64 windows)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Path for the JSON results file "
             "(default: results/stage6_<quant_stem>_eval.json)",
    )
    p.add_argument(
        "--skip_wikitext2", action="store_true",
        help="Skip WikiText-2 evaluation and only run C4.",
    )
    p.add_argument(
        "--warn_delta_wt2", type=float, default=1.5,
        help="Warn if WikiText-2 Δppl exceeds this (default: 1.5)",
    )
    p.add_argument(
        "--warn_delta_c4", type=float, default=2.0,
        help="Warn if C4 Δppl exceeds this (default: 2.0)",
    )
    args = p.parse_args()

    # Derive stems and default paths from model folder name
    quant_stem = args.model_path.name   # e.g. "gpt-oss-20b-BF16-NVFP4"

    # Determine the format suffix to strip.
    # If --quant_format is given, use it directly (e.g. "nvfp4" → "-NVFP4").
    # Otherwise auto-detect by trying all known format tags in order.
    _known_tags = ["NVFP4", "MXINT4", "INT4_PERCHANNEL", "INT4", "INT8", "FP8"]
    if args.quant_format is not None:
        fmt_suffix = f"-{args.quant_format.upper()}"
        bf16_stem  = quant_stem.removesuffix(fmt_suffix)
    else:
        bf16_stem = quant_stem
        for tag in _known_tags:
            if quant_stem.endswith(f"-{tag}"):
                bf16_stem = quant_stem[: -len(f"-{tag}")]
                break

    if args.baseline is None:
        args.baseline = _REPO_ROOT / "results" / f"stage4_{bf16_stem}_baseline.json"
    if args.wt2_samples is None:
        args.wt2_samples = _REPO_ROOT / "results" / f"stage4_{bf16_stem}_wikitext2_samples.json"
    if args.c4_samples is None:
        args.c4_samples  = _REPO_ROOT / "results" / f"stage4_{bf16_stem}_c4_samples.json"
    if args.output is None:
        args.output = _REPO_ROOT / "results" / f"stage6_{quant_stem}_eval.json"

    # Attach stems for use in main()
    args._quant_stem = quant_stem
    args._bf16_stem  = bf16_stem

    return args


# ── Perplexity evaluation ──────────────────────────────────────────────────────

def eval_perplexity(model, tokenizer, texts, seq_len, max_tokens=None, label=""):
    """Compute perplexity via non-overlapping windows over concatenated text.

    Identical to the Stage 4 method so that BF16 and NVFP4 results are
    measured on exactly the same token sequence in the same window positions.

    Args:
        model:      HuggingFace causal LM (already on device, in eval mode).
        tokenizer:  Matching tokenizer.
        texts:      List of strings (same cache as Stage 4).
        seq_len:    Window size in tokens.
        max_tokens: If set, truncate evaluation to this many tokens.
        label:      Display name for progress output.

    Returns:
        dict with keys: ppl, avg_nll, n_tokens, n_windows, elapsed_s
    """
    import torch

    full_text = "\n\n".join(t for t in texts if t.strip())
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[0]   # [total_tokens]

    total_tokens = input_ids.size(0)
    if max_tokens is not None:
        total_tokens = min(total_tokens, max_tokens)
        input_ids = input_ids[:total_tokens]

    n_windows = total_tokens // seq_len
    if n_windows == 0:
        raise ValueError(
            f"Not enough tokens ({total_tokens}) for even one window "
            f"of length {seq_len}."
        )

    device = next(model.parameters()).device
    nlls = []
    t0   = time.perf_counter()

    for w in range(n_windows):
        start = w * seq_len
        end   = start + seq_len
        ids   = input_ids[start:end].unsqueeze(0).to(device)

        with torch.no_grad():
            out  = model(ids, labels=ids)
            nlls.append(out.loss.item())

        if (w + 1) % 10 == 0 or (w + 1) == n_windows:
            elapsed  = time.perf_counter() - t0
            avg_nll  = sum(nlls) / len(nlls)
            cur_ppl  = math.exp(avg_nll)
            progress = (w + 1) / n_windows * 100
            tok_done = (w + 1) * seq_len
            print(f"  {label}  window {w+1:>4d}/{n_windows}  "
                  f"({progress:5.1f}%)  tokens={tok_done:>8,}  "
                  f"ppl={cur_ppl:8.3f}  [{elapsed:6.1f}s]",
                  flush=True)

    elapsed  = time.perf_counter() - t0
    avg_nll  = sum(nlls) / len(nlls)
    ppl      = math.exp(avg_nll)

    return {
        "ppl":       ppl,
        "avg_nll":   avg_nll,
        "n_tokens":  n_windows * seq_len,
        "n_windows": n_windows,
        "elapsed_s": elapsed,
    }


# ── Dataset fallbacks (if Stage 4 cache is missing) ───────────────────────────

def load_wikitext2_fallback():
    from datasets import load_dataset
    print("  Streaming WikiText-2 test split...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test", trust_remote_code=False)
    return ds["text"]


def load_c4_fallback(max_tokens):
    from datasets import load_dataset
    n_samples = max(256, (max_tokens // 256) * 2)
    print(f"  Streaming C4 validation ({n_samples} samples)...", flush=True)
    ds = load_dataset("allenai/c4", "en",
                      split="validation",
                      streaming=True,
                      trust_remote_code=False)
    texts = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        texts.append(row["text"])
    print(f"  Fetched {len(texts)} samples", flush=True)
    return texts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    print("=" * 68)
    print(f"Stage 6 — NVFP4 Perplexity Evaluation  [{args._quant_stem}]")
    print("=" * 68)
    print(f"Quantized model    : {args.model_path}")
    print(f"BF16 stem          : {args._bf16_stem}")
    print(f"Baseline           : {args.baseline}")
    print(f"WikiText-2 samples : {args.wt2_samples}")
    print(f"C4 samples         : {args.c4_samples}")
    print(f"Seq len            : {args.seq_len}")
    print(f"C4 tokens          : {args.c4_tokens:,}")
    print(f"Skip WikiText-2    : {args.skip_wikitext2}")
    print(f"Output             : {args.output}")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    errors = []

    if not args.model_path.exists():
        errors.append(
            f"Quantized model not found: {args.model_path}\n"
            "  → Run Stage 5 first: python stage5_quantize_model.py"
        )
    elif not (args.model_path / "config.json").exists():
        errors.append(
            f"config.json missing in {args.model_path}\n"
            "  → Stage 5 may not have completed successfully."
        )

    if not args.baseline.exists():
        errors.append(
            f"Stage 4 baseline not found: {args.baseline}\n"
            f"  → Run Stage 4 first: python stage4_baseline_perplexity.py "
            f"--model_path models/{args._bf16_stem}"
        )

    if errors:
        for e in errors:
            print(f"\nERROR: {e}")
        sys.exit(1)

    # ── Load baseline ─────────────────────────────────────────────────────────
    baseline     = json.loads(args.baseline.read_text())
    base_wt2_ppl = baseline["wikitext2"]["ppl"]
    base_c4_ppl  = baseline["c4"]["ppl"]

    print(f"\nBaseline (Stage 4 BF16 — {args._bf16_stem}):")
    print(f"  WikiText-2 ppl : {base_wt2_ppl:.4f}")
    print(f"  C4         ppl : {base_c4_ppl:.4f}")

    # ── Load cached samples ───────────────────────────────────────────────────
    if not args.skip_wikitext2:
        if args.wt2_samples.exists():
            print(f"\nLoading cached WikiText-2 samples...", flush=True)
            wt2_texts = json.loads(args.wt2_samples.read_text())
            print(f"  Loaded {len(wt2_texts)} samples")
        else:
            print(f"\nWARN: {args.wt2_samples} not found — falling back to streaming.")
            print("  Re-run Stage 4 to generate the cache for a fair comparison.")
            wt2_texts = load_wikitext2_fallback()

    if args.c4_samples.exists():
        print(f"\nLoading cached C4 samples...", flush=True)
        c4_texts = json.loads(args.c4_samples.read_text())
        print(f"  Loaded {len(c4_texts)} samples")
    else:
        print(f"\nWARN: {args.c4_samples} not found — falling back to streaming.")
        print("  Re-run Stage 4 to generate the cache for a fair comparison.")
        c4_texts = load_c4_fallback(args.c4_tokens)

    # ── Load tokenizer + model ────────────────────────────────────────────────
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size   : {tokenizer.vocab_size:,}")
    print(f"  BOS token ID : {tokenizer.bos_token_id}")

    print(f"\nLoading quantized model {args._quant_stem!r} (device_map=auto)...",
          flush=True)
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.perf_counter() - t_load

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params / 1e9:.2f} B")
    print(f"  Load time  : {load_time:.1f} s")
    if torch.cuda.is_available():
        print(f"  GPU memory : {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

    # ── WikiText-2 evaluation ─────────────────────────────────────────────────
    wt2 = None
    if not args.skip_wikitext2:
        print("\n" + "─" * 68)
        print(f"WikiText-2 ({len(wt2_texts)} cached samples, non-overlapping windows)")
        print("─" * 68)

        wt2 = eval_perplexity(
            model, tokenizer, wt2_texts,
            seq_len=args.seq_len,
            label="wikitext2",
        )

        delta_wt2 = wt2["ppl"] - base_wt2_ppl
        wt2_ok    = delta_wt2 <= args.warn_delta_wt2
        print(f"\n  NVFP4 ppl : {wt2['ppl']:.4f}")
        print(f"  BF16  ppl : {base_wt2_ppl:.4f}")
        print(f"  Δppl      : {delta_wt2:+.4f}  "
              f"({'OK' if wt2_ok else 'WARN — above threshold'})")
    else:
        delta_wt2 = None
        wt2_ok    = True

    # ── C4 evaluation ─────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print(f"C4 ({len(c4_texts)} cached samples, up to {args.c4_tokens:,} tokens)")
    print("─" * 68)

    c4 = eval_perplexity(
        model, tokenizer, c4_texts,
        seq_len=args.seq_len,
        max_tokens=args.c4_tokens,
        label="c4         ",
    )

    delta_c4 = c4["ppl"] - base_c4_ppl
    c4_ok    = delta_c4 <= args.warn_delta_c4
    print(f"\n  NVFP4 ppl : {c4['ppl']:.4f}")
    print(f"  BF16  ppl : {base_c4_ppl:.4f}")
    print(f"  Δppl      : {delta_c4:+.4f}  "
          f"({'OK' if c4_ok else 'WARN — above threshold'})")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "stage":          "6_nvfp4_eval",
        "model":          args._quant_stem,
        "model_path":     str(args.model_path),
        "bf16_stem":      args._bf16_stem,
        "quant_format":   "nvfp4",
        "seq_len":        args.seq_len,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "baseline_path":  str(args.baseline),
        "wikitext2": (
            {
                "ppl_nvfp4":  round(wt2["ppl"], 6),
                "ppl_bf16":   round(base_wt2_ppl, 6),
                "delta_ppl":  round(delta_wt2, 6),
                "avg_nll":    round(wt2["avg_nll"], 6),
                "n_tokens":   wt2["n_tokens"],
                "n_windows":  wt2["n_windows"],
                "elapsed_s":  round(wt2["elapsed_s"], 2),
            } if wt2 is not None else "skipped"
        ),
        "c4": {
            "ppl_nvfp4":  round(c4["ppl"], 6),
            "ppl_bf16":   round(base_c4_ppl, 6),
            "delta_ppl":  round(delta_c4, 6),
            "avg_nll":    round(c4["avg_nll"], 6),
            "n_tokens":   c4["n_tokens"],
            "n_windows":  c4["n_windows"],
            "elapsed_s":  round(c4["elapsed_s"], 2),
        },
        "thresholds": {
            "warn_delta_wt2": args.warn_delta_wt2,
            "warn_delta_c4":  args.warn_delta_c4,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"Stage 6 Summary — {args._quant_stem}  (NVFP4 vs BF16)")
    print("=" * 68)

    print(f"  {'Dataset':<12} {'BF16 ppl':>10} {'NVFP4 ppl':>10} {'Δppl':>8}  Status")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*8}  {'─'*20}")

    if wt2 is not None:
        wt2_status = "✓  OK" if wt2_ok else "⚠  ABOVE THRESHOLD"
        print(f"  {'WikiText-2':<12} {base_wt2_ppl:>10.4f} {wt2['ppl']:>10.4f} "
              f"{delta_wt2:>+8.4f}  {wt2_status}")
    else:
        print(f"  {'WikiText-2':<12} {'—':>10} {'skipped':>10} {'—':>8}")

    c4_status = "✓  OK" if c4_ok else "⚠  ABOVE THRESHOLD"
    print(f"  {'C4':<12} {base_c4_ppl:>10.4f} {c4['ppl']:>10.4f} "
          f"{delta_c4:>+8.4f}  {c4_status}")

    print(f"\n  Results saved to: {args.output}")

    any_warn = not c4_ok or not wt2_ok
    if any_warn:
        print("\n⚠  One or more Δppl values exceeded the warning threshold.")
        print("   Consider: lower percdamp, larger nsamples, or blocksize tuning.")
    else:
        print("\n✓  NVFP4 quantization quality is within expected range.")

    sys.exit(0)


if __name__ == "__main__":
    main()