"""
Stage 4 — Baseline Perplexity Evaluation

Loads any HuggingFace causal LM (BF16) and measures perplexity on
WikiText-2 and C4 BEFORE any quantization. The results and sample caches
are saved to the results directory; Stage 6 loads the same cached samples
to ensure an exact BF16 vs quantized comparison.

Output filenames are derived from the model folder name so multiple models
can be evaluated without overwriting each other's results.

Requires:
  - Model weights present at --model_path
  - Internet access (first run fetches dataset shards from HuggingFace)

Usage:
    python stage4_baseline_perplexity.py [--model_path PATH] [--seq_len N]
                                         [--c4_tokens N] [--output PATH]
                                         [--skip_wikitext2]

Defaults:
    --model_path  <repo_root>/models/gpt-oss-20b-BF16
    --seq_len     2048
    --c4_tokens   131072   (64 windows × 2048 = ~2 min eval)
    --output      <repo_root>/results/stage4_<model_name>_baseline.json

Exit:  0 = success, 1 = failure
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# ── Argument parsing ──────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]  # tests/ → repo root

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=Path,
                   default=_REPO_ROOT / "models" / "gpt-oss-20b-BF16",
                   help="Path to the downloaded GPT-OSS 20B BF16 weights")
    p.add_argument("--seq_len", type=int, default=2048,
                   help="Evaluation sequence length (default: 2048)")
    p.add_argument("--c4_tokens", type=int, default=131_072,
                   help="Max tokens to evaluate on C4 (default: 131072 = 64 windows)")
    p.add_argument("--output", type=Path,
                   default=None,
                   help="Path to write the JSON results file "
                        "(default: results/stage4_<model_name>_baseline.json)")
    p.add_argument("--skip_wikitext2", action="store_true",
                   help="Skip WikiText-2 evaluation and only run C4.")
    args = p.parse_args()
    # Derive default output path from model name so different models
    # never overwrite each other's results or sample caches.
    model_stem = args.model_path.name
    if args.output is None:
        args.output = _REPO_ROOT / "results" / f"stage4_{model_stem}_baseline.json"
    return args


# ── Core evaluation ───────────────────────────────────────────────────────────

def eval_perplexity(model, tokenizer, texts, seq_len, max_tokens=None, label=""):
    """Compute perplexity via non-overlapping windows over concatenated text.

    Args:
        model:      HuggingFace causal LM (already on device, in eval mode).
        tokenizer:  Matching tokenizer.
        texts:      List of strings to concatenate and evaluate.
        seq_len:    Window size in tokens.
        max_tokens: If set, truncate evaluation to this many tokens.
        label:      Display name for progress output.

    Returns:
        dict with keys: ppl (float), n_tokens (int), n_windows (int),
                        avg_nll (float), elapsed_s (float)
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


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_wikitext2_test():
    """Return list of strings from WikiText-2 test split."""
    from datasets import load_dataset
    print("  Loading WikiText-2 test split...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                      split="test", trust_remote_code=False)
    return ds["text"]


def load_c4_validation(max_tokens, seq_len, tokenizer):
    """Return enough C4 validation samples to cover max_tokens.

    Uses streaming=True so no shard is downloaded — samples are fetched
    on-demand from the Hub. This avoids a multi-GB download stall on first run.
    """
    from datasets import load_dataset
    # Estimate how many samples we need (rough: ~256 tokens/sample)
    n_samples = max(256, (max_tokens // 256) * 2)
    print(f"  Loading C4 validation ({n_samples} samples, streaming)...",
          flush=True)
    ds = load_dataset("allenai/c4", "en",
                      split="validation",
                      streaming=True,
                      trust_remote_code=False)
    texts = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        texts.append(row["text"])
    print(f"  Fetched {len(texts)} samples via streaming", flush=True)
    return texts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = _parse_args()
    model_stem = args.model_path.name   # e.g. "gpt-oss-20b-BF16" or "DeepSeek-V2-Lite"

    print("=" * 68)
    print("Stage 4 — Baseline Perplexity Evaluation")
    print("=" * 68)
    print(f"Model path : {args.model_path}")
    print(f"Seq len    : {args.seq_len}")
    print(f"C4 tokens  : {args.c4_tokens:,}")
    print(f"Output     : {args.output}")

    # ── Pre-flight: model path ────────────────────────────────────────────────
    if not args.model_path.exists():
        print(f"\nERROR: model path not found: {args.model_path}")
        print("Make sure the model is downloaded to that path.")
        sys.exit(1)

    config_ok = (args.model_path / "config.json").exists()
    if not config_ok:
        print(f"\nERROR: config.json missing in {args.model_path}")
        print("The model download may be incomplete.")
        sys.exit(1)

    safetensors = list(args.model_path.glob("*.safetensors"))
    print(f"\nModel shards: {len(safetensors)} × .safetensors")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model {model_stem!r} (BF16, device_map=auto)...", flush=True)

    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.perf_counter() - t_load

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params / 1e9:.2f} B")
    print(f"  Load time  : {load_time:.1f} s")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"  GPU memory : {mem_gb:.1f} GB")

    # ── WikiText-2 evaluation ─────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("WikiText-2 (test split, non-overlapping 2048-token windows)")
    print("─" * 68)

    wt2_texts = load_wikitext2_test()
    _wt2_cache = args.output.parent / f"stage4_{model_stem}_wikitext2_samples.json"
    _wt2_cache.write_text(json.dumps(list(wt2_texts)))
    print(f"  Cached {len(wt2_texts)} samples → {_wt2_cache}")
    wt2 = eval_perplexity(model, tokenizer, wt2_texts,
                          seq_len=args.seq_len,
                          label="wikitext2")

    print(f"\n  WikiText-2 perplexity : {wt2['ppl']:.4f}")
    print(f"  Tokens evaluated      : {wt2['n_tokens']:,}  ({wt2['n_windows']} windows)")
    print(f"  Time                  : {wt2['elapsed_s']:.1f} s")

    # ── C4 evaluation ─────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print(f"C4 (validation, up to {args.c4_tokens:,} tokens)")
    print("─" * 68)

    c4_texts = load_c4_validation(args.c4_tokens, args.seq_len, tokenizer)
    _c4_cache = args.output.parent / f"stage4_{model_stem}_c4_samples.json"
    _c4_cache.write_text(json.dumps(c4_texts))
    print(f"  Cached {len(c4_texts)} samples → {_c4_cache}")
    c4 = eval_perplexity(model, tokenizer, c4_texts,
                         seq_len=args.seq_len,
                         max_tokens=args.c4_tokens,
                         label="c4         ")

    print(f"\n  C4 perplexity   : {c4['ppl']:.4f}")
    print(f"  Tokens evaluated: {c4['n_tokens']:,}  ({c4['n_windows']} windows)")
    print(f"  Time            : {c4['elapsed_s']:.1f} s")

    # ── Save results ──────────────────────────────────────────────────────────
    from datetime import datetime, timezone

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "stage":      "4_baseline",
        "model":      model_stem,
        "model_path": str(args.model_path),
        "dtype":      "bfloat16",
        "seq_len":    args.seq_len,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "wikitext2": {
            "ppl":       round(wt2["ppl"], 6),
            "avg_nll":   round(wt2["avg_nll"], 6),
            "n_tokens":  wt2["n_tokens"],
            "n_windows": wt2["n_windows"],
            "elapsed_s": round(wt2["elapsed_s"], 2),
        },
        "c4": {
            "ppl":       round(c4["ppl"], 6),
            "avg_nll":   round(c4["avg_nll"], 6),
            "n_tokens":  c4["n_tokens"],
            "n_windows": c4["n_windows"],
            "elapsed_s": round(c4["elapsed_s"], 2),
        },
    }

    args.output.write_text(json.dumps(results, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Stage 4 summary — Baseline perplexity")
    print("=" * 68)
    print(f"  WikiText-2 ppl : {wt2['ppl']:.4f}")
    print(f"  C4         ppl : {c4['ppl']:.4f}")
    print(f"\n  Results saved to: {args.output}")
    print("\n✓  Baseline recorded. Run Stage 5 to quantize the model.")
    sys.exit(0)


if __name__ == "__main__":
    main()