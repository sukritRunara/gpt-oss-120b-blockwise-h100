#!/usr/bin/env python3
"""
benchmark_nvfp4_vs_bf16.py

Compare NVFP4-quantized model vs BF16 baseline on:
  1. Quality   — greedy-decode outputs side-by-side (temperature=0)
  2. Throughput — output tokens/s at batch sizes 1 / 4 / 8
  3. TTFT      — time-to-first-token (single request, ms)
  4. Memory    — peak GPU memory after model load

Architecture
────────────
vLLM V1 runs its engine in a child process.  If two LLM objects are created
in the same Python process, the first child process does not release GPU
memory when the LLM is deleted, causing the second load to OOM.

This script avoids that by running each model in its own subprocess that
fully exits before the next one starts.  Results are exchanged via a small
JSON tempfile.

Usage:
    python benchmark_nvfp4_vs_bf16.py \\
        --bf16_model   /path/to/DeepSeek-V2-Lite \\
        --nvfp4_model  models/DeepSeek-V2-Lite-NVFP4-modelopt-v5

Options:
    --enforce_eager              Disable CUDAGraphs (faster startup)
    --max_model_len 2048
    --gpu_memory_utilization 0.88
    --n_runs 5                   Timed repetitions per batch size
    --n_warmup 2
    --max_new_tokens 80
    --skip_quality / --skip_throughput / --skip_ttft
"""

import argparse
import datetime
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "def fibonacci(n):\n    ",
    "The largest planet in the solar system is",
    "The speed of light in a vacuum is approximately",
    "In Python, a list comprehension that squares numbers 1-10 is:",
]

THROUGHPUT_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "What are the main differences between supervised and unsupervised learning?",
    "Write a Python function that checks if a string is a palindrome.",
    "Describe the water cycle and its importance to the ecosystem.",
    "What is the difference between a list and a tuple in Python?",
    "Explain how transformers work in natural language processing.",
    "What are the SOLID principles in software engineering?",
    "Write a SQL query to find the top 5 highest-paid employees.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess launcher — each model runs in its own process so GPU is fully freed
# ─────────────────────────────────────────────────────────────────────────────

def run_in_subprocess(model_path: str, args_dict: dict, label: str) -> dict:
    import re

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = f.name

    # Build a fully self-contained script — no import from parent module needed.
    quality_prompts_repr    = repr(QUALITY_PROMPTS)
    throughput_prompts_repr = repr(THROUGHPUT_PROMPTS)
    code = f"""
import json, time, sys
from pathlib import Path

QUALITY_PROMPTS    = {quality_prompts_repr}
THROUGHPUT_PROMPTS = {throughput_prompts_repr}

from vllm import LLM, SamplingParams

model_path = {model_path!r}
out_json   = {out_json!r}
args_dict  = {args_dict!r}

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=args_dict["max_model_len"],
    gpu_memory_utilization=args_dict["gpu_memory_utilization"],
    enforce_eager=args_dict["enforce_eager"],
    disable_log_stats=True,
)

results = {{"peak_mem_gb": 0.0, "quality": [], "throughput": [], "ttft_ms": None}}

greedy  = SamplingParams(temperature=0.0, max_tokens=args_dict["max_new_tokens"])
greedy1 = SamplingParams(temperature=0.0, max_tokens=1)
n_w, n_r = args_dict["n_warmup"], args_dict["n_runs"]

if not args_dict["skip_quality"]:
    outs = llm.generate(QUALITY_PROMPTS, greedy, use_tqdm=False)
    results["quality"] = [o.outputs[0].text for o in outs]

if not args_dict["skip_ttft"]:
    for _ in range(n_w):
        llm.generate([THROUGHPUT_PROMPTS[0]], greedy1, use_tqdm=False)
    t0 = time.perf_counter()
    for _ in range(n_r):
        llm.generate([THROUGHPUT_PROMPTS[0]], greedy1, use_tqdm=False)
    results["ttft_ms"] = (time.perf_counter() - t0) / n_r * 1000

if not args_dict["skip_throughput"]:
    for bs in [1, 4, 8]:
        prompts = THROUGHPUT_PROMPTS[:bs]
        for _ in range(n_w):
            llm.generate(prompts, greedy, use_tqdm=False)
        t0 = time.perf_counter()
        for _ in range(n_r):
            outs = llm.generate(prompts, greedy, use_tqdm=False)
        elapsed = (time.perf_counter() - t0) / n_r
        total_out = sum(len(o.outputs[0].token_ids) for o in outs)
        results["throughput"].append({{
            "batch_size": bs,
            "decode_tps": total_out / elapsed,
            "elapsed_s": elapsed,
        }})

Path(out_json).write_text(json.dumps(results, indent=2))
"""

    print(f"\n{'─'*60}")
    print(f"Running {label} benchmark in subprocess …")
    print(f"  Model: {model_path}")
    t0 = time.perf_counter()

    # Stream stdout+stderr line-by-line so the user sees vLLM logs live.
    # Simultaneously parse vLLM's memory log line to extract model weight GiB.
    # vLLM logs (any version): "model weights take X.XX GiB" or
    #   "memory_usage_post_profile=X.XX GiB" or "weights_memory=X.XX GiB"
    MEM_PATTERNS = [
        re.compile(r'Model loading took ([\d.]+)\s*GiB'),   # most accurate: actual in-memory weight size
        re.compile(r'Checkpoint size:\s*([\d.]+)\s*GiB'),   # fallback: on-disk size
        re.compile(r'model weights take ([\d.]+)\s*GiB'),
        re.compile(r'memory_usage_post_profile=([\d.]+)\s*GiB'),
        re.compile(r'weights_memory=([\d.]+)\s*GiB'),
    ]
    parsed_mem_gib = None

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr → stdout so we see everything
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        if parsed_mem_gib is None:
            for pat in MEM_PATTERNS:
                m = pat.search(line)
                if m:
                    parsed_mem_gib = float(m.group(1))
                    break
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")
    if parsed_mem_gib is not None:
        print(f"  Model weights memory (from vLLM log): {parsed_mem_gib:.2f} GiB")

    result = json.loads(Path(out_json).read_text())
    if parsed_mem_gib is not None:
        result["peak_mem_gb"] = parsed_mem_gib
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "─" * 68
SEP2 = "═" * 68


def _format_report(bf16: dict, nvfp4: dict, args) -> str:
    out = io.StringIO()
    w = lambda s="": print(s, file=out)

    w(f"\n\n{SEP2}")
    w("BENCHMARK RESULTS")
    w(SEP2)

    # Memory
    bm = bf16["peak_mem_gb"]
    nm = nvfp4["peak_mem_gb"]
    w(f"\nPEAK GPU MEMORY")
    w(f"  BF16  : {bm:.2f} GiB")
    w(f"  NVFP4 : {nm:.2f} GiB")
    if bm > 0:
        w(f"  Saved : {bm - nm:.2f} GiB  ({(1 - nm/bm)*100:.1f}% reduction)")
    else:
        w(f"  Saved : n/a  (nvidia-smi unavailable or returned 0)")

    # TTFT
    if bf16.get("ttft_ms") and nvfp4.get("ttft_ms"):
        bt, nt = bf16["ttft_ms"], nvfp4["ttft_ms"]
        w(f"\nTIME-TO-FIRST-TOKEN  (ms, single request)")
        w(f"  BF16  : {bt:.1f} ms")
        w(f"  NVFP4 : {nt:.1f} ms")
        w(f"  Delta : {bt - nt:+.1f} ms  ({bt/nt:.2f}x faster)")

    # Throughput
    if bf16.get("throughput") and nvfp4.get("throughput"):
        w(f"\nTHROUGHPUT  (output tokens/s)")
        w(f"  {'Batch':>6}  {'BF16':>10}  {'NVFP4':>10}  {'Speedup':>8}")
        w(f"  {SEP[:50]}")
        for b_entry, n_entry in zip(bf16["throughput"], nvfp4["throughput"]):
            bs = b_entry["batch_size"]
            speedup = n_entry["decode_tps"] / b_entry["decode_tps"]
            w(f"  {bs:>6}  {b_entry['decode_tps']:>10.1f}  "
              f"{n_entry['decode_tps']:>10.1f}  {speedup:>7.2f}x")

    # Quality
    if bf16.get("quality") and nvfp4.get("quality"):
        w(f"\nQUALITY  (greedy, temperature=0)")
        w(SEP)
        for prompt, bo, no in zip(QUALITY_PROMPTS, bf16["quality"], nvfp4["quality"]):
            match = "✓" if bo.strip() == no.strip() else "✗"
            w(f"\n  [{match}] Prompt: {prompt!r}")
            w(f"      BF16  : {bo.strip()!r}")
            w(f"      NVFP4 : {no.strip()!r}")

    w(f"\n{SEP2}\n")
    return out.getvalue()


def print_report(bf16: dict, nvfp4: dict, args):
    print(_format_report(bf16, nvfp4, args))


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(bf16: dict, nvfp4: dict, args):
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eager_tag  = "eager" if args.enforce_eager else "cudagraph"
    stem       = f"stage8_benchmark_{eager_tag}_{timestamp}"

    # JSON — full raw data
    json_path = results_dir / f"{stem}.json"
    payload = {
        "timestamp":              timestamp,
        "enforce_eager":          args.enforce_eager,
        "max_model_len":          args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "n_runs":                 args.n_runs,
        "n_warmup":               args.n_warmup,
        "max_new_tokens":         args.max_new_tokens,
        "bf16_model":             args.bf16_model,
        "nvfp4_model":            args.nvfp4_model,
        "bf16":                   bf16,
        "nvfp4":                  nvfp4,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # TXT — human-readable comparison report
    txt_path = results_dir / f"{stem}.txt"
    txt_path.write_text(_format_report(bf16, nvfp4, args))

    print(f"Results saved → {json_path}")
    print(f"Report  saved → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bf16_model",  required=True)
    p.add_argument("--nvfp4_model", required=True)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    p.add_argument("--n_runs",  type=int, default=5)
    p.add_argument("--n_warmup", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--skip_quality",    action="store_true")
    p.add_argument("--skip_throughput", action="store_true")
    p.add_argument("--skip_ttft",       action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()

    args_dict = {
        "max_model_len":          args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager":          args.enforce_eager,
        "n_runs":                 args.n_runs,
        "n_warmup":               args.n_warmup,
        "max_new_tokens":         args.max_new_tokens,
        "skip_quality":           args.skip_quality,
        "skip_throughput":        args.skip_throughput,
        "skip_ttft":              args.skip_ttft,
    }

    print(SEP2)
    print("NVFP4 vs BF16 Benchmark")
    print(SEP2)
    print(f"  BF16  model : {args.bf16_model}")
    print(f"  NVFP4 model : {args.nvfp4_model}")
    for k, v in args_dict.items():
        print(f"  {k:<28}: {v}")

    # Each model runs in its own subprocess so GPU memory is fully released
    bf16_results  = run_in_subprocess(args.bf16_model,  args_dict, "BF16")
    nvfp4_results = run_in_subprocess(args.nvfp4_model, args_dict, "NVFP4")

    print_report(bf16_results, nvfp4_results, args)
    save_results(bf16_results, nvfp4_results, args)


if __name__ == "__main__":
    main()