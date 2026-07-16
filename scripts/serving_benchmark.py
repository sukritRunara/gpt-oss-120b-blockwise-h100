#!/usr/bin/env python
"""Async serving benchmark against a live OpenAI-compatible vLLM server (P0.9).

True concurrent async requests via aiohttp streaming; one JSONL record per
request (handoff §18). Replaces the old single-process generate() timer.

Suites (cells are (input_len, output_len, concurrency)):
  prefill : in {1k, 8k}        × out 1   × conc {1, 8, 32}
  decode  : in 128             × out {256} × conc {1, 8, 32, 64}
  mixed   : in {1k, 8k}        × out 256 × conc {1, 8, 32, 64}
  (32k+ prompt cells require a server started with a larger --max-model-len;
   pass --suite mixed32k etc. explicitly after a capacity check.)

Prompts: built from natural-text seed paragraphs, tokenized with the model's
tokenizer, sliced to the EXACT requested token length, and prefixed with a
unique random header so no two requests share a prefix (prefix caching is
additionally disabled server-side). Requests send token IDs, so measured
input lengths are exact. Prompt SHA-256 and token counts are recorded.

Per request: TTFT, inter-token latencies, e2e latency, output token count,
success/failure. Per cell: p50/p90/p99, request & token throughput.
GPU telemetry is sampled via nvidia-smi in a sidecar thread.

Run inside .venv-serve (server already up):
    python scripts/serving_benchmark.py --base_url http://localhost:8000 \\
        --model <served-name> --label pilot --suites decode \\
        --warmup 4 --requests 16 --reps 1
"""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

SEED_TEXT = """The history of computing spans mechanical calculators, vacuum
tubes, transistors, and integrated circuits. Each generation reduced cost and
power while increasing speed. Modern datacenters host accelerators optimized
for dense linear algebra, and neural networks exploit this hardware through
batched matrix multiplication. Attention mechanisms compute pairwise token
interactions, while mixture-of-experts layers route tokens to specialized
feed-forward networks, trading memory for conditional computation. Weather
systems emerge from the interaction of solar radiation, planetary rotation,
and topography. Ocean currents transport heat between latitudes, moderating
coastal climates. Cities adapt infrastructure to seasonal extremes, from
storm drainage to district heating. Cuisine reflects geography: coastal
regions favor seafood preserved by salting and fermentation, while inland
plains developed grain staples and cured meats. Trade routes carried spices,
techniques, and crops between continents, reshaping local dishes. Legal
systems balance precedent and statute; courts interpret ambiguity while
legislatures respond with amendments. Financial markets aggregate dispersed
information into prices, though bubbles and panics reveal the limits of
rational expectation. Engineering disciplines codify safety margins learned
from failures: bridge resonance, fatigue cracks, and corrosion each taught
costly lessons now embedded in standards. """


SUITES = {
    "prefill": {"in": [1024, 8192], "out": [1], "conc": [1, 8, 32]},
    "decode":  {"in": [128],       "out": [256], "conc": [1, 8, 32, 64]},
    "mixed":   {"in": [1024, 8192], "out": [256], "conc": [1, 8, 32, 64]},
    # explicit-opt-in long-context cells (need --max-model-len >= 33k server)
    "prefill32k": {"in": [32768], "out": [1], "conc": [1, 8]},
    "mixed32k":   {"in": [32768], "out": [256], "conc": [1, 8]},
}


class GpuMonitor:
    """Samples nvidia-smi to CSV lines in a background thread."""

    def __init__(self, out_path: Path, interval: float = 1.0):
        self.out_path = out_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with open(self.out_path, "a") as f:
            f.write("ts,util.gpu,mem.used_mib,power_w,sm_clock,temp\n")
            while not self._stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,memory.used,power.draw,"
                         "clocks.sm,temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5).stdout.strip()
                    f.write(f"{time.time():.1f},{out.replace(', ', ',')}\n")
                    f.flush()
                except Exception:                          # noqa: BLE001
                    pass
                self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join(timeout=3)


def build_prompt_ids(tokenizer, target_len: int, rng: random.Random):
    """Exact-length token-ID prompt with a unique anti-prefix-cache header."""
    header = f"[req {rng.getrandbits(64):016x}] "
    header_ids = tokenizer(header, add_special_tokens=False).input_ids
    body = SEED_TEXT * (1 + target_len // 320)
    body_ids = tokenizer(body, add_special_tokens=False).input_ids
    need = target_len - len(header_ids)
    if need <= 0:
        return header_ids[:target_len]
    if len(body_ids) < need:
        body_ids = (body_ids * (need // len(body_ids) + 2))
    start = rng.randrange(0, len(body_ids) - need)
    ids = header_ids + body_ids[start:start + need]
    assert len(ids) == target_len
    return ids


async def one_request(session, base_url, model, prompt_ids, out_len, timeout_s):
    rec = {"prompt_tokens": len(prompt_ids),
           "prompt_sha": hashlib.sha256(
               json.dumps(prompt_ids).encode()).hexdigest()[:16],
           "requested_output_tokens": out_len}
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": out_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    t0 = time.perf_counter()
    token_times = []
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                rec.update(success=False,
                           error=f"http {resp.status}: {(await resp.text())[:200]}")
                return rec
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("choices") and (
                        chunk["choices"][0].get("text")
                        or chunk["choices"][0].get("finish_reason")):
                    token_times.append(time.perf_counter())
        t_end = time.perf_counter()
        if not token_times:
            rec.update(success=False, error="no tokens streamed")
            return rec
        itls = [token_times[i] - token_times[i - 1]
                for i in range(1, len(token_times))]
        rec.update(
            success=True,
            ttft_s=token_times[0] - t0,
            e2e_s=t_end - t0,
            output_chunks=len(token_times),
            itl_mean_s=(statistics.mean(itls) if itls else None),
            itl_p99_s=(sorted(itls)[max(0, int(len(itls) * 0.99) - 1)]
                       if itls else None),
        )
    except asyncio.TimeoutError:
        rec.update(success=False, error=f"timeout>{timeout_s}s")
    except Exception as exc:                               # noqa: BLE001
        rec.update(success=False, error=f"{type(exc).__name__}: {exc}")
    return rec


async def run_cell(base_url, model, tokenizer, in_len, out_len, conc,
                   n_warmup, n_requests, timeout_s, rng, jsonl_path):
    """One benchmark cell: warmup then measured requests at fixed concurrency."""
    sem = asyncio.Semaphore(conc)
    results = []

    async with aiohttp.ClientSession() as session:
        async def bounded(idx, warm):
            async with sem:
                ids = build_prompt_ids(tokenizer, in_len, rng)
                rec = await one_request(session, base_url, model, ids,
                                        out_len, timeout_s)
                rec.update(warmup=warm, idx=idx, in_len=in_len,
                           out_len=out_len, concurrency=conc,
                           t_wall=datetime.now(timezone.utc).isoformat())
                if not warm:
                    results.append(rec)
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        t_cell0 = time.perf_counter()
        await asyncio.gather(*(bounded(i, True) for i in range(n_warmup)))
        t_meas0 = time.perf_counter()
        await asyncio.gather(*(bounded(i, False) for i in range(n_requests)))
        t_meas = time.perf_counter() - t_meas0

    ok = [r for r in results if r["success"]]
    fail = len(results) - len(ok)

    def pct(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        return s[min(len(s) - 1, int(len(s) * p))]

    ttfts = [r["ttft_s"] for r in ok]
    e2es = [r["e2e_s"] for r in ok]
    itls = [r["itl_mean_s"] for r in ok if r.get("itl_mean_s") is not None]
    out_tok = sum(r["output_chunks"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)
    summary = {
        "in_len": in_len, "out_len": out_len, "concurrency": conc,
        "n_ok": len(ok), "n_fail": fail,
        "measured_wall_s": t_meas,
        "warmup_wall_s": t_meas0 - t_cell0,
        "ttft_p50_s": pct(ttfts, 0.50), "ttft_p90_s": pct(ttfts, 0.90),
        "ttft_p99_s": pct(ttfts, 0.99),
        "e2e_p50_s": pct(e2es, 0.50), "e2e_p90_s": pct(e2es, 0.90),
        "e2e_p99_s": pct(e2es, 0.99),
        "itl_mean_s": (statistics.mean(itls) if itls else None),
        "req_per_s": len(ok) / t_meas if t_meas > 0 else None,
        "output_tok_per_s": out_tok / t_meas if t_meas > 0 else None,
        "input_tok_per_s": in_tok / t_meas if t_meas > 0 else None,
        "total_tok_per_s": (in_tok + out_tok) / t_meas if t_meas > 0 else None,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path (default: --tokenizer_path or model dir "
                         "from the server is not reachable; pass explicitly)")
    ap.add_argument("--label", required=True,
                    help="run label (e.g. pilot / final-rep1)")
    ap.add_argument("--suites", nargs="+", default=["prefill", "decode", "mixed"],
                    choices=list(SUITES.keys()))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout_s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/workspace/results/serving"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok_path = args.tokenizer or "/workspace/models/gpt-oss-20b-official-mxfp4"
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / f"{args.model}_{args.label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    with GpuMonitor(run_dir / "gpu_telemetry.csv"):
        for rep in range(args.reps):
            for suite in args.suites:
                spec = SUITES[suite]
                for in_len in spec["in"]:
                    for out_len in spec["out"]:
                        for conc in spec["conc"]:
                            rng = random.Random(
                                args.seed + hash((suite, in_len, out_len,
                                                  conc, rep)) % 2**32)
                            jsonl = run_dir / (f"{suite}_in{in_len}_out{out_len}"
                                               f"_c{conc}_rep{rep}.jsonl")
                            print(f"[cell] {suite} in={in_len} out={out_len} "
                                  f"conc={conc} rep={rep} …", flush=True)
                            s = asyncio.run(run_cell(
                                args.base_url, args.model, tokenizer,
                                in_len, out_len, conc, args.warmup,
                                args.requests, args.timeout_s, rng, jsonl))
                            s.update(suite=suite, rep=rep, model=args.model,
                                     label=args.label)
                            all_summaries.append(s)
                            print(f"       ok={s['n_ok']} fail={s['n_fail']} "
                                  f"ttft_p50={s['ttft_p50_s'] and round(s['ttft_p50_s'], 4)}s "
                                  f"out_tok/s={s['output_tok_per_s'] and round(s['output_tok_per_s'], 1)}",
                                  flush=True)

    (run_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "label": args.label, "base_url": args.base_url,
        "warmup": args.warmup, "requests": args.requests, "reps": args.reps,
        "created_utc": stamp, "cells": all_summaries}, indent=2))
    print(f"\nSummary → {run_dir / 'summary.json'}")
    fails = sum(s["n_fail"] for s in all_summaries)
    print(f"Total failures: {fails}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
