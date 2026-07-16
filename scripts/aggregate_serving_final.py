#!/usr/bin/env python
"""Aggregate ALL serving arms (night1 + night2 full-NVFP4) into the final
comparison table (JSON + MD). Night-1 rows are reused as-is; the full-NVFP4
C and D arms come from the night2-fullnvfp4 runs after the P0.10 fix."""

import glob
import json
import sys
from pathlib import Path

# (model dir name, run label glob, display label)
ARMS = [
    ("gpt-oss-20b-official-mxfp4", "night1", "A (official MXFP4)"),
    ("gpt-oss-20b-mxfp4-dequant-bf16", "night1", "B (BF16)"),
    ("gpt-oss-20b-mxfp4-dequant-rtn-nvfp4", "night2-fullnvfp4",
     "C (RTN NVFP4)"),
    ("gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4", "night2-fullnvfp4",
     "D (GPTQ NVFP4)"),
    ("gpt-oss-20b-gptq-nvfp4-HYBRID-experts-bf16", "night1",
     "D-hybrid (historical)"),
]


def latest_summary(model, run):
    runs = sorted(glob.glob(f"/workspace/results/serving/{model}_{run}_*"))
    if not runs:
        return None
    return json.load(open(Path(runs[-1]) / "summary.json"))


def main():
    table = {}
    for model, run, label in ARMS:
        s = latest_summary(model, run)
        if s is None:
            print(f"missing: {model} ({run})")
            continue
        table[label] = {
            f"{c['suite']}_in{c['in_len']}_c{c['concurrency']}": {
                "ttft_p50_s": c["ttft_p50_s"], "ttft_p99_s": c["ttft_p99_s"],
                "e2e_p50_s": c["e2e_p50_s"],
                "out_tok_s": c["output_tok_per_s"],
                "total_tok_s": c["total_tok_per_s"],
                "fails": c["n_fail"],
            } for c in s["cells"]
        }
    out = Path("/workspace/results/serving/comparison_final.json")
    out.write_text(json.dumps(table, indent=2))

    lines = ["| Cell | " + " | ".join(table.keys()) + " |",
             "|------|" + "---|" * len(table)]
    keys = sorted({k for v in table.values() for k in v})
    for k in keys:
        row = [k]
        for label in table:
            c = table[label].get(k)
            row.append(f"{c['ttft_p50_s']:.3f}s / {c['out_tok_s']:.0f}t/s"
                       if c else "—")
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines)
    Path("/workspace/results/serving/comparison_final.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
