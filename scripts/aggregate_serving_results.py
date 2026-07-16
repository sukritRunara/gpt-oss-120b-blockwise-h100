#!/usr/bin/env python
"""Aggregate per-arm serving summaries into one comparison table (JSON + MD)."""

import glob
import json
import sys
from pathlib import Path

ARMS = {
    "gpt-oss-20b-official-mxfp4": "A (official MXFP4)",
    "gpt-oss-20b-mxfp4-dequant-bf16": "B (BF16)",
    "gpt-oss-20b-gptq-nvfp4-HYBRID-experts-bf16": "D-hybrid (attn NVFP4)",
}


def latest_summary(model):
    runs = sorted(glob.glob(f"/workspace/results/serving/{model}_night1_*"))
    if not runs:
        return None
    return json.load(open(Path(runs[-1]) / "summary.json"))


def main():
    table = {}
    for model, label in ARMS.items():
        s = latest_summary(model)
        if s is None:
            print(f"missing: {model}")
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
    out = Path("/workspace/results/serving/comparison_night1.json")
    out.write_text(json.dumps(table, indent=2))

    # Markdown table for the report — decode + mixed subsets
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
    Path("/workspace/results/serving/comparison_night1.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
