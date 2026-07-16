#!/usr/bin/env python
"""Numerical A/B check: tiny QDQ model (transformers, .venv-quant writes the
reference) vs packed model (vLLM). Greedy continuations on fixed token IDs.

Two modes:
  --write_reference : run in .venv-quant against the QDQ dir
  --check           : run in .venv-serve against the packed dir
"""

import argparse
import json
import sys
from pathlib import Path

PROMPT_IDS = [
    [1000 + i * 37 for i in range(32)],
    [50000 + i * 101 for i in range(32)],
    [123, 456, 789, 1011, 1213, 1415, 1617, 1819] * 4,
]
GEN_LEN = 24


def write_reference(qdq_dir: str, out: Path):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        qdq_dir, dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True)
    model.eval()
    outs = []
    with torch.no_grad():
        for ids in PROMPT_IDS:
            t = torch.tensor([ids], device="cuda")
            g = model.generate(t, max_new_tokens=GEN_LEN, do_sample=False)
            outs.append(g[0, len(ids):].tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"greedy": outs}))
    print(f"reference → {out}")


def check(packed_dir: str, ref_path: Path):
    from vllm import LLM, SamplingParams

    ref = json.loads(ref_path.read_text())["greedy"]
    llm = LLM(model=packed_dir, max_model_len=256,
              gpu_memory_utilization=0.25, enforce_eager=True,
              disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=GEN_LEN, ignore_eos=True)
    outs = llm.generate([{"prompt_token_ids": ids} for ids in PROMPT_IDS], sp)
    ok = 0
    for o, r in zip(outs, ref):
        got = list(o.outputs[0].token_ids)
        match = sum(1 for a, b in zip(got, r) if a == b)
        first_mismatch = next((i for i, (a, b) in enumerate(zip(got, r))
                               if a != b), len(r))
        print(f"  got {got[:8]}… ref {r[:8]}… match {match}/{len(r)} "
              f"first_diff@{first_mismatch}")
        ok += int(got == r)
    print(f"FULL_MATCH {ok}/{len(ref)}")
    return 0 if ok == len(ref) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_reference", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--qdq")
    ap.add_argument("--packed")
    ap.add_argument("--ref", type=Path, required=True)
    a = ap.parse_args()
    if a.write_reference:
        write_reference(a.qdq, a.ref)
        sys.exit(0)
    sys.exit(check(a.packed, a.ref))
