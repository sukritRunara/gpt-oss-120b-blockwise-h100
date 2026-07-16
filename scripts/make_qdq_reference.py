#!/usr/bin/env python
"""Produce greedy reference continuations from a QDQ checkpoint (transformers).

Used by pilot_serving_check.py to compare the vLLM-served packed model
against the exact model GPTQ produced. Run inside .venv-quant.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "def quicksort(arr):",
    "The theory of general relativity states that",
    "To bake sourdough bread, you first need",
    "Quantum entanglement is a phenomenon where",
    "The difference between TCP and UDP is",
    "Newton's second law states that force equals",
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True)
    model.eval()

    prompts_ids, greedy = [], []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        prompts_ids.append(ids[0].tolist())
        gen = model.generate(ids.cuda(), max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        greedy.append(gen[0, ids.shape[1]:].tolist())
        print(f"  {p!r} → {tok.decode(greedy[-1][:16])!r}…")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": args.model, "prompts_token_ids": prompts_ids,
         "greedy_64": greedy}, indent=2))
    print(f"Reference → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
