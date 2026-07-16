#!/usr/bin/env python
"""Logit-level paired evaluation across arms (handoff §16).

Loads each model sequentially (GPU), computes final-position logits and
greedy continuations on a FIXED held-out prompt set (disjoint from the C4
calibration data), and reports per-arm-vs-reference metrics:
mean/worst logit cosine, KL divergence, next-token top-1 agreement, top-k
overlap, max abs logit diff, greedy 64-token prefix agreement.

Run inside .venv-quant:
    python scripts/logit_eval.py \\
        --reference B=/workspace/models/gpt-oss-20b-mxfp4-dequant-bf16 \\
        --candidates C=<rtn-qdq> D=<gptq-qdq> \\
        --out results/quality/logit_eval.json
"""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Held-out prompts — general knowledge, math, code, multilingual, reasoning.
# Deliberately disjoint from C4 calibration samples.
PROMPTS = [
    "The capital of Australia is",
    "Photosynthesis converts sunlight, water, and carbon dioxide into",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n",
    "The derivative of x^3 + 2x is",
    "In object-oriented programming, inheritance means",
    "The Treaty of Westphalia in 1648 established",
    "La distancia entre la Tierra y la Luna es de aproximadamente",
    "A SQL LEFT JOIN returns",
    "The boiling point of water at sea level is",
    "To reverse a linked list in place, you need three pointers:",
    "The Pythagorean theorem states that",
    "DNA replication occurs during the",
    "Compound interest differs from simple interest because",
    "The time complexity of merge sort is",
    "Die Hauptstadt von Deutschland ist",
    "If a train travels 60 km in 45 minutes, its average speed is",
    "The three branches of the US government are",
    "In thermodynamics, entropy measures",
    "import numpy as np\narr = np.arange(12).reshape(3, 4)\nprint(arr.sum(axis=",
    "The Great Barrier Reef is located off the coast of",
    "Ohm's law relates voltage, current, and",
    "A binary tree with n leaves has exactly",
    "The French Revolution began in the year",
    "Water's chemical polarity explains why it",
]


@torch.no_grad()
def profile_model(path: str, n_greedy: int = 8):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()

    logits, greedy = [], []
    for i, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        logits.append(model(ids).logits[0, -1].float().cpu())
        if i < n_greedy:
            g = model.generate(ids, max_new_tokens=64, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            greedy.append(g[0, ids.shape[1]:].cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits, greedy


def compare(ref, cand):
    ref_logits, ref_greedy = ref
    logits, greedy = cand
    cos, kls, top1, top10, maxd = [], [], [], [], []
    for la, lb in zip(ref_logits, logits):
        cos.append(torch.nn.functional.cosine_similarity(
            la.unsqueeze(0), lb.unsqueeze(0)).item())
        kls.append(torch.nn.functional.kl_div(
            torch.log_softmax(lb, -1), torch.log_softmax(la, -1),
            log_target=True, reduction="sum").item())
        top1.append(int(la.argmax() == lb.argmax()))
        sa = set(la.topk(10).indices.tolist())
        sb = set(lb.topk(10).indices.tolist())
        top10.append(len(sa & sb) / 10)
        maxd.append((la - lb).abs().max().item())
    prefix = []
    for ga, gb in zip(ref_greedy, greedy):
        n = min(len(ga), len(gb))
        same = 0
        for i in range(n):
            if ga[i] != gb[i]:
                break
            same += 1
        prefix.append(same / n if n else 0.0)
    return {
        "cosine_mean": sum(cos) / len(cos), "cosine_min": min(cos),
        "kl_mean": sum(kls) / len(kls), "kl_max": max(kls),
        "top1_agreement": sum(top1) / len(top1),
        "top10_overlap_mean": sum(top10) / len(top10),
        "max_abs_logit_diff": max(maxd),
        "greedy64_prefix_agreement_mean": sum(prefix) / len(prefix),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="NAME=PATH")
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="NAME=PATH …")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref_name, ref_path = args.reference.split("=", 1)
    print(f"[logit_eval] profiling reference {ref_name}: {ref_path}")
    ref = profile_model(ref_path)

    out = {"reference": {ref_name: ref_path},
           "n_prompts": len(PROMPTS),
           "created_utc": datetime.now(timezone.utc).strftime(
               "%Y-%m-%dT%H:%M:%SZ"),
           "comparisons": {}}
    for spec in args.candidates:
        name, path = spec.split("=", 1)
        print(f"[logit_eval] profiling {name}: {path}")
        cand = profile_model(path)
        out["comparisons"][name] = {"path": path, **compare(ref, cand)}
        m = out["comparisons"][name]
        print(f"  {name} vs {ref_name}: cos_min={m['cosine_min']:.5f} "
              f"kl_mean={m['kl_mean']:.5f} top1={m['top1_agreement']:.3f} "
              f"prefix={m['greedy64_prefix_agreement_mean']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
