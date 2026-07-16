#!/usr/bin/env python
"""Small reproducible task-level suite (handoff §16): general knowledge,
math, code, instruction following — Harmony chat formatting via the model's
chat template, greedy decoding, substring/expression scoring. Every prompt,
raw output, and per-item verdict is saved.

This is a MODEST diagnostic suite (40 items), not a leaderboard benchmark;
identical items and decoding across arms make the deltas meaningful.

Run inside .venv-quant:
    python scripts/task_eval.py --model <path> --name D --out results/quality/task_D.json
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# (category, question, accept) — accept: list of case-insensitive substrings,
# ANY of which marks the answer correct.
ITEMS = [
    ("knowledge", "What is the capital of Japan? Answer in one word.", ["tokyo"]),
    ("knowledge", "Which planet is known as the Red Planet?", ["mars"]),
    ("knowledge", "Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("knowledge", "What is the largest ocean on Earth?", ["pacific"]),
    ("knowledge", "In what year did World War II end?", ["1945"]),
    ("knowledge", "What gas do plants absorb from the atmosphere?", ["carbon dioxide", "co2"]),
    ("knowledge", "What is the chemical symbol for gold?", ["au"]),
    ("knowledge", "Which country has the largest population in the world?", ["india", "china"]),
    ("knowledge", "What organ pumps blood through the human body?", ["heart"]),
    ("knowledge", "How many continents are there on Earth?", ["seven", "7"]),
    ("math", "What is 17 multiplied by 23?", ["391"]),
    ("math", "What is the square root of 144?", ["12"]),
    ("math", "If x + 7 = 19, what is x?", ["12"]),
    ("math", "What is 15% of 240?", ["36"]),
    ("math", "What is 2 to the power of 10?", ["1024"]),
    ("math", "A rectangle has sides 8 and 5. What is its area?", ["40"]),
    ("math", "What is 1000 divided by 8?", ["125"]),
    ("math", "What is the sum of the first 10 positive integers?", ["55"]),
    ("math", "Convert 3/4 to a decimal.", ["0.75", ".75"]),
    ("math", "What is 99 plus 47?", ["146"]),
    ("code", "In Python, what built-in function returns the number of items in a list?", ["len"]),
    ("code", "What does SQL SELECT DISTINCT do? Answer briefly.", ["duplicate", "unique"]),
    ("code", "In Python, what keyword defines a function?", ["def"]),
    ("code", "What is the output of print(2 ** 3) in Python?", ["8"]),
    ("code", "Which HTTP method is idempotent: POST or PUT?", ["put"]),
    ("code", "In Python, what exception is raised when dividing by zero?", ["zerodivisionerror", "zero division"]),
    ("code", "What data structure uses LIFO ordering?", ["stack"]),
    ("code", "What does the 'g' flag do in a JavaScript regex?", ["global", "all matches", "every match"]),
    ("code", "In git, which command creates a new branch and switches to it in one step?", ["checkout -b", "switch -c"]),
    ("code", "What is the time complexity of looking up a key in a hash table on average?", ["o(1)", "constant"]),
    ("instruct", "Reply with exactly the word banana and nothing else.", ["banana"]),
    ("instruct", "Count from 1 to 5, separated by commas.", ["1, 2, 3, 4, 5", "1,2,3,4,5"]),
    ("instruct", "Write the word 'hello' in uppercase.", ["HELLO"]),
    ("instruct", "What is the third word of this sentence: 'The quick brown fox jumps'?", ["brown"]),
    ("instruct", "Spell the word cat backwards.", ["tac"]),
    ("instruct", "Give a one-word synonym for happy.", ["joyful", "glad", "cheerful", "content", "elated", "joyous", "merry", "pleased"]),
    ("instruct", "Repeat this exactly: 42 is the answer.", ["42 is the answer"]),
    ("instruct", "Name any three primary colors, separated by commas.", ["red", "blue", "yellow"]),
    ("instruct", "How many letters are in the word 'quantum'?", ["7", "seven"]),
    ("instruct", "Answer yes or no: Is the Earth flat?", ["no"]),
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=384,
                    help="room for Harmony reasoning before the final answer")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True)
    model.eval()

    records, correct_by_cat, count_by_cat = [], {}, {}
    for cat, q, accept in ITEMS:
        msgs = [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").cuda()
        gen = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=False)
        # Harmony: score the FINAL channel if present, else the whole output.
        final = raw
        m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|(?:return|end)\|>|$)",
                      raw, re.DOTALL)
        if m:
            final = m.group(1)
        ok = any(a.lower() in final.lower() for a in accept)
        records.append({"category": cat, "question": q, "accept": accept,
                        "raw_output": raw[:2000], "final_answer": final[:500],
                        "correct": ok})
        correct_by_cat[cat] = correct_by_cat.get(cat, 0) + int(ok)
        count_by_cat[cat] = count_by_cat.get(cat, 0) + 1
        print(f"  [{'ok ' if ok else 'MISS'}] ({cat}) {q[:50]}")

    summary = {cat: {"correct": correct_by_cat[cat], "total": count_by_cat[cat],
                     "accuracy": correct_by_cat[cat] / count_by_cat[cat]}
               for cat in count_by_cat}
    total_ok = sum(correct_by_cat.values())
    out = {"model": args.model, "name": args.name,
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "overall_accuracy": total_ok / len(ITEMS),
           "n_items": len(ITEMS), "by_category": summary, "items": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"{args.name}: overall {total_ok}/{len(ITEMS)} "
          f"({100 * total_ok / len(ITEMS):.0f}%) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
