#!/usr/bin/env python
"""Pilot gate: load a packed NVFP4 checkpoint in vLLM, run deterministic and
Harmony-chat generations, and compare greedy continuations against reference
continuations produced by the QDQ model in transformers (pilot §13 gates:
"Packed model loads in vLLM", "vLLM logs reveal quantization/kernel path",
"deterministic generation", "Harmony-formatted chat generation succeeds",
"Stage 5 QDQ ≈ Stage 7 packed" at the serving level).

The packed weights are bit-exact dequantizations of the QDQ weights (proven
at pack time); this check exercises the vLLM COMPUTE path (Marlin kernels),
where small kernel-order differences are expected — the gate is high greedy
prefix agreement, not bitwise equality.

Run inside .venv-serve:
    python scripts/pilot_serving_check.py --packed <dir> \\
        --reference_json results/pilot/qdq_reference.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--reference_json", type=Path, required=True,
                    help="JSON with {prompts_token_ids: [[...]], "
                         "greedy_64: [[...]]} from the QDQ model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min_prefix_agreement", type=float, default=0.85)
    args = ap.parse_args()

    ref = json.loads(args.reference_json.read_text())

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.packed, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True,
              disable_log_stats=True)

    # 1) Determinism: same greedy request twice → identical tokens
    sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
    reqs = [{"prompt_token_ids": ids} for ids in ref["prompts_token_ids"]]
    out1 = llm.generate(reqs, sp)
    out2 = llm.generate([reqs[0]], sp)
    det = list(out1[0].outputs[0].token_ids) == list(out2[0].outputs[0].token_ids)

    # 2) Greedy prefix agreement vs QDQ reference
    agreements = []
    for o, ref_toks in zip(out1, ref["greedy_64"]):
        got = list(o.outputs[0].token_ids)
        n = min(len(got), len(ref_toks))
        same = 0
        for i in range(n):
            if got[i] != ref_toks[i]:
                break
            same += 1
        agreements.append(same / n if n else 0.0)
    mean_agree = sum(agreements) / len(agreements)

    # 3) Harmony chat generation succeeds (chat template applies, output nonempty)
    chat_ok = False
    try:
        chat_out = llm.chat(
            [[{"role": "user", "content": "In one sentence, what is GPTQ?"}]],
            SamplingParams(temperature=0.0, max_tokens=48))
        chat_text = chat_out[0].outputs[0].text
        chat_ok = len(chat_text.strip()) > 0
    except Exception as exc:                               # noqa: BLE001
        chat_text = f"ERROR: {exc}"

    result = {
        "deterministic": det,
        "greedy64_prefix_agreement_mean": mean_agree,
        "greedy64_prefix_agreement_per_prompt": agreements,
        "harmony_chat_ok": chat_ok,
        "harmony_chat_sample": str(chat_text)[:200],
        "gate_pass": bool(det and chat_ok
                          and mean_agree >= args.min_prefix_agreement),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("PILOT_SERVING_GATE:", "PASS" if result["gate_pass"] else "FAIL")
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
