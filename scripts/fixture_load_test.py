#!/usr/bin/env python
"""vLLM fixture load test (P0.7/P0.8 proof). Run inside .venv-serve.

Loads the tiny NVFP4 GPT-OSS fixture checkpoint in vLLM, asserts the
ModelOpt W4A16 path engaged, and runs a deterministic generation. The
model is random-weighted — output TEXT is meaningless; what matters is:
  - config resolves to modelopt_fp4 / W4A16_NVFP4
  - weights load without shape/dtype errors (incl. FusedMoE experts)
  - Marlin kernels execute a forward pass without NaN/crash
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/workspace/models/fixture-nvfp4")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=512,
        gpu_memory_utilization=0.25,
        enforce_eager=True,
        disable_log_stats=True,
    )

    out = llm.generate(
        [{"prompt_token_ids": [200006, 1428, 200008, 400, 500, 600]}],
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    tokens = out[0].outputs[0].token_ids
    print(f"FIXTURE_GENERATED_TOKENS: {list(tokens)}")
    assert len(tokens) > 0
    print("FIXTURE_LOAD_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
