"""Smoke-test: load a packed quantized checkpoint in vLLM and generate.

Usage (run inside .venv-serve):
    python test_vLLM_deploy_quantized_model.py --model <path-to-packed-checkpoint>

The model path is required (P0.1 fix — no hard-coded developer paths).
"""

import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Path to the packed quantized checkpoint directory")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    # Several prompts with non-zero temperature to avoid repetition loops
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=80)

    prompts = [
        "The capital of England is",
        "2 * 2 =",
        "def fibonacci(n):\n    ",
        "The largest planet in the solar system is",
    ]

    outputs = llm.generate(prompts, sampling_params)
    for out in outputs:
        print(f"PROMPT: {out.prompt!r}")
        print(f"OUTPUT: {out.outputs[0].text!r}")
        print()


if __name__ == "__main__":
    main()
