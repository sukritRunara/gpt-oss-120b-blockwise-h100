from vllm import LLM, SamplingParams
MODEL = "/home/runara_dgx_spark_1/Itamar/projects/Block-wise-GPTQ-GPT-OSS-20B-NVFP4/blockwise-gptq/models/DeepSeek-V2-Lite-NVFP4-modelopt-v5"

llm = LLM(
    model=MODEL,
    trust_remote_code=True,
    dtype="auto",
    max_model_len=2048,
    gpu_memory_utilization=0.88,
    enforce_eager=True,
)

# Try several prompts with non-zero temperature to avoid repetition loops
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


'''


    

    MODEL = "/home/runara_dgx_spark_1/Itamar/projects/Block-wise-GPTQ-GPT-OSS-20B-NVFP4/blockwise-gptq/models/DeepSeek-V2-Lite-NVFP4"  # Stage 5 BF16 output

    llm = LLM(model=MODEL, trust_remote_code=True, dtype="bfloat16",
            max_model_len=2048, gpu_memory_utilization=0.88, enforce_eager=True)

    outputs = llm.generate(["2*2 =", "The capital of Israel is"],
                        SamplingParams(temperature=0.7, max_tokens=30))
    for o in outputs:
        print(repr(o.outputs[0].text))
'''