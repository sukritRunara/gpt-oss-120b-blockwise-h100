# Dequantization provenance (arm B)

## What arm B is — and is not

```
openai/gpt-oss-20b official checkpoint          (arm A)
    │  expert weights stored as MXFP4:
    │  *_blocks  uint8 [E, out, in/32, 16]  (2 FP4 codes/byte, 32-elem blocks)
    │  *_scales  uint8 [E, out, in/32]      (E8M0 shared exponents)
    ▼
exact MXFP4 decode                              transformers 5.14.0
    │  Mxfp4Config(dequantize=True)             (the pinned, supported path;
    │  value = FP4_LUT[nibble] · 2^(scale-127)   integrations/mxfp4.py)
    ▼
gpt-oss-20b-mxfp4-dequant-bf16                  (arm B, 20.915B params, BF16)
```

Arm B is **not** the original pre-MXFP4 master checkpoint (which is not
publicly available). Every MXFP4-representable value is exactly representable
in BF16 (FP4 magnitudes × power-of-two scales carry ≤ 3 mantissa bits), so
the decode itself is lossless; what is "lost" is whatever the original
master had before OpenAI quantized it. All downstream results are therefore a
**transquantization** experiment:

```
official MXFP4  →  dequantized BF16 source  →  {RTN, blockwise-GPTQ} NVFP4
```

Comparisons against arm B measure NVFP4-vs-decoded-BF16, NOT
NVFP4-vs-original-BF16.

## Pinning

- Source revision + per-file SHA-256: `models/gpt-oss-20b-official-mxfp4/PROVENANCE.json`
  (written by `scripts/download_official_model.py`).
- Arm B provenance (source revision, decode mechanism, transformers version,
  output hashes): `models/gpt-oss-20b-mxfp4-dequant-bf16/PROVENANCE.json`
  (written by `scripts/dequantize_gpt_oss_20b.py`).

## Only the experts were quantized

In the official checkpoint, ONLY the MoE expert tensors
(`experts.gate_up_proj`, `experts.down_proj`) are MXFP4. Attention
projections, router, embeddings, norms, biases, and sinks are plain BF16 and
pass through the decode byte-identical.

## Save-path pitfall (recorded for reproducers)

transformers 5.14.0's `save_pretrained` runs `revert_weight_conversion()`,
which maps dequantized expert weights back toward their checkpoint-format
names and silently DROPS them when the reverse conversion cannot run — the
result is a ~4.9 GB "checkpoint" with no expert weights.
`scripts/dequantize_gpt_oss_20b.py` therefore writes the state dict manually
(safetensors shards + index) and fails closed if the state dict holds fewer
parameters than the model.

## Validation

`scripts/validate_dequantized_source.py` (results:
`results/dequant_validation.json`, log: `logs/dequantization/`):

1. **structure** — reloads as ordinary BF16: no `quantization_config`, no
   packed `*_blocks/*_scales`, BF16/FP32 dtypes only, all values finite.
2. **decode** — EVERY expert tensor equals the exact MXFP4 decode of arm A's
   blocks/scales (bit-exact in BF16); every non-expert tensor is
   byte-identical to arm A; tensor-set coverage is exact in both directions.
3. **determinism** — repeated forward passes are bitwise identical.
4. **logits A-vs-B** — cosine/KL/top-1/top-10/max-diff/greedy-prefix metrics
   when arm A loads natively in transformers; otherwise recorded as DEFERRED
   and performed at the vLLM serving stage (arm A always runs natively there).
