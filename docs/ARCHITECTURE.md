# Architecture

## Repository layout (working tree = `/workspace`)

```
/workspace
├── blockwise-gptq-main/            # the quantization library + stage scripts
│   ├── opteam-blockwise-gptq/
│   │   ├── apply.py                # orchestration: grouped Hessian collection
│   │   │                           #   (P0.4) + quantize-from-cache, records
│   │   ├── gptq.py                 # GPTQ + canonical accumulate_hessian (P0.3)
│   │   ├── quantizer.py            # NVFP4 (E2M1 + fp8 scales + D-010 global
│   │   │                           #   scale) with exact-artifact capture (P0.6)
│   │   ├── quant_artifacts.py      # artifact store + manifest (P0.5/P0.6)
│   │   ├── hessian_cache.py        # resumable manifest-verified cache (P0.4)
│   │   ├── expert_dispatch.py      # MoE handlers (GPT-OSS batched experts)
│   │   ├── gpt_oss_expert_gptq.py  # expert forward patch (P0.2) + shims
│   │   ├── calibration.py          # C4/WikiText-2 calibration loading
│   │   └── model_utils.py          # architecture discovery
│   └── tests/
│       ├── stage1..stage6          # unit → quality pipeline stages
│       ├── stage7_save_modelopt.py # exact-artifact vLLM packing (P0.5-P0.8)
│       └── internalTests/          # pytest batteries (routing, hessians,
│                                   #   grouped collection, artifacts, stage7)
├── scripts/                        # project-level tooling (download, dequant,
│                                   #   validation, RTN control, pilot, serving
│                                   #   benchmark, fixture proof)
├── configs/serve-*.env             # per-arm vLLM launch configs
├── vllm-gptoss-nvfp4-plugin/       # vLLM 0.25.1 patches (D-011)
├── docs/                           # this file + contract/provenance/design
├── envs/                           # lockfiles + system manifest
├── models/                         # (gitignored) checkpoints & artifacts
├── cache/                          # (gitignored) HF cache, hessian caches
├── logs/                           # (gitignored) timestamped run logs
└── results/                        # benchmark/eval outputs
```

## Quantization data flow (arms C and D)

```
arm B (dequant BF16)
  │  full-model forwards over cached, hashed calibration tokens
  ▼
grouped Hessian collection (P0.4)          hessian_cache.py
  │  ≤ group_size layers per pass; all MoE layers pinned to one expert
  │  forward implementation (D-008); per-layer shards, SHA-256 manifest,
  │  resumable; fail-closed on sample errors / zero-sample sublayers /
  │  patched-forward bypass
  ▼
blockwise GPTQ per layer                   apply.py + gptq.py
  │  lazy per-layer GPTQ instances from cache; D-010 global scales fixed
  │  from original weights (q/k/v shared); expert slices via fp32 shims
  ▼
exact artifact capture (P0.6)              quantizer.py begin/end_capture
  │  E2M1 codes + normalized fp8 scales + global scale recorded per tensor,
  │  verified bit-exact against the QDQ weight IMMEDIATELY
  ▼
QDQ checkpoint + manifest (P0.5)           save_pretrained + write_quant_manifest
  │  one disposition record per tensor/expert-slice; records ∪ excluded
  │  == all named parameters
  ▼
stage 7 packing (P0.7/P0.8)                stage7_save_modelopt.py
  │  re-verifies every artifact vs the on-disk QDQ tensor; linears →
  │  ModelOpt W4A16 layout; experts → vLLM FusedMoE layout (per-expert
  │  transposed codes/scales); fail-closed hybrid policy
  ▼
vLLM serving                               .venv-serve + gptoss-nvfp4 plugin
     Marlin FP4 weight-only kernels (linear + MoE) on H100
```

Arm C (RTN) runs the same capture/manifest/packing path via
`scripts/build_rtn_control.py`, optionally mask-matched to a GPTQ manifest —
the only difference between C and D is the quantization algorithm.

## Two environments (mandatory)

- `.venv-quant` — torch 2.13.0+cu130, transformers 5.14.0: everything up to
  and including packing + QDQ evaluation.
- `.venv-serve` — vllm 0.25.1 (torch 2.11.0+cu130) + vllm-gptoss-nvfp4
  plugin: serving, fixture proofs, serving benchmarks.
The torch versions genuinely conflict; never merge the environments.
