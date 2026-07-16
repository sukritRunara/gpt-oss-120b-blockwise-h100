# Draft upstream issue — vLLM `moe_wna16_marlin_gemm` corrupt output on `mul_topk_weights=True` path

Status: **draft, not yet filed** (filing requires the user's go-ahead; no repo
content may be published without permission). Everything needed to file is in
this repo.

## Title

`[Bug] moe_wna16_marlin_gemm produces corrupt output (~1e33) when
mul_topk_weights=True at certain shapes (NVFP4 MoE, gpt-oss-20b dims)`

## Environment

- vLLM 0.25.1 (pip wheel), CUDA 12.8, H100 80GB (SM90), torch 2.9.0
- Model family: gpt-oss-20b (E=32 experts, top_k=4, hidden=2880,
  intermediate=2880), quantized to ModelOpt-style NVFP4 W4A16
  (`quant_algo: NVFP4`, group 16), served via the Marlin FP4 MoE path
  (`MarlinConfig` → `fused_marlin_moe`).

## Symptom

At gpt-oss dims, the second MoE gemm (`w2`, down-projection) returns rows that
are either exact garbage (~1e33 ≈ correct value × an uninitialized/OOB fp32
multiplier) or zeros, poisoning generation (repeated token 0 / NaN logits).
Small test shapes (e.g. E=8, N=K=512) are healthy, which masked the bug in
unit-style testing; the corruption is allocation-layout dependent.

## Isolation (standalone replay, no engine)

We captured a live failing `fused_marlin_moe` call's full argument set to disk
and replayed `ops.moe_wna16_marlin_gemm` directly against an exact fp32
reference computed in torch from the dequantized weights:

| variant | result |
|---|---|
| gemm1 (w13), as captured | all rows match (maxrel ≤ 3e-3) |
| gemm2 (w2), as captured (`top_k=4, mul_topk_weights=True`) | **128/128 rows bad, absmax 3.98e+32** |
| gemm2, identical args but `top_k=1, mul_topk_weights=False`, weights applied externally | all rows match (maxrel 2.4e-3) |
| thread-config sweeps, `use_atomic_add`, `use_fp32_reduce`, block sizes | no effect on the failure |
| same call pattern with different (random) weight values | still fails → not data-dependent |

The multiply path is the trigger: with everything else bit-identical, flipping
`mul_topk_weights` True→False (and applying `topk_weights` as an external
elementwise multiply on the `[M·topk, N]` output) restores bit-healthy output.
The magnitudes (~1e33 with fp32-multiplier structure) point at
`topk_weights` being read out of bounds / with a wrong stride for some
`(moe_block_size, thread-config, shape)` combinations inside the
`mul_topk_weights` branch of `marlin_moe_wna16` (csrc marlin moe kernel).

## Minimal repro in this repo

- `scripts/marlin_replay.py`, `scripts/marlin_replay2.py` — gemm1 replay (passes)
- `scripts/marlin_replay3.py` — gemm2 replay against exact reference (fails);
  flipping to `top_k=1, mul_topk_weights=False` + external multiply passes
- `results/pilot/marlin_call_capture.pt` — captured argument blob (1.6 GB, local only)
- `models/fixture-real2l-packed` — 2-layer real-weight checkpoint that
  reproduces end-to-end through vLLM serve (fails without the workaround,
  matches its QDQ reference with it)

## Workaround (what we ship)

`vllm-gptoss-nvfp4-plugin` P5: wrap `fused_marlin_moe`'s gemm2 call to run the
kernel with `mul_topk_weights=False` and apply the routing weights externally:

```python
out = ops.moe_wna16_marlin_gemm(..., mul_topk_weights=False, ...)
out.mul_(topk_weights.reshape(-1, 1).to(out.dtype))
```

Validated on the full 20B NVFP4 pack: deterministic serving, greedy-64
prefix agreement 0.869 vs the QDQ reference, coherent chat output.

## Possible addendum (P1.1)

Separately from the corruption: the Marlin MoE path shows intermittent
bitwise nondeterminism between identical greedy reruns (single near-tie
token flips; BF16 control bitwise-stable; `VLLM_MARLIN_USE_ATOMIC_ADD`
defaults off, ruled out). Quality-neutral but worth mentioning in the same
issue — suspected order-nondeterministic token grouping in the MoE
align/gemm path. Evidence: `results/quality/serving_check_{C,D}.json`
(`rerun_prefix_agreement_per_prompt`), KNOWN_ISSUES.md P1.1.

## Filing checklist (when approved)

1. Re-verify against vLLM main (`csrc/moe/marlin_moe_wna16/`) — the bug may
   already be fixed upstream; cite the commit if so.
2. Trim the capture blob to a <10 MB self-contained repro script (weights can
   be random — failure is not data-dependent).
3. Attach the elimination table above and the workaround.
