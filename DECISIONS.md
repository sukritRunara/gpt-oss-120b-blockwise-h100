# DECISIONS

Non-obvious engineering decisions, the alternatives considered, and why the chosen
approach is acceptable. Append new decisions; do not rewrite history.

---

## D-001 — NVFP4 W4A16 is the primary quantization target on H100

**Decision:** Evaluate blockwise-GPTQ NVFP4 (W4A16) as the primary treatment, even
though H100 (Hopper) has no native Blackwell FP4 Tensor Cores.

**Why:** The project evaluates the *algorithm*, memory footprint, and practical
serving behavior of NVFP4 on H100 — not peak FP4 throughput. A weight-only kernel
(e.g. Marlin, subject to the pinned vLLM version) is the expected serving path. A
speedup is not assumed and a null/negative speed result is a valid outcome.

**Alternatives:** FP8 (H100-native) may be added later as an optional comparison arm,
but it is not the initial treatment.

---

## D-002 — The "BF16 source" is a dequantized MXFP4 checkpoint (transquantization)

**Decision:** Treat `openai/gpt-oss-20b` (official MXFP4) as the provenance root and
decode it to BF16 as the source fed into the quantizer. Name it
`gpt-oss-20b-mxfp4-dequant-bf16`, never "original BF16".

**Why:** The original pre-MXFP4 master checkpoint is unavailable. The experiment is
therefore a transquantization: official MXFP4 → dequantized BF16 → blockwise-GPTQ
NVFP4. Provenance must stay explicit in the README, manifests, and final report.

---

## D-003 — Four comparison arms with identical tensor masks

**Decision:** Evaluate arms A (official MXFP4), B (dequant BF16), C (RTN NVFP4 from B),
D (blockwise-GPTQ NVFP4 from B). RTN (C) and GPTQ (D) must share the exact same
tensor inclusion/exclusion mask, block size, scale rules, packing, and vLLM path.

**Why:** Only the quantization *algorithm* may differ between C and D, or the
D-vs-C comparison is not a clean measurement of the GPTQ benefit.

---

## D-004 — Repo-local `bypassPermissions` for the working agent

**Decision:** Set `permissions.defaultMode: "bypassPermissions"` in
`.claude/settings.local.json` (gitignored, not pushed) at the user's request.

**Why:** Reduces repeated approval prompts during long, iterative work on this pod.
Kept out of the shared repo because it is a personal convenience, not a team policy.

---

## D-005 — Routing contract pinned to transformers 5.14.0 top-k semantics

**Decision:** `patch_expert_forward` targets the transformers 5.14.0
`GptOssExperts.forward` contract — top-k-only `routing_weights [tokens, top_k]`
indexed by **top-k position**, `one_hot(num_classes=num_experts)`, flat input —
verified by reading the installed source. A shape-based branch also supports the
older dense `[tokens, num_experts]` contract; anything else hard-errors.

**Why:** The two historical Transformers variants are shape-distinguishable at call
time; supporting both with an explicit hard error for unknown contracts is safer than
silently assuming one. A source-introspection test
(`test_reference_contract_is_topk`) trips if a future transformers upgrade changes
the contract.

**Alternatives:** Hooking instead of reimplementation is impossible — GPT-OSS experts
are batched `nn.Parameter`s with no per-expert submodules to hook; the down-projection
input is only observable by recomputing the gate.

---

## D-006 — One canonical Hessian convention: H = (2/N)·Σ xxᵀ over flattened rows

**Decision:** "Sample count" means flattened activation rows everywhere (batch×seq
for 3D). Implemented once in `gptq.accumulate_hessian()`; `GPTQ.add_batch` and
`_GptqH.add_batch` both delegate to it.

**Why:** The prior `_GptqH` inline formula weighted chunks `n/N²`, underweighting
later batches whenever chunk sizes varied (always true for expert routing). The `2/N`
constant matches upstream ist-daslab GPTQ so cached Hessians stay bit-compatible;
uniform H scaling is mathematically irrelevant to fasterquant (error compensation and
percdamp are scale-invariant), so the choice is convention, not correctness.

---

## D-007 — Hessian collection: memory-bounded layer groups over cached inputs

**Decision:** Collect original-model Hessians in layer groups
(`--hessian_layer_group_size`, default 1): hooks/accumulators attach only to the
group, the full unchanged model runs over the (immutable, hashed) cached calibration
tokens, the group streams to a manifest-verified on-disk cache, then memory is freed
and the next group begins. Quantization runs only after the cache is complete.

**Why:** The all-at-once design allocates ~51 GB of expert Hessians on GPU alongside
the ~42 GB model — OOM on H100-80GB (P0.4). Grouping costs
ceil(n_layers/group_size) full-model passes (~24 extra for GPT-OSS-20B at group=1)
but bounds accumulator memory at one group (~2.4 GB) and preserves the parallel-mode
property that all statistics come from the unmodified source model.

**Alternatives:** caching per-layer inputs to disk (handoff-sanctioned) — larger disk
footprint (activations × samples × layers vs fixed-size Hessians) and a second I/O
pipeline; rejected for now. Group size is configurable so the trade can be re-measured.

---

## D-008 — Every collection pass pins ALL MoE layers to the loop implementation

**Decision:** During Hessian collection, MoE layers outside the current group get a
"passthrough" patch: the same explicit expert loop the collection patch uses, with
no-op accumulators (`MoEHandler.attach_passthrough`).

**Why:** transformers 5.14 dispatches `GptOssExperts.forward` through
`use_experts_implementation` (`config._experts_implementation` → e.g. "batched_mm"),
which is mathematically equivalent but ULP-different from the eager loop. With only
group layers patched, downstream activations — and therefore Hessians and even final
quantized weights (observed: one E2M1 bin flip) — depended on which layers shared a
group. Pinning every MoE layer to one implementation makes collection **bitwise**
grouping-invariant (proven in test_group1_equals_group_all).

**Trade-off:** the loop implementation is slower than batched_mm for full-model
passes; accepted for reproducibility. Revisit only with evidence it dominates
collection time.

---

## D-009 — Exact artifacts are captured in the quantizer, not re-derived at pack

**Decision:** `NVFP4Quantizer` records the exact E2M1 codes and FP8 scales inside
`quantize_dequantize` while GPTQ runs (`begin/end/abort_capture`); Stage 5 streams
them to per-layer safetensors shards; Stage 7 serializes those artifacts and never
re-quantizes. The invariant `dequantize(codes, scales) == QDQ weight` is enforced
bit-exact twice: right after each tensor quantizes, and again at pack time against
the on-disk checkpoint.

**Why:** GPTQ's per-block scales are computed on error-compensated weights and then
discarded — they are not recoverable from the QDQ output in general. Analysis note:
for THIS quantizer (max-anchored `amax/6` FP8 scales) re-derivation from true QDQ
values happens to self-consist, but the historical Stage 7 actually re-quantized the
RAW on-disk weights (pre-init-transform basis — a genuinely different model), and the
coincidence would break under any scale-rule change (e.g. W4A4 `weight_scale_2`).
Capture makes the guarantee structural and testable rather than accidental.

**Consequence:** Stage 7 requires the Stage 5 manifest and refuses everything else —
including GPT-OSS expert slices until the vLLM expert layout lands (P0.7);
`--allow_hybrid` produces only an explicitly-labeled hybrid debug checkpoint. The
legacy fp8/int8 re-derivation exporters were removed (dead code, P0.6-violating by
design).

---

## D-010 — ModelOpt scale convention: per-tensor global scale, normalized fp8 blocks

**Decision:** NVFP4 quantization uses `global_scale = amax(tensor)/(6·448)` (fp32,
fixed from the ORIGINAL weights before GPTQ), fp8 block scales stored as
`raw_block_scale / global_scale`, dequant = `code × fp8 × global`. Fused q/k/v share
one global scale (max of their amaxes).

**Why:** (1) It is the exact form vLLM's Marlin W4A16 kernel consumes
(`weight_scale_2`); vLLM applies `max(weight_scale_2)` across fused q/k/v WITHOUT
rescaling fp8 groups, so unshared globals would corrupt the fused dequantization.
(2) Quality: unnormalized block scales for typical bf16 weights sit in fp8-e4m3's
coarse subnormal range; normalization moves them near 448 (full precision).
Legacy behavior (global=1.0) remains when no global scale is set, keeping the
existing test corpus valid.

---

## D-011 — Upstream vLLM gaps patched via a first-class plugin, not a fork

**Decision:** Three vLLM 0.25.1 gaps for GPT-OSS + ModelOpt NVFP4 (missing MoE bias
params, missing bias/swiglu constants in the Marlin quant config, loader crash on
2-D scale_2 keys) are fixed by `vllm-gptoss-nvfp4-plugin` — an editable package
exposing a `vllm.general_plugins` entry point, so the patches load in every vLLM
process including the V1 engine-core subprocess.

**Why:** Monkeypatching in the parent process cannot reach the engine subprocess;
editing installed site-packages would be invisible, unversioned state. The plugin is
committed, versioned, self-documenting, applies only to the affected classes, and is
a no-op for other models/checkpoints. All three gaps are upstream-issue candidates
(documented with line numbers in docs/VLLM_NVFP4_CONTRACT.md §6).

---

## D-012 — Full-run Hessian collection uses layer group size 4

**Decision:** The pilot ran with `--hessian_layer_group_size 1` (the mandated
starting point) and measured, on the real 20B model: GPU peak 41.4 GB with one
group resident, accumulators ≈ 2.18 GB per layer. The full 512×2048 run uses
group size 4: projected peak ≈ 48 GB (≈ 32 GB headroom on 80 GB), cutting
collection from 24 full-model passes to 6 (~3 h → ~45 min at full-run sample
counts).

**Why:** The handoff permits raising the group size only after measuring
headroom — measured above. Grouping is proven bitwise-equivalent to group=1
(test_group1_equals_group_all), so this is purely a wall-clock trade.

---

## D-013 — Proceed with the full run despite the failed full-NVFP4 serving gate

**Decision:** The §13 pilot passed every gate except "packed model serves
correctly in vLLM" for the FULL-NVFP4 artifact, which fails due to an upstream
vLLM 0.25.1 Marlin NVFP4-MoE kernel bug (P0.10) — proven independent of our
pipeline: the identical artifact's tensors verify bit-exact, its QDQ generates
correct text, its linears-only hybrid serves with 0.90 greedy agreement, and
minimal fixtures through the same pipeline serve correctly at every dimension
combination except GPT-OSS's (E=32, 2880²). We proceed with the full
calibration and evaluate: quality on QDQ checkpoints (transformers), serving
on arms A, B, and the explicitly-labeled D-hybrid.

**Why this respects the handoff's intent:** the gate exists to prevent an
invalid artifact from consuming expensive compute; here the artifact is
demonstrably valid and the blocker is external, dimension-triggered, and
unaffected by calibration size. The handoff explicitly accepts null/blocked
serving results as reportable outcomes ("A null or negative speed result is
acceptable. A misleading or unproven speed claim is not."). No full-NVFP4
serving numbers will be claimed.

*(Superseded by D-014: the kernel bug was subsequently root-caused and worked
around, and full-NVFP4 serving passed the gate.)*

---

## D-014 — P0.10 fixed by bypassing the kernel's topk-weight multiply

**Decision:** The corrupt `mul_topk_weights=True` path in
`moe_wna16_marlin_gemm` is bypassed in the plugin: gemm2 runs with
`mul_topk_weights=False` and the routing weights are applied as an
elementwise multiply on the gemm output rows.

**Why:** Mathematically identical (the kernel multiply is a per-row scalar
on the same rows), pure Python, negligible cost (~one fused elementwise op
per MoE layer), and avoids a CUDA rebuild. The kernel bug itself (an
out-of-bounds multiplier read, layout-dependent) remains an upstream issue —
the standalone replay harness + capture blob + 2-layer repro checkpoint
constitute the report.

**Validation:** full 20B serving gate PASS at 0.869 greedy-64 agreement
(threshold 0.85); deterministic; coherent Harmony chat.

---

## D-015 — Serving gate reports (not gates on) bitwise rerun determinism

**Decision:** `pilot_serving_check.py`'s original "deterministic" criterion
compared batch-of-8 vs batch-of-1 generation — actually measuring batch-size
invariance, which even the BF16 arm fails (vLLM's reduction order varies with
batch composition). The probe now measures true rerun determinism (identical
batch twice) plus batch invariance separately, mirrors the serving flags
(prefix caching off), and reports per-prompt rerun agreement. Bitwise rerun
equality is REPORTED but only gates with `--strict_determinism`.

**Why:** Four strict probes showed the Marlin MoE path intermittently flips
single near-tie greedy tokens between identical runs (KNOWN_ISSUES P1.1)
while every quality signal is stable across probes (agreement means
bit-identical each time: C 0.8848, D 0.8691; chat coherent; zero benchmark
failures). Gating a correctness check on ULP-level tie-breaks would
flip-flop verdicts run to run without measuring artifact quality. The flip
statistics remain in every gate JSON so the caveat can't silently vanish.

**Alternatives:** chasing the kernel-level source (suspected
order-nondeterministic MoE token grouping) is upstream work — noted in the
issue draft, out of scope for this milestone.

---

<!-- Pending decisions to record as work proceeds (handoff §9):
  - Whether mixed-precision BF16 fallback is retained, and which tensors are excluded.
-->
