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

<!-- Pending decisions to record as work proceeds (handoff §9):
  - Which vLLM version and NVFP4 contract were pinned.
  - Whether mixed-precision BF16 fallback is retained, and which tensors are excluded.
-->
