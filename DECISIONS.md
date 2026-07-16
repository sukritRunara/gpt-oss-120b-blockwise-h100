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

<!-- Pending decisions to record as work proceeds (handoff §9):
  - Which Hessian-collection strategy was chosen (memory-bounded layer groups).
  - Which vLLM version and NVFP4 contract were pinned.
  - Whether mixed-precision BF16 fallback is retained, and which tensors are excluded.
-->
