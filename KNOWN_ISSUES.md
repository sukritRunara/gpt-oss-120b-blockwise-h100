# KNOWN ISSUES

Open issues carried over from the agent handoff (§10) plus anything found during work.
Each item: severity (P0/P1/P2), repro, evidence, hypothesis, next action, and whether
it invalidates current artifacts. Close items by moving them to a "Resolved" section
with the proving evidence.

Status key: 🔴 open · 🟡 in progress · 🟢 resolved

---

## Open (from handoff §10 — not yet verified against the actual source)

These are the documented P0 risks from the prior GPT-OSS work. They must be confirmed
against the real `blockwise-gptq-main/` source before the full run — do not assume they
are present or absent.

- 🔴 **P0.1 Hard-coded paths** — stage scripts may contain DGX-specific absolute paths.
  *Next:* grep the source for developer home paths; make paths repo-relative; add a
  portability smoke test that runs from outside the repo root.

- 🔴 **P0.2 GPT-OSS expert-routing bug** — expert patch may index `routing_weights` by
  expert ID and conflate top-k with expert count.
  *Next:* add top-k routing tests (32 experts / top-4) and fix against the current
  Transformers GPT-OSS forward.

- 🔴 **P0.3 Expert Hessian accumulation not trustworthy** — `_GptqH` and
  `GPTQ.add_batch` may use different accumulation semantics.
  *Next:* define one canonical accumulator; prove chunked == one-shot within tolerance.

- 🔴 **P0.4 "Parallel Hessian" mode can exceed 80 GB** — hooks on all layers + all
  expert Hessians during one pass can OOM the H100.
  *Next:* implement memory-bounded, resumable layer-group Hessian collection.

- 🔴 **P0.5 Stage 5 result JSON insufficient** — lacks a complete per-tensor manifest.
  *Next:* define a tensor-manifest schema; make Stage 7 fail without required fields.

- 🔴 **P0.6 Stage 5 QDQ vs Stage 7 requantization drift** — packing may re-quantize
  independently, diverging from the values GPTQ optimized.
  *Next:* preserve exact FP4 codes/scales from Stage 5; Stage 7 serializes those.

- 🔴 **P0.7 Stage 7 may not pack GPT-OSS expert tensors** — iterating only over
  `nn.Linear` misses batched `experts.gate_up_proj` / `experts.down_proj`.
  *Next:* prove every expert tensor is in the manifest and packed checkpoint; document
  the vLLM contract in `docs/VLLM_NVFP4_CONTRACT.md`.

- 🔴 **P0.8 NVFP4 scale/packing contract unverified vs pinned vLLM.**
  *Next:* inspect vLLM's NVFP4/ModelOpt loader in `.venv-serve`; add round-trip and
  minimal load tests.

- 🔴 **P0.9 Benchmark script inadequate for final serving claims** — single-process
  `generate(max_tokens=1)` is not a valid TTFT/concurrency benchmark.
  *Next:* build a true async benchmark against a live OpenAI-compatible vLLM server.

---

## Resolved

_(none yet)_
