# Agent Handoff: GPT-OSS-20B Blockwise GPTQ → NVFP4 on a RunPod H100

## Audience

This document is intended to be pasted into or referenced by **Codex CLI, Claude Code, or another coding agent running directly inside the RunPod instance**.

The agent should treat this file as the project specification, engineering plan, and progress-management contract.

---

## 1. Mission

Use the existing `blockwise-gptq` source repository on a **single RunPod H100** to:

1. Download and preserve the official Hugging Face `openai/gpt-oss-20b` checkpoint.
2. Dequantize the official MXFP4 checkpoint into a clean BF16 source checkpoint.
3. Repair and harden the repository's GPT-OSS blockwise GPTQ implementation.
4. Produce a blockwise-GPTQ **NVFP4 W4A16** model artifact.
5. Produce a matched **RTN NVFP4 W4A16** control artifact.
6. Validate tensor-level, logit-level, and task-level correctness.
7. Serve the baseline and treatment models on the same H100 using vLLM.
8. Benchmark quality, memory use, TTFT, inter-token latency, throughput, and concurrency behavior.
9. Leave behind documented, reproducible code, locked environments, raw results, and a useful README.

Everything happens on the RunPod H100. This is **not an AWS Trainium assignment**.

---

## 2. Repository availability and assumptions

### Present on the RunPod instance

- The primary `blockwise-gptq` repository/source code.
- A single NVIDIA H100 with CUDA support.
- Persistent storage mounted under `/workspace` or an equivalent persistent path.

### Not present on the RunPod instance

- The prior GPT-OSS-20B sample repository.

Do **not** search for or depend on the sample repository. The important lessons and known fixes from it are included in this document.

### Expected source repository structure

The existing repository is expected to contain files similar to:

```text
blockwise-gptq/
├── README.md
├── requirements.txt
├── opteam-blockwise-gptq/
│   ├── apply.py
│   ├── calibration.py
│   ├── expert_dispatch.py
│   ├── gpt_oss_expert_gptq.py
│   ├── gptq.py
│   ├── model_utils.py
│   └── quantizer.py
├── scripts/
├── tests/
│   ├── stage1_nvfp4_unit_tests.py
│   ├── stage2_gptq_algorithm_tests.py
│   ├── stage3_gpt_oss_shape_tests.py
│   ├── stage4_baseline_perplexity.py
│   ├── stage5_quantize_model.py
│   ├── stage6_eval_perplexity.py
│   ├── stage7_save_modelopt.py
│   ├── stage8_benchmark_nvfp4_vs_bf16.py
│   └── internalTests/
└── results/
```

Inspect the actual tree before changing anything. Do not assume filenames or APIs are unchanged.

---

## 3. Scope decisions already made

### Primary quantization target

**NVFP4 W4A16**:

- FP4 E2M1 weight values.
- NVFP4-style local scaling.
- BF16/FP16 activations at inference.
- Non-native H100 serving path, likely through a weight-only kernel such as Marlin depending on the pinned vLLM version.

H100 does not have native Blackwell FP4 Tensor Cores. That is understood and is not a reason to abandon NVFP4. The project is evaluating the algorithm, memory footprint, and practical serving behavior on H100; a speedup is not guaranteed.

### Important non-goals

Do not silently expand the assignment into any of these projects:

- AWS Trainium or Neuron deployment.
- Native Blackwell benchmarking.
- Implementing a complete MXFP4 GPTQ exporter.
- Rewriting the entire repository before establishing a pilot.
- Publishing checkpoints to Hugging Face without explicit permission.
- Claiming that NVFP4 is the optimal H100 format.

FP8 may be added later as an optional H100-native comparison, but it is not the initial treatment.

### Format support in the existing repository

The source quantizer registry is expected to include:

- `nvfp4`
- `fp8`
- `int8`
- `int4`
- `int4_perchannel`
- `mxint4`

For GPT-OSS expert tensors, the existing special handler is expected to support only:

- `nvfp4`
- `fp8`
- `int8`

The existing Stage 7 exporter is expected to attempt packed output for NVFP4, FP8, and INT8, while INT4/MXINT4 packing is incomplete. Verify this in the actual source.

**MXINT4 is not MXFP4.** The repository does not currently generate the official GPT-OSS MXFP4 format.

---

## 4. Source-model provenance and the meaning of “BF16 baseline”

The official `openai/gpt-oss-20b` checkpoint contains MXFP4-quantized expert weights. We will load/decode those weights and save their dequantized values as BF16.

The resulting source is:

```text
Official MXFP4 checkpoint
    → exact MXFP4 decode/dequantization
    → BF16 tensors representing the decoded MXFP4 values
```

It is **not** the unavailable original pre-MXFP4 master checkpoint.

Use accurate names everywhere, for example:

```text
gpt-oss-20b-official-mxfp4
gpt-oss-20b-mxfp4-dequant-bf16
```

Do not label the dequantized source as “original BF16.”

The resulting experiment is a **transquantization experiment**:

```text
Official MXFP4
    → dequantized BF16 source
    → blockwise GPTQ NVFP4
```

This is sound, but provenance must remain explicit in the README, manifests, and report.

---

## 5. Required comparison arms

Create and evaluate these four arms:

| ID | Model | Purpose |
|---|---|---|
| A | Official `openai/gpt-oss-20b` MXFP4 | Real-world Hugging Face deployment baseline |
| B | Official MXFP4 decoded and saved as BF16 | Exact source passed into the quantizer |
| C | RTN NVFP4 generated from B | Same target format and packing path, without GPTQ |
| D | Blockwise-GPTQ NVFP4 generated from B | Primary treatment |

Interpret comparisons correctly:

- **D vs C:** pure blockwise-GPTQ algorithm benefit at the same format.
- **D vs B:** incremental quality/memory/performance change caused by NVFP4 conversion.
- **D vs A:** practical custom artifact vs the official Hugging Face checkpoint.
- **A vs B:** validates the dequantization process.

For a valid D-vs-C comparison, RTN and GPTQ must use:

- The exact same tensor inclusion/exclusion mask.
- The exact same block size and scale rules.
- The exact same packing and vLLM loading path.
- The same BF16 fallback tensors, if mixed precision is retained.

The only intended difference must be the quantization algorithm.

---

## 6. Agent operating rules

### Work carefully and visibly

Before changing code:

1. Inspect the repository tree.
2. Record `git status`, current branch, current commit, and repository remotes.
3. Create a dedicated branch such as:

```bash
git switch -c h100-gpt-oss-20b-nvfp4
```

4. Preserve the original source. Do not overwrite the only copy without Git history.

### Do not run the full quantization immediately

The existing code has correctness, memory, manifest, expert-packing, and benchmark issues. A full run before repairing them can waste hours and produce a misleading artifact.

Use this order:

```text
static audit
→ environment bootstrap
→ focused unit tests
→ dequantization validation
→ tiny end-to-end pilot
→ packed-load validation
→ RTN/GPTQ matched pilot
→ full calibration
→ final serving benchmarks
```

### Fail closed

Do not silently:

- Pack all `nn.Linear` modules when a manifest is missing.
- Leave GPT-OSS expert tensors in BF16 while calling the model “full NVFP4.”
- Replace failed GPTQ tensors with RTN without recording it.
- Replace failed NVFP4 packing with BF16 without recording it.
- Ignore missing tensor keys.
- Ignore vLLM warnings about unsupported kernels or unrecognized quantization metadata.
- Continue after NaN/Inf, mismatched tensor counts, or incomplete expert coverage.

A partial/hybrid model is acceptable as an intermediate debugging artifact, but it must be named and documented as partial/hybrid.

### Protect persistent data and secrets

- Work under `/workspace` or the instance's persistent volume.
- Do not store Hugging Face tokens, SSH keys, API keys, or credentials in the repository.
- Do not terminate or resize the RunPod instance from code.
- Do not delete official model downloads or completed artifacts unless explicitly requested.
- Use `tmux`, `screen`, or another durable session for long jobs.
- Pipe long-running command output through `tee` into timestamped log files.

---

## 7. Required workspace layout

Use a clear persistent layout. Adapt the root if the repository already lives elsewhere.

```text
/workspace/gpt-oss-20b-blockwise/
├── repo/                         # Git working tree
├── models/
│   ├── official-mxfp4/
│   ├── mxfp4-dequant-bf16/
│   ├── rtn-nvfp4-qdq/
│   ├── rtn-nvfp4-packed/
│   ├── gptq-nvfp4-qdq/
│   └── gptq-nvfp4-packed/
├── cache/
│   ├── huggingface/
│   ├── datasets/
│   ├── calibration_tokens/
│   ├── layer_inputs/             # only if this design is selected
│   └── hessians/
├── logs/
│   ├── setup/
│   ├── tests/
│   ├── dequantization/
│   ├── quantization/
│   ├── serving/
│   └── benchmarks/
├── results/
│   ├── raw/
│   ├── quality/
│   ├── serving/
│   ├── profiles/
│   └── summaries/
└── manifests/
```

Set persistent caches explicitly, for example:

```bash
export PROJECT_ROOT=/workspace/gpt-oss-20b-blockwise
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$PROJECT_ROOT/cache/datasets"
```

Add an `env.sh.example` to the repository, but do not commit secrets.

---

## 8. Two isolated Python environments are mandatory

Create two independent environments because quantization/export dependencies and vLLM dependencies may conflict.

### Environment 1: `.venv-quant`

Purpose:

- Transformers model loading and MXFP4 dequantization.
- Calibration datasets.
- Blockwise GPTQ.
- QDQ validation.
- Exact-code/scale artifact creation.
- ModelOpt-compatible checkpoint export.
- Unit/integration tests.

### Environment 2: `.venv-serve`

Purpose:

- vLLM serving.
- OpenAI-compatible API tests.
- Async serving benchmarks.
- vLLM source inspection for the accepted NVFP4 contract.
- Optional profiling utilities.

### Environment requirements

Prefer `uv` when available; otherwise use `python -m venv`.

Create reproducibility artifacts such as:

```text
envs/
├── quant-requirements.in
├── quant-requirements.lock.txt
├── serve-requirements.in
├── serve-requirements.lock.txt
└── system-manifest.txt
```

The system manifest must record at least:

```text
UTC timestamp
RunPod pod/host identifier if available
GPU name and VRAM
nvidia-smi output
NVIDIA driver
CUDA runtime/toolkit
Python version
PyTorch version
Transformers version
Datasets version
Safetensors version
ModelOpt version if used
vLLM version and Git commit if available
Triton version
Git repository commit
```

Do not use an unpinned `latest` dependency for the final benchmark. A temporary exploratory install is acceptable, but the final successful versions must be frozen.

Add scripts similar to:

```text
scripts/bootstrap_quant_env.sh
scripts/bootstrap_serve_env.sh
scripts/capture_system_manifest.sh
```

Each script must be idempotent or clearly document destructive behavior.

---

## 9. Required project documentation and progress tracking

Create or improve all of the following:

```text
README.md
PROGRESS.md
DECISIONS.md
KNOWN_ISSUES.md
docs/ARCHITECTURE.md
docs/EXPERIMENT_DESIGN.md
docs/VLLM_NVFP4_CONTRACT.md
docs/DEQUANTIZATION_PROVENANCE.md
docs/TROUBLESHOOTING.md
```

### README requirements

The README must eventually contain:

1. Project goal and non-goals.
2. Hardware and model scope.
3. Exact source-model provenance.
4. The four experimental arms.
5. Repository architecture.
6. Two-environment setup and activation commands.
7. Dequantization command and validation command.
8. Unit-test commands.
9. Pilot quantization command.
10. Full quantization command.
11. RTN-control command.
12. Packing/export command.
13. vLLM launch commands for each model.
14. Benchmark commands.
15. Output directory descriptions.
16. How to resume an interrupted run.
17. Known limitations, especially non-native NVFP4 on H100.
18. Current results or a link to the generated summary.

Do not leave the original README's inaccurate “quick start” as the primary path if it would trigger an unsafe full run.

### PROGRESS.md requirements

Update `PROGRESS.md` before and after each major action. Use entries like:

```markdown
## 2026-07-15 23:10 UTC — Fix GPT-OSS routing

**Status:** complete

**Goal:** Correct expert routing-weight indexing for top-k routing.

**Commands:**
- `pytest -q tests/test_gpt_oss_routing.py`

**Files changed:**
- `opteam-blockwise-gptq/gpt_oss_expert_gptq.py`
- `tests/test_gpt_oss_routing.py`

**Results:**
- 7 tests passed.
- Expert IDs above top-k are handled correctly.

**Next:** Repair Hessian accumulator equivalence.

**Blockers:** none.
```

Never write “done” without recording the test or artifact that proves it.

### DECISIONS.md requirements

Record non-obvious decisions, alternatives considered, and why the chosen approach is acceptable. Important examples:

- Why NVFP4 remains the primary format on H100.
- Why dequantized MXFP4 is a transquantization source.
- Which Hessian-collection strategy was chosen.
- Which vLLM version and NVFP4 contract were pinned.
- Whether mixed-precision fallback is retained.
- Which tensors are intentionally excluded.

### KNOWN_ISSUES.md requirements

Track open issues with:

- Severity: P0/P1/P2.
- Reproduction command.
- Evidence/log path.
- Current hypothesis.
- Next action.
- Whether it invalidates current artifacts.

### Code documentation requirements

- Add docstrings to new public functions and classes.
- Comment non-obvious tensor layouts, transposes, scale conventions, and Hessian math.
- Include shapes and dtypes in comments where confusion is likely.
- Avoid comments that merely repeat code.
- Keep CLI help text accurate.
- Add type hints to new code where practical.
- Do not leave dead experimental code in the main path; place experiments under `experiments/` or remove them after conclusions are recorded.

### Git progress requirements

Make small, reviewable commits. Suggested sequence:

```text
chore: bootstrap h100 project structure and environments
fix: make repository paths portable
fix: correct gpt-oss top-k expert routing
fix: unify expert and standard hessian accumulation
feat: add memory-bounded hessian collection
feat: add source dequantization and provenance checks
feat: emit complete tensor quantization manifest
feat: preserve exact nvfp4 codes and scales
feat: pack gpt-oss expert tensors for vllm
feat: add matched rtn control
feat: add serving benchmark harness
 docs: complete readme and experiment report
```

Do not combine unrelated fixes into one giant commit.

---

## 10. Known P0 source issues that must be addressed

### P0.1 — Hard-coded paths

Several stage scripts may contain DGX-specific paths such as `/home/.../projects/...`.

Replace them with repository-relative paths, normally:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "opteam-blockwise-gptq"
```

Add a focused portability test or smoke command that runs from a working directory outside the repository root.

Do not rely on the process current working directory.

---

### P0.2 — GPT-OSS expert-routing bug

The existing expert patch may incorrectly treat the top-k dimension as the number of experts and may index `routing_weights` by expert ID.

For GPT-OSS-20B, there are many routed experts but only a small top-k per token. The correct logic must distinguish:

- `expert_id`
- `top_k_position`
- `token_index`

The essential correction is expected to resemble:

```python
num_experts = experts.gate_up_proj.shape[0]

mask = F.one_hot(router_indices, num_classes=num_experts + 1)
mask = mask.permute(2, 1, 0)

for expert_index_tensor in active_experts:
    expert_id = expert_index_tensor.item()
    if expert_id == num_experts:
        continue

    top_k_position, token_index = torch.where(mask[expert_id])
    current_state = flat_hidden_states[token_index]

    # ... expert computation ...

    weighted = expert_output * routing_weights[
        token_index, top_k_position, None
    ]
```

Do not copy this blindly without checking the current Transformers GPT-OSS forward implementation. Match its exact shape semantics and masking behavior.

Required tests:

- 32 experts with top-4 routing.
- Expert IDs greater than 4.
- Repeated expert IDs across tokens.
- An expert selected at different top-k positions.
- Tokens routed to multiple experts.
- Unselected experts.
- Padding/masking slot if the model uses one.
- Patched forward vs a direct reference implementation.
- Output and gradient are not required during calibration if the path is explicitly no-grad, but numerical forward equivalence is required.

---

### P0.3 — Expert Hessian accumulation is not trustworthy

The lightweight `_GptqH` accumulator and `GPTQ.add_batch` must have identical, clearly defined accumulation semantics.

The existing implementation may weight chunks incorrectly, especially when expert batches contain variable numbers of routed tokens.

Required redesign/test behavior:

1. Define one canonical Hessian accumulator function.
2. Define whether the sample count means sequences, batches, or flattened activation rows. Prefer a mathematically explicit row-count convention unless compatibility requires otherwise.
3. Make both standard linears and GPT-OSS experts use the same convention.
4. Verify that chunked accumulation equals one-shot accumulation within tolerance.
5. Verify that unequal chunk sizes produce the same result.
6. Verify 2D and 3D activation inputs.
7. Verify cached-and-restored Hessians produce the same quantized output.
8. Verify no accidental extra factors of `n`, `1/n`, or `sqrt(n)` are introduced.

A useful direct reference is a normalized Gram matrix of the form:

```text
H ∝ XᵀX
```

The exact scalar convention must be documented and must match the damping/quantization implementation. Do not change only `_GptqH` while leaving `GPTQ.add_batch` semantically different.

---

### P0.4 — Existing “parallel Hessian” mode can exceed H100 memory

The current parallel path may attach hooks to all transformer layers and allocate all expert Hessians during one full-model calibration pass. The source code itself anticipates expert Hessians consuming tens of gigabytes. Combined with a roughly 40 GB BF16 source model, this can exceed an 80 GB H100.

The current cache path may save Hessians only **after** the full collection pass, which does not reduce peak collection memory.

Do not attempt the full GPT-OSS-20B calibration until this is repaired.

#### Recommended memory-bounded design

Preserve original-model Hessians without quantization cascade by collecting them in layer groups:

```text
1. Keep the entire source model unchanged.
2. Cache the exact calibration token IDs once.
3. Select a configurable group of layers, initially group_size=1.
4. Attach hooks/accumulators only to that group.
5. Run the full unchanged model over the cached calibration samples.
6. Save each group's Hessians immediately to CPU/disk.
7. Detach hooks and free GPU memory.
8. Repeat until all layers have cached Hessians.
9. Only after the complete cache exists, quantize layers sequentially from cache.
```

This costs additional full-model forward passes but preserves the key benefit of “parallel” Hessians: all statistics come from the unmodified source model.

Make `--hessian_layer_group_size` configurable. Begin with 1 for the pilot and increase only after measuring memory headroom.

Alternative designs are allowed, such as caching original per-layer inputs, but they must be documented and proven to avoid both OOM and quantization cascade.

Required memory evidence:

- Peak GPU memory during collection.
- Peak host RAM if measurable.
- Hessian cache size on disk.
- Runtime per layer group.
- Successful resume from an existing partial cache.

Cache completeness must be verified by a manifest, not merely by checking that a directory exists.

---

### P0.5 — Stage 5 result JSON is insufficient

The existing Stage 5 JSON may record only high-level settings and total loss, while Stage 7 expects fields such as a quantized-layer list or per-layer losses.

Replace this with a complete machine-readable tensor manifest.

For every eligible tensor or expert slice, record at least:

```text
canonical tensor name
module/parameter name
layer index
projection type
expert index when applicable
original shape
logical quantization orientation
original dtype
requested format
actual disposition
  - GPTQ_NVFP4
  - RTN_NVFP4
  - BF16_FALLBACK
  - EXCLUDED_BY_POLICY
  - FAILED_INVALID_ARTIFACT
block size
local scale granularity
loss metric
normalized loss metric if available
Hessian sample count
condition diagnostics if available
reason for fallback/exclusion
packed artifact key(s)
checksum or hash where practical
```

Save both:

- A JSON summary/manifest.
- Exact tensor metadata in safetensors or another non-pickle format where possible.

Missing required manifest fields must be a hard error in Stage 7.

---

### P0.6 — Stage 5 QDQ followed by Stage 7 requantization can change the result

The current flow may:

1. Quantize with GPTQ.
2. Save dequantized BF16 values.
3. Recompute NVFP4 codes and scales independently during packing.

That can produce a packed model different from the one GPTQ optimized.

Redesign the pipeline so Stage 5 preserves the **exact representable result** selected by GPTQ:

```text
GPTQ
├── QDQ BF16 checkpoint for correctness and quality evaluation
└── exact quantization artifact
    ├── FP4 codes
    ├── local scales
    ├── global scales if required
    ├── tensor layout/orientation metadata
    └── tensor disposition manifest
```

Stage 7 must serialize those exact codes/scales rather than run an unrelated second quantization.

A clean API could look like:

```python
@dataclass
class QuantizedTensorArtifact:
    qdq_weight: torch.Tensor
    packed_codes: torch.Tensor
    local_scales: torch.Tensor
    global_scale: torch.Tensor | None
    metadata: dict[str, object]
```

The exact design may differ, but the following invariant is mandatory:

```text
Stage 5 QDQ weight
≈ dequantize(Stage 7 exact packed codes and scales)
```

Use strict tolerances appropriate to the format and dtype.

---

### P0.7 — Existing Stage 7 does not prove GPT-OSS experts are packed

GPT-OSS expert weights are batched parameters such as:

```text
experts.gate_up_proj
experts.down_proj
```

They are not ordinary `nn.Linear` modules. A Stage 7 implementation that iterates only over `model.named_modules()` and packs `nn.Linear` weights will miss the dominant expert tensors.

Do not call the artifact full NVFP4 unless:

- Every intended expert tensor/slice appears in the manifest.
- Every intended expert tensor is represented in the packed checkpoint.
- The selected vLLM loader recognizes those tensors.
- vLLM actually executes the intended low-bit expert path.

Inspect the installed pinned vLLM source and the official GPT-OSS model implementation. Create `docs/VLLM_NVFP4_CONTRACT.md` documenting:

```text
vLLM version/commit
quantization class used
model class used for GPT-OSS
expected config.json quantization fields
expected tensor names
expected tensor shapes
expected tensor dtypes
expert packing layout
scale layout
kernel selected on H100
fallback behavior
```

Include source-file paths and line numbers from the pinned local installation.

Before exporting the full 20B model, construct a small synthetic or single-layer fixture that uses the same tensor-key/layout contract and prove that the loader accepts it.

---

### P0.8 — NVFP4 scale and packing contract must be verified against pinned vLLM

The existing exporter may make assumptions such as:

- `weight_scale_2` always equals one.
- A particular dtype for global scales.
- A particular local-scale normalization rule.
- One checkpoint layout works identically across vLLM, SGLang, and TensorRT-LLM.

Treat all of those assumptions as untrusted until checked against the pinned vLLM implementation.

Required actions:

1. Inspect vLLM's actual NVFP4/ModelOpt loader and kernel code in `.venv-serve`.
2. Document exact expected shapes and dtypes.
3. Add pack→dequantize round-trip tests.
4. Add a minimal vLLM load test.
5. Capture server logs showing the quantization method and selected kernel.
6. Treat an unexpected fallback as a benchmark result, not as proof of NVFP4 acceleration.

Do not rely on format names alone. Prove the runtime tensor contract.

---

### P0.9 — Existing benchmark script is not adequate for final serving claims

Do not use a single-process `generate(max_tokens=1)` timer as the final TTFT benchmark.

The final benchmark must operate against a live OpenAI-compatible vLLM server and issue true concurrent asynchronous requests.

Use vLLM's maintained benchmark tooling when suitable, supplemented by a custom script when necessary to capture all required metrics and raw per-request records.

---

## 11. Dequantization implementation

Create a dedicated script, for example:

```text
scripts/dequantize_gpt_oss_20b.py
```

The script must:

1. Accept model ID, revision, output path, dtype, and device-placement arguments.
2. Pin and record the exact Hugging Face revision.
3. Use the installed Transformers MXFP4 dequantization mechanism supported by that pinned version.
4. Decode the official checkpoint into BF16 tensors.
5. Save a clean BF16 safetensors checkpoint.
6. Save tokenizer, config, generation config, and required custom code/config files.
7. Remove or rewrite stale quantization metadata only after verifying that reload behaves as an ordinary BF16 checkpoint.
8. Produce a provenance manifest with file hashes and source revision.
9. Be resumable or clearly detect a complete prior output.

Do not assume a specific Transformers API without checking the installed version. An expected pattern may involve an MXFP4 quantization config with a dequantization option, but verify it locally.

### Dequantization validation

Create a script such as:

```text
scripts/validate_dequantized_source.py
```

Required checks:

- Reload succeeds without reconstructing packed MXFP4 modules.
- All model parameters use the expected BF16/FP32 dtypes.
- No unexpected packed-MXFP4 scale/code tensors remain.
- The local `config.json` no longer incorrectly claims the checkpoint should be loaded as MXFP4.
- Parameter count matches the official model.
- Tensor names and shapes are complete.
- No NaN/Inf.
- Deterministic outputs are stable.
- A vs B logit comparison is acceptably close on a held-out prompt set.

Record:

```text
mean/worst logit cosine similarity
KL divergence
next-token top-1 agreement
maximum absolute logit difference
text-generation prefix agreement
```

Do not use only generated prose as proof of equivalence.

---

## 12. Required tests before any full model run

Preserve existing tests, but do not assume they are sufficient.

Add focused pytest-style tests where practical.

### Quantizer tests

- Legal E2M1 values.
- Zero tensor.
- Constant tensor.
- Very small and very large values.
- Odd/even input widths where packing requires even dimensions.
- Local-scale shape and dtype.
- Global-scale behavior.
- Determinism.
- Pack/unpack ordering.
- QDQ equals dequantized packed representation.

### GPTQ tests

- GPTQ reconstruction is no worse than matched RTN on representative synthetic data.
- Chunked Hessian accumulation equals one-shot accumulation.
- Unequal chunk sizes.
- Cached Hessian reload.
- Cholesky/damping failure behavior.
- NaN/Inf guards.
- Exact quantized-code preservation.

### GPT-OSS expert tests

- 32 experts, top-4 routing.
- Expert IDs above top-k.
- Correct routing-weight lookup by top-k position.
- Gate/up interleaving is preserved.
- Down projection orientation is preserved.
- Write-back does not transpose incorrectly.
- Zero-activation expert behavior is explicit.
- All expert dispositions are manifested.
- Packed batched-expert round trip.

### Portability tests

- Scripts import successfully from outside the repo working directory.
- No path contains a developer-specific home directory.
- Output paths are CLI-configurable.

### Stage-contract tests

- Stage 5 manifest contains all required fields.
- Stage 7 refuses to run without a complete manifest.
- Stage 7 consumes exact codes/scales.
- Tensor counts before and after export match policy.
- Ignored/excluded tensors exactly match config.
- A tiny exported artifact loads in the pinned vLLM version.

All P0 tests must pass before a full quantization run.

---

## 13. Pilot protocol

The first end-to-end pilot should be deliberately small:

```text
calibration samples: 16 or 32
sequence length: 256 or 512
Hessian layer group size: 1
GPTQ block width: 128
NVFP4 microblock: 16
percdamp: 0.01
fixed random seed
cached token IDs
```

The pilot must run the complete path:

```text
dequantized BF16 source
→ original-model Hessian cache
→ blockwise GPTQ
→ exact codes/scales
→ QDQ checkpoint
→ complete manifest
→ packed NVFP4 checkpoint
→ vLLM load
→ deterministic generation
→ QDQ/packed equivalence
```

Also produce a matched RTN pilot using the same tensor mask.

### Pilot exit gates

Do not begin the full run until all are true:

- Portable-path tests pass.
- Expert routing tests pass.
- Hessian equivalence tests pass.
- Peak H100 memory is within a safe bound.
- Hessian caching resumes correctly.
- All intended dense and expert tensors receive a disposition.
- No silent BF16 or RTN fallback.
- Stage 5 QDQ equals Stage 7 packed-dequantized values within tolerance.
- Packed model loads in vLLM.
- vLLM logs reveal the actual quantization/kernel path.
- Harmony-formatted chat generation succeeds.
- Official MXFP4 and dequantized BF16 baseline smoke tests succeed.
- RTN and GPTQ artifacts use identical tensor masks.

If any gate fails, update `KNOWN_ISSUES.md` and do not launch the expensive full calibration.

---

## 14. Full quantization protocol

After the pilot is validated, use an initial full configuration similar to:

```text
calibration dataset: C4 or another documented general-text calibration set
calibration samples: target 512
sequence length: 2048
fixed seed: recorded
GPTQ block width: 128
NVFP4 local block: 16
percdamp: 0.01
Hessian source: unchanged dequantized BF16 model
Hessian collection: memory-bounded layer groups
```

Do not blindly accept defaults. Run a small representative sweep before the final full run on:

- One attention projection.
- One output projection.
- One expert `gate_up` slice.
- One expert `down` slice.

Candidate GPTQ block widths:

```text
64, 128, 256
```

Candidate damping values, only if conditioning requires it:

```text
0.01, 0.03, 0.10
```

Choose one documented global configuration rather than overfitting each tensor to evaluation data.

### Mixed precision policy

The current raw-loss threshold is not sufficient by itself because loss magnitude can depend on tensor dimensions and activation scale.

Record normalized metrics such as:

- Relative output MSE.
- Relative Frobenius error.
- Output cosine similarity.
- Held-out logit impact.
- Hessian conditioning diagnostics.

A mixed-precision mask may be used, but:

1. It must be frozen before final RTN vs GPTQ evaluation.
2. RTN and GPTQ must share the exact same mask.
3. BF16 fallbacks must be explicit in the model name and manifest.
4. The percentage of model bytes remaining BF16 must be reported.

A fully NVFP4-eligible model is preferable for the primary claim if quality remains acceptable, but never force quantization through an invalid tensor merely to achieve a label.

### Resume behavior

The full run must support interruption and restart:

- Calibration token cache is immutable and hashed.
- Hessian cache has a per-layer completeness manifest.
- Quantized tensor artifacts are written atomically.
- Completed layers are skipped only after checksum validation.
- Partial output cannot be mistaken for a complete checkpoint.
- Progress is recorded after each layer.

---

## 15. Matched RTN control

Create a first-class RTN path, not an ad hoc fallback.

The RTN control must:

- Start from the same dequantized BF16 source.
- Use the frozen tensor mask from the GPTQ experiment.
- Use the same NVFP4 codebook.
- Use the same local/global scale rules.
- Use the same tensor orientations.
- Use the same exporter.
- Use the same vLLM config and kernel path.
- Record exact codes/scales and a complete manifest.

The RTN implementation should be callable independently, for example:

```text
scripts/build_rtn_control.py
```

Do not classify “expert received no calibration samples and fell back to RTN” as the matched RTN control. That is a failure/fallback case, not the control model.

---

## 16. Correctness and quality evaluation

Keep performance and quality evaluation separate.

### Tensor-level correctness

For every quantized tensor or expert slice:

```text
exact stored codes/scales
    → dequantization
    ≈ Stage 5 QDQ tensor
```

Record mean, max, and percentile errors. Any layout mismatch, expert-index mismatch, or transpose mismatch invalidates the artifact.

### Layer-level correctness

Use representative layer inputs to compare:

- BF16 source output.
- RTN NVFP4 QDQ output.
- GPTQ NVFP4 QDQ output.
- Packed-dequantized output.

Record:

- Cosine similarity.
- Relative MSE.
- Maximum absolute error.
- NaN/Inf count.

### Logit-level paired tests

Use a fixed held-out set, separate from calibration. Compare B, C, and D with identical inputs.

Record:

- Mean and worst logit cosine similarity.
- KL divergence.
- Next-token top-1 agreement.
- Top-k agreement.
- Maximum absolute logit error.
- Generation prefix agreement.

### Perplexity

Perplexity may be retained as a diagnostic, but it must not be the only quality metric. Validate the evaluation code carefully if quantization appears to improve perplexity implausibly.

Use the exact same cached token sequences for baseline and quantized models.

### Task-level quality

Use a modest, reproducible suite that samples at least:

- General knowledge/reasoning.
- Math.
- Code.
- Instruction following.

Save every prompt, raw output, parser result, score, and error reason.

Use Harmony-compatible formatting for GPT-OSS chat/instruction prompts.

Do not tune calibration parameters against the final held-out evaluation set.

---

## 17. vLLM serving validation

Use `.venv-serve` only.

For each model arm:

1. Start from a fresh server process.
2. Save the exact launch command.
3. Save complete stdout/stderr.
4. Record model-load time.
5. Record VRAM after load.
6. Confirm tokenizer/chat-template behavior.
7. Run deterministic smoke prompts.
8. Inspect logs for selected quantization method and kernel.
9. Confirm the OpenAI-compatible endpoint.
10. Shut down cleanly before launching another arm.

Create launch scripts or a parameterized launcher, for example:

```text
scripts/serve_vllm.sh
configs/serve-official-mxfp4.env
configs/serve-dequant-bf16.env
configs/serve-rtn-nvfp4.env
configs/serve-gptq-nvfp4.env
```

Keep these identical across arms except for the model path and only those flags strictly required by the format.

Use one H100 and tensor parallel size 1 unless the environment unexpectedly exposes multiple GPUs and the user explicitly changes scope.

---

## 18. Serving benchmark design

Build or adapt a true asynchronous serving benchmark that writes one JSONL record per request.

### Core suites

#### Prefill-focused

```text
prompt lengths: 1k, 8k, 32k tokens
output length: 1 token
concurrency: 1, 8, 32
```

#### Decode-focused

```text
prompt length: 64–256 tokens
output length: 256 or 512 tokens
concurrency: 1, 8, 32, 64
```

#### Mixed online serving

```text
prompt lengths: 1k, 8k, 32k tokens
output length: 256 tokens
concurrency: 1, 8, 32, 64
```

#### Long-context capacity

```text
prompt length: 128k tokens
begin at concurrency 1
run only after a memory/capacity check
```

A model's inability to fit a long-context case is a valid result. Do not hide it.

### Prompt construction

- Construct prompts using the same tokenizer and Harmony/chat template.
- Verify **post-template token length**, not character length.
- Save prompt hashes and token counts.
- Avoid accidental prefix-cache reuse in the core comparison.
- Disable prefix caching unless it is the subject of a separate benchmark.

### Sampling and output settings

For performance runs:

- Fixed output-token count.
- Deterministic or controlled sampling.
- Ignore EOS only when necessary to guarantee equal output lengths.

For quality runs:

- Natural EOS behavior.
- Identical sampling parameters across arms.

### Warmup and repetitions

Use:

```text
pilot: 10–20 warmup requests and 20–50 measured requests per cell
final: at least 200 measured requests per normal cell
repetitions: 3 independent runs per cell
```

If runtime is excessive, document the reduced sample count rather than silently changing it.

Alternate model order across repetitions to reduce thermal/time-order bias.

### Required metrics

Per request and aggregate:

- TTFT.
- Inter-token latency or time per output token.
- End-to-end latency.
- p50, p90, and p99 latency.
- Request throughput.
- Input-token throughput.
- Output-token throughput.
- Total-token throughput.
- Success/failure/timeout/OOM counts.
- Actual input and output token counts.

Per server/run:

- Model-load time.
- Idle and peak VRAM.
- KV-cache capacity if exposed.
- GPU utilization.
- HBM/memory utilization when available.
- Power draw.
- SM and memory clocks.
- Temperature.
- vLLM version and launch flags.
- Selected quantization and kernel path.

Use `nvidia-smi` logging or DCGM where available. Do not require privileged tools that the pod does not provide.

### Statistical reporting

For each benchmark cell, retain:

- Raw JSONL requests.
- Each of the three repetition summaries.
- Median across repetitions.
- Bootstrap 95% confidence interval when practical.
- Percentage change vs the relevant control.
- Failure counts and censored requests.

Do not report only one averaged tokens-per-second number.

---

## 19. Results and artifact naming

Use names that encode provenance and algorithm clearly.

Examples:

```text
gpt-oss-20b-official-mxfp4
gpt-oss-20b-mxfp4-dequant-bf16
gpt-oss-20b-mxfp4-dequant-rtn-nvfp4
gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4
```

If mixed precision remains, append a clear suffix such as:

```text
-mixed-bf16
```

Do not call a checkpoint simply `gpt-oss-20b-nvfp4` if it contains unreported BF16 expert weights.

Every final model directory must include or reference:

```text
source revision
source file hashes
quantization config
calibration manifest
exact tensor disposition manifest
environment manifest
Git commit
quality summary
serving compatibility summary
```

---

## 20. Final deliverables

The repository/project is complete only when it contains:

1. Patched source code in Git.
2. Clean, current README.
3. `PROGRESS.md`, `DECISIONS.md`, and `KNOWN_ISSUES.md`.
4. Two environment bootstrap scripts and lockfiles.
5. System/version manifest.
6. Official MXFP4 source manifest.
7. Dequantized BF16 source checkpoint and provenance report.
8. Memory-bounded, resumable Hessian collection.
9. Complete tensor quantization manifest.
10. GPTQ NVFP4 QDQ checkpoint.
11. GPTQ NVFP4 packed checkpoint.
12. Matched RTN NVFP4 QDQ and packed checkpoints.
13. Proof that dense and GPT-OSS expert tensors follow the intended packed path.
14. QDQ-vs-packed equivalence report.
15. Held-out tensor/layer/logit/task quality results.
16. vLLM launch configurations.
17. Raw per-request benchmark JSONL.
18. Aggregated CSV/JSON summaries and plots.
19. Kernel-selection logs and at least one representative profiler or runtime trace if feasible.
20. A final report separating:

```text
GPTQ vs RTN
NVFP4 vs decoded BF16
custom NVFP4 vs official MXFP4
quality vs memory vs serving performance
```

A null or negative speed result is acceptable. A misleading or unproven speed claim is not.

---

## 21. Definition of done

The main assignment is done when all of the following are true:

- The official model revision is pinned and reproducible.
- The dequantized BF16 source is validated and accurately named.
- P0 path, routing, Hessian, memory, manifest, and packing issues are resolved or explicitly documented as blockers.
- The full calibration can run without exceeding H100 memory.
- The run can resume from a partial Hessian cache.
- Every intended tensor has an explicit disposition.
- No silent fallback occurs.
- Exact GPTQ codes/scales survive export.
- GPT-OSS expert tensors are actually packed or the artifact is explicitly labeled hybrid.
- The packed model loads in pinned vLLM.
- Runtime logs identify the actual H100 kernel/fallback path.
- RTN and GPTQ use identical tensor masks.
- Correctness and quality results are saved.
- All four arms have comparable serving results where they can load.
- Raw benchmark data and commands are retained.
- README instructions work from a fresh shell.
- Git working tree is clean or remaining local files are documented.

---

## 22. Immediate first actions for the coding agent

Perform these actions first, in order:

1. Locate the repository and persistent project root.
2. Print and save:

```bash
pwd
find . -maxdepth 3 -type f | sort
git status --short --branch
git rev-parse HEAD
nvidia-smi
python3 --version
df -h
free -h
```

3. Create the project branch.
4. Create `PROGRESS.md`, `DECISIONS.md`, and `KNOWN_ISSUES.md`.
5. Create the two environment bootstrap scripts.
6. Freeze an initial system manifest.
7. Audit and replace hard-coded paths.
8. Add the GPT-OSS top-k routing tests and fix the routing implementation.
9. Add Hessian chunking/equivalence tests and unify accumulator semantics.
10. Redesign Hessian collection so peak memory is bounded before any full run.
11. Add a Stage 5 tensor-manifest schema and make Stage 7 fail without it.
12. Inspect the pinned vLLM NVFP4 and GPT-OSS loader source and document the exact contract.
13. Implement and validate the dequantization script.
14. Run only the tiny end-to-end pilot.
15. Update the README with verified commands, not speculative commands.

Do not start the 512 × 2048 full quantization until the pilot exit gates pass.

---

## 23. Status-report format to give the user

When reporting progress, use this structure:

```markdown
## Current status

- Phase:
- Completed:
- In progress:
- Blocked:

## Evidence

- Tests:
- Logs:
- Artifacts:
- Git commit:

## Important finding

One concise explanation of the most important technical result or blocker.

## Next action

The exact next command or engineering task.
```

Be direct about uncertainty. Never report an artifact as valid merely because a script completed.
