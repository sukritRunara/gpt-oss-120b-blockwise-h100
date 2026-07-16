# PROGRESS

Chronological log of major actions. Newest entries at the top. Never record "done"
without pointing to the test, log, or artifact that proves it.

---

## 2026-07-16 07:12 UTC — Environments frozen; P0.1 path fix landed and verified

**Status:** complete

**Goal:** Finish §22 items 5–7: both environments bootstrapped with lockfiles, system
manifest frozen, hard-coded paths repaired and proven portable.

**Commands:**
- `scripts/bootstrap_quant_env.sh` → `.venv-quant` (log: `logs/setup/bootstrap_quant.log`)
- `scripts/bootstrap_serve_env.sh` → `.venv-serve` (log: `logs/setup/bootstrap_serve.log`)
- `scripts/capture_system_manifest.sh` → `envs/system-manifest.txt`
- Stage 1–3 CPU tests run **from outside the repo root** (P0.1 portability proof):
  - `stage1_nvfp4_unit_tests.py` → 5/5 PASS (`logs/tests/stage1_20260716.log`)
  - `stage2_nvfp4_algorithm_tests.py` → 7/7 PASS (`logs/tests/stage2_20260716.log`)
  - `stage3_gpt_oss_shape_tests.py` → 6/6 PASS (`logs/tests/stage3_gpt_oss_20260716.log`)
- `grep -rn "runara_dgx|/home/"` → zero hits

**Environment (frozen in envs/*.lock.txt + system-manifest.txt):**
- `.venv-quant`: torch 2.13.0+cu130, transformers 5.14.0, datasets 5.0.0,
  safetensors 0.8.0, accelerate 1.14.0, kernels 0.16.0, triton 3.7.1
- `.venv-serve`: vllm 0.25.1, torch 2.11.0+cu130, transformers 5.14.0,
  flashinfer 0.6.13 — note the torch version differs from .venv-quant,
  which is exactly why two isolated environments are mandatory.

**Files changed:**
- 10 × `tests/**.py` — `_CODE_ROOT` now `Path(__file__).resolve().parents[N] / "opteam-blockwise-gptq"`
- `tests/internalTests/test_vLLM_deploy_quantized_model.py` — hard-coded MODEL path →
  required `--model` CLI arg
- `envs/{quant,serve}-requirements.lock.txt`, `envs/system-manifest.txt` — new

**Next:** P0.2 — inspect installed transformers 5.14.0 `GptOssExperts.forward`, pin the
routing contract, write the routing test battery, fix the expert patch. Then P0.3
(canonical Hessian accumulator + equivalence tests).

**Blockers:** none.

---

## 2026-07-16 07:05 UTC — §22 first actions: branch, recon, static P0 audit, env bootstrap

**Status:** complete (see follow-up entry above for env results)

**Goal:** Execute handoff §22 items 1–7: recon snapshot, working branch, static audit
of all 9 documented P0 issues against the real source, environment scaffolding.

**Commands:**
- Recon (`pwd`, tree, `git`, `nvidia-smi`, `df`, `free`) → `logs/setup/recon_20260716.log`
- `git switch -c h100-gpt-oss-20b-nvfp4` + pushed to origin
- Read/grepped all core sources: `apply.py`, `gptq.py`, `quantizer.py`,
  `expert_dispatch.py`, `gpt_oss_expert_gptq.py`, `model_utils.py`,
  `stage5_quantize_model.py`, `stage7_save_modelopt.py`, `stage8_benchmark_*.py`
- `scripts/bootstrap_quant_env.sh` / `bootstrap_serve_env.sh` (running,
  logs in `logs/setup/bootstrap_{quant,serve}.log`)

**Results — audit (all 9 P0s verified with file/line evidence in KNOWN_ISSUES.md):**
- P0.1 confirmed: hard-coded DGX `_CODE_ROOT` + RuntimeError guard in 10 files —
  nothing under `tests/` imports on this pod.
- P0.2 version-dependent: expert patch assumes *dense* `routing_weights
  [tokens, num_experts]`; crashes/mis-weights for the top-k-only Transformers variant.
- P0.3 confirmed mathematically: `_GptqH` weights batches `n/N²` vs `GPTQ.add_batch`
  invariant `2/N` — later calibration batches systematically underweighted.
- P0.4 confirmed: ~51 GB expert Hessians allocated **on GPU** during single-pass
  collection + ~42 GB model > 80 GB H100. Cache is written only after the pass.
- P0.5 confirmed: stage5 JSON has no `quantized_attn_keys`/`layer_losses`;
  stage7 requires them.
- P0.6 confirmed: `pack_nvfp4` re-derives scales from QDQ weights (independent second
  quantization → packed ≠ GPTQ-optimized).
- P0.7 confirmed: packing iterates `nn.Linear` only → GPT-OSS batched experts can never
  pack; missing manifest triggers fail-open "pack everything".
- P0.8 confirmed: `weight_scale_2=1.0` Marlin assumption unverified; shared-scale
  computation dead code.
- P0.9 confirmed: TTFT = offline `generate(max_tokens=1)`; no server/concurrency.

**Files changed:**
- `KNOWN_ISSUES.md` — all 9 P0s moved from "unverified" to confirmed with evidence
- `envs/{quant,serve}-requirements.in`, `scripts/bootstrap_{quant,serve}_env.sh`,
  `scripts/capture_system_manifest.sh` — new

**Next:** Freeze lockfiles + system manifest when installs finish; then P0.1 path fix
(smallest, unblocks all tests) → P0.2/P0.3 with tests.

**Blockers:** none.

---

## 2026-07-16 06:42 UTC — Repo bootstrap and project scaffolding

**Status:** complete

**Goal:** Create the GitHub repo, import the `blockwise-gptq` source, and stand up
the progress-tracking / documentation scaffolding before any engineering work.

**Commands:**
- `git add -A && git commit` — initial import (commit `b8329ba`)
- `git push origin main` — pushed to `github.com/sukritRunara/gpt-oss-120b-blockwise-h100`
- `nvidia-smi`, `python3 --version`, `df -h /workspace`, `free -h` — environment probe

**Files changed:**
- `.gitignore` (root) — excludes weights, venvs, caches, logs, secrets, local Claude settings
- `PROGRESS.md`, `DECISIONS.md`, `KNOWN_ISSUES.md` — new
- `README.md` (root) — rewritten to describe the H100 project accurately

**Results / environment (RunPod H100):**
- GPU: 1× NVIDIA H100 80GB HBM3, driver 580.126.09, CUDA 13.0
- Host: Python 3.12.3, ~2 TB RAM, `/workspace` persistent (199 TB free)
- Source tree present under `blockwise-gptq-main/` (code, tests, results, scripts)
- Remote push verified: `main` is on GitHub.

**Next:** Perform handoff §22 immediate actions — audit the source tree, create the
`h100-gpt-oss-20b-nvfp4` working branch, bootstrap the two Python environments
(`.venv-quant`, `.venv-serve`), and freeze a system manifest.

**Blockers:** none.
