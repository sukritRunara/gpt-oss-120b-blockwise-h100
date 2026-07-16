# PROGRESS

Chronological log of major actions. Newest entries at the top. Never record "done"
without pointing to the test, log, or artifact that proves it.

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
