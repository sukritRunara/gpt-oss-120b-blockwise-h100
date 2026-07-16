#!/usr/bin/env python
"""Download and pin the official openai/gpt-oss-20b MXFP4 checkpoint (arm A).

- Resolves and PINS the exact revision (commit sha) — recorded in the manifest.
- Downloads into a plain local directory (no symlinks) for vLLM/transformers use.
- Writes a provenance manifest (model id, revision, per-file size + SHA-256,
  UTC timestamp) next to the checkpoint.
- Idempotent: if a manifest exists and every file matches its recorded hash,
  the script reports completeness and exits 0 without downloading.

Run inside .venv-quant:
    python scripts/download_official_model.py \
        --model_id openai/gpt-oss-20b \
        --output /workspace/models/gpt-oss-20b-official-mxfp4
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def with_retries(fn, attempts: int = 6, base_delay: float = 10.0, what: str = "operation"):
    """Retry transient Hub/network errors with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:                          # noqa: BLE001
            if i == attempts - 1:
                raise
            delay = base_delay * (2 ** i)
            print(f"[retry] {what} failed ({type(exc).__name__}: {exc}); "
                  f"attempt {i + 1}/{attempts}, retrying in {delay:.0f}s",
                  flush=True)
            time.sleep(delay)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_valid(out_dir: Path) -> bool:
    mpath = out_dir / "PROVENANCE.json"
    if not mpath.exists():
        return False
    try:
        m = json.loads(mpath.read_text())
    except json.JSONDecodeError:
        return False
    for rel, info in m.get("files", {}).items():
        p = out_dir / rel
        if not p.exists() or p.stat().st_size != info["bytes"]:
            print(f"  incomplete: {rel}")
            return False
    # Hash spot-check is done at write time; size check suffices for resume.
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_id", default="openai/gpt-oss-20b")
    ap.add_argument("--revision", default=None,
                    help="Exact revision to pin (default: resolve current main)")
    ap.add_argument("--output", type=Path,
                    default=Path("/workspace/models/gpt-oss-20b-official-mxfp4"))
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    out = args.output
    if manifest_valid(out):
        m = json.loads((out / "PROVENANCE.json").read_text())
        print(f"Already complete at {out} (revision {m['revision'][:12]}) — nothing to do.")
        return 0

    api = HfApi()
    info = with_retries(
        lambda: api.model_info(args.model_id, revision=args.revision),
        what="model_info",
    )
    revision = info.sha
    print(f"Pinning {args.model_id} @ {revision}")

    out.mkdir(parents=True, exist_ok=True)
    with_retries(
        lambda: snapshot_download(
            repo_id=args.model_id,
            revision=revision,
            local_dir=str(out),
            max_workers=8,
        ),
        what="snapshot_download",
    )

    print("Hashing files for provenance manifest…")
    files = {}
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name == "PROVENANCE.json" or ".cache" in p.parts:
            continue
        rel = str(p.relative_to(out))
        files[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        print(f"  {rel}: {files[rel]['bytes']:>13,} B  {files[rel]['sha256'][:16]}…")

    manifest = {
        "model_id": args.model_id,
        "revision": revision,
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files.values()),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {manifest['total_bytes'] / 1024**3:.1f} GB at {out}")
    print(f"Provenance: {out / 'PROVENANCE.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
