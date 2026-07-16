"""Resumable, memory-bounded Hessian cache (P0.4).

Backs the grouped parallel-Hessian collection in apply.py. Design goals
(handoff §P0.4):

  - Completeness is proven by a manifest (per-layer file + SHA-256 + sample
    counts), never by "the directory exists".
  - Layer files are written atomically (temp file + os.replace) so an
    interrupted run can never leave a truncated file that later reads as
    a valid cache entry.
  - The calibration token IDs are cached once, hashed, and reloaded on
    resume — every collection pass (including resumed ones) is guaranteed
    to see the exact same token stream.
  - NaN/Inf Hessians are rejected at save time (fail closed).

Cache layout:

    <cache_dir>/
        manifest.json        (see _empty_manifest for schema)
        calib_tokens.pt      list of (input_ids, targets) CPU tensors
        layer_00.pt          {"attn": {name: {"H": Tensor, "nsamples": int}},
                              "experts": handler payload or None}
        layer_01.pt
        ...
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path, chunk_bytes: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_of_token_samples(samples) -> str:
    """Order-sensitive hash of calibration token IDs.

    Args:
        samples: list of (input_ids, targets) tensor tuples.

    Returns:
        Hex SHA-256 over the concatenated int64 little-endian bytes of all
        input_ids (targets are derived from the same stream and excluded).
    """
    h = hashlib.sha256()
    for input_ids, _targets in samples:
        h.update(input_ids.cpu().to(torch.int64).numpy().tobytes())
    return h.hexdigest()


class HessianCache:
    """Manifest-backed per-layer Hessian cache with atomic writes and resume.

    One instance corresponds to one (model, dataset, nsamples, seqlen, seed)
    collection configuration — the caller derives cache_dir from that key.
    """

    def __init__(self, cache_dir, n_layers: int, meta: dict):
        """
        Args:
            cache_dir: Directory for this specific cache key.
            n_layers:  Total transformer layers the cache must cover.
            meta:      Identifying config (model_name, dataset, nsamples,
                       seqlen, seed) stored in the manifest and validated
                       on resume.
        """
        self.cache_dir = Path(cache_dir)
        self.n_layers = n_layers
        self.meta = dict(meta)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_or_init_manifest()

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def _empty_manifest(self) -> dict:
        return {
            "schema_version": _SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "n_layers": self.n_layers,
            "meta": self.meta,
            "token_hash": None,
            "layers": {},          # str(layer_idx) → entry dict
            "collection_stats": [],  # one entry per collected group
        }

    def _load_or_init_manifest(self) -> dict:
        path = self._manifest_path()
        if not path.exists():
            return self._empty_manifest()
        try:
            manifest = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"Hessian cache manifest is unreadable: {path} ({exc}). "
                f"Delete the cache directory to recollect from scratch."
            ) from exc
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError(
                f"Hessian cache schema mismatch at {path} "
                f"(found {manifest.get('schema_version')}, "
                f"expected {_SCHEMA_VERSION}). Delete the cache to recollect."
            )
        if manifest.get("n_layers") != self.n_layers:
            raise RuntimeError(
                f"Hessian cache at {self.cache_dir} was built for "
                f"{manifest.get('n_layers')} layers but the model has "
                f"{self.n_layers}. Wrong cache key or wrong model — refusing "
                f"to reuse it."
            )
        if manifest.get("meta") != self.meta:
            raise RuntimeError(
                f"Hessian cache meta mismatch at {self.cache_dir}:\n"
                f"  cached : {manifest.get('meta')}\n"
                f"  current: {self.meta}\n"
                f"Refusing to mix calibration configurations."
            )
        return manifest

    def _write_manifest(self):
        """Atomic manifest write (temp + replace)."""
        path = self._manifest_path()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2))
        os.replace(tmp, path)

    # ── Calibration token cache ───────────────────────────────────────────────

    def ensure_tokens(self, samples):
        """Return the authoritative calibration samples for this cache.

        First run: saves `samples` to calib_tokens.pt, records their hash.
        Resume:    reloads the CACHED tokens (ignoring `samples`) and verifies
                   the file still matches the recorded hash, guaranteeing
                   every group pass sees the identical token stream even if
                   the upstream dataset or sampler changed.
        """
        tok_path = self.cache_dir / "calib_tokens.pt"

        if self.manifest["token_hash"] is not None:
            if not tok_path.exists():
                raise RuntimeError(
                    f"Hessian cache records token_hash but {tok_path} is "
                    f"missing. Cache is corrupt — delete it to recollect."
                )
            cached = torch.load(tok_path, map_location="cpu", weights_only=False)
            actual = sha256_of_token_samples(cached)
            if actual != self.manifest["token_hash"]:
                raise RuntimeError(
                    f"Calibration token cache hash mismatch at {tok_path}: "
                    f"manifest {self.manifest['token_hash'][:16]}…, "
                    f"file {actual[:16]}…. The token cache must be immutable "
                    f"— delete the cache directory to recollect."
                )
            return cached

        samples_cpu = [(i.cpu(), t.cpu()) for i, t in samples]
        tmp = tok_path.with_suffix(".pt.tmp")
        torch.save(samples_cpu, tmp)
        os.replace(tmp, tok_path)
        self.manifest["token_hash"] = sha256_of_token_samples(samples_cpu)
        self._write_manifest()
        return samples_cpu

    # ── Layer entries ─────────────────────────────────────────────────────────

    def _layer_file(self, layer_idx: int) -> Path:
        return self.cache_dir / f"layer_{layer_idx:02d}.pt"

    def layer_complete(self, layer_idx: int, verify_hash: bool = True) -> bool:
        """A layer is complete only if the manifest entry exists, the file
        exists, and (by default) the file's SHA-256 matches the manifest."""
        entry = self.manifest["layers"].get(str(layer_idx))
        if entry is None:
            return False
        path = self._layer_file(layer_idx)
        if not path.exists():
            return False
        if entry.get("file") != path.name:
            return False
        if verify_hash and _sha256_file(path) != entry.get("sha256"):
            return False
        return True

    def pending_layers(self, verify_hash: bool = True):
        return [i for i in range(self.n_layers)
                if not self.layer_complete(i, verify_hash=verify_hash)]

    def is_complete(self, verify_hash: bool = True) -> bool:
        return not self.pending_layers(verify_hash=verify_hash)

    @staticmethod
    def _check_finite(layer_idx: int, payload: dict):
        """Reject NaN/Inf Hessians at save time (fail closed)."""
        def _chk(tag, H):
            if H is not None and not torch.isfinite(H).all():
                raise RuntimeError(
                    f"Layer {layer_idx} Hessian '{tag}' contains NaN/Inf — "
                    f"refusing to cache a corrupt Hessian."
                )
        for name, e in payload.get("attn", {}).items():
            _chk(name, e.get("H"))
        experts = payload.get("experts")
        if experts:
            for side in ("gu", "dn"):
                for e_idx, e in experts.get(side, {}).items():
                    _chk(f"experts.{side}[{e_idx}]", e.get("H"))

    def save_layer(self, layer_idx: int, payload: dict, stats: dict = None):
        """Atomically persist one layer's Hessians and update the manifest.

        Args:
            layer_idx: Transformer layer index.
            payload:   {"attn": {name: {"H": cpu Tensor, "nsamples": int}},
                        "experts": handler payload or None}. Tensors must
                        already be on CPU.
            stats:     Optional collection stats merged into the entry
                       (seconds, gpu_peak_bytes, …).
        """
        self._check_finite(layer_idx, payload)

        path = self._layer_file(layer_idx)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

        attn_nsamples = {n: e["nsamples"] for n, e in payload["attn"].items()}
        expert_nsamples = None
        if payload.get("experts"):
            expert_nsamples = {
                side: {str(k): v["nsamples"] for k, v in
                       payload["experts"].get(side, {}).items()}
                for side in payload["experts"]
            }

        entry = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "attn_nsamples": attn_nsamples,
            "expert_nsamples": expert_nsamples,
            "completed_utc": _utc_now(),
        }
        if stats:
            entry.update(stats)
        self.manifest["layers"][str(layer_idx)] = entry
        self._write_manifest()

    def load_layer(self, layer_idx: int) -> dict:
        if not self.layer_complete(layer_idx, verify_hash=False):
            raise RuntimeError(
                f"Layer {layer_idx} is not complete in the Hessian cache at "
                f"{self.cache_dir} — collect it before quantizing."
            )
        return torch.load(self._layer_file(layer_idx), map_location="cpu",
                          weights_only=False)

    def record_group_stats(self, stats: dict):
        """Append per-group collection stats (runtime, memory) to the manifest."""
        self.manifest["collection_stats"].append(stats)
        self._write_manifest()

    def total_bytes(self) -> int:
        return sum(e.get("bytes", 0) for e in self.manifest["layers"].values())
