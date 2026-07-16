"""Exact NVFP4 quantization artifacts (P0.6).

GPTQ chooses a specific representable value for every weight: an E2M1 code and
an FP8 microblock scale. The QDQ checkpoint stores only their product, and a
scale cannot be recovered from QDQ values alone (a block's amax equals
6·scale only if some value landed on the ±6.0 grid point). Re-quantizing at
pack time therefore ships a model that is NOT the one GPTQ optimized (P0.6).

This module defines the exact-artifact container and the pack/unpack/dequant
helpers shared by the quantizer capture path (quantizer.py), Stage 5 emission,
Stage 7 serialization, and the invariant tests:

    Stage 5 QDQ weight  ==  dequantize(artifact.codes, artifact.scales)
                            (bit-exact after the same dtype cast)

Storage convention (matches the vLLM ModelOpt W4A16 checkpoint layout):
  codes        : uint8 [out, in//2] — two E2M1 nibbles per byte,
                 low nibble = even column, high nibble = odd column,
                 nibble = grid_index (0-7) | sign_bit << 3
  scales       : float8_e4m3fn [out, in//block_size] — per-microblock scale,
                 normalized by global_scale (D-010 / ModelOpt convention)
  global_scale : float32 [1] — per-tensor scale amax/(6·448); 1.0 when the
                 quantizer ran without a global scale (legacy)

  dequant = E2M1(code) × fp8(scale) × global_scale
"""

from dataclasses import dataclass, field

import torch

# E2M1 representable magnitudes, indexed by the 3 low bits of a nibble
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@dataclass
class QuantizedTensorArtifact:
    """Exact quantization result for one 2-D weight (or expert slice).

    Attributes:
        codes:  uint8 [out, in//2], packed E2M1 nibble pairs (see module doc).
        scales: float8_e4m3fn [out, in//block_size] microblock scales
                (normalized by global_scale).
        block_size: microscaling block width (16 for NVFP4).
        shape: logical (out, in) of the quantized matrix BEFORE packing.
        global_scale: float32 [1] per-tensor scale (vLLM weight_scale_2);
                default 1.0 for artifacts produced without a global scale.
        metadata: free-form (orientation, source module, expert index, …).
    """
    codes: torch.Tensor
    scales: torch.Tensor
    block_size: int
    shape: tuple
    global_scale: torch.Tensor = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.global_scale is None:
            self.global_scale = torch.ones(1, dtype=torch.float32)


def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """[out, in] uint8 nibbles (0-15) → [out, in//2] packed bytes."""
    out_f, in_f = nibbles.shape
    if in_f % 2 != 0:
        raise ValueError(f"in_features={in_f} must be even to pack nibbles.")
    pairs = nibbles.reshape(out_f, in_f // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """[out, in//2] packed bytes → [out, in] uint8 nibbles (0-15)."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)


def nibbles_to_values(nibbles: torch.Tensor) -> torch.Tensor:
    """uint8 nibbles → float32 E2M1 grid values (sign applied)."""
    grid = E2M1_GRID.to(nibbles.device)
    mag = grid[(nibbles & 0x07).long()]
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x01).float()
    return mag * sign


def dequantize_artifact(art: QuantizedTensorArtifact) -> torch.Tensor:
    """Reconstruct the exact float32 QDQ tensor from codes and scales.

    Returns:
        float32 [out, in] — identical (bit-exact after dtype cast) to the
        weight the GPTQ loop wrote back.
    """
    out_f, in_f = art.shape
    values = nibbles_to_values(unpack_nibbles(art.codes))          # [out, in]
    # Effective per-block scale = fp8 stored scale × per-tensor global scale
    # (must match NVFP4Quantizer.find_params exactly for bit-exactness).
    scales = (art.scales.to(torch.float32)
              * art.global_scale.to(torch.float32).item())         # [out, in/bs]
    expanded = scales.repeat_interleave(art.block_size, dim=1)[:, :in_f]
    return values[:, :in_f] * expanded


def verify_artifact_matches(art: QuantizedTensorArtifact,
                            qdq_weight: torch.Tensor,
                            what: str = "tensor"):
    """Enforce the P0.6 invariant: dequantize(artifact) == QDQ weight,
    bit-exact after casting to the QDQ weight's dtype. Raises on mismatch."""
    rec = dequantize_artifact(art).to(qdq_weight.dtype).to(qdq_weight.device)
    if not torch.equal(rec, qdq_weight):
        diff = (rec.float() - qdq_weight.float()).abs()
        raise RuntimeError(
            f"P0.6 invariant violated for {what}: dequantize(artifact) != QDQ "
            f"weight (max abs diff {diff.max().item():.3e}, "
            f"{int((diff > 0).sum())} differing elements). The exported model "
            f"would not be the model GPTQ optimized."
        )


# ── On-disk artifact store (safetensors, one shard per layer) ─────────────────

def artifact_keys(name: str, expert_index=None) -> dict:
    """Canonical safetensors keys for a tensor's codes/scales."""
    stem = name if expert_index is None else f"{name}.expert_{expert_index}"
    return {"codes": f"{stem}.codes", "scales": f"{stem}.scales"}


def save_layer_artifacts(artifact_dir, layer_idx: int, artifacts: dict) -> str:
    """Atomically write one layer's artifacts to a safetensors shard.

    Args:
        artifact_dir: Directory for the store (created if needed).
        layer_idx:    Transformer layer index (names the shard).
        artifacts:    {(name, expert_index_or_None): QuantizedTensorArtifact}

    Returns:
        The shard filename (relative to artifact_dir).
    """
    import os
    from pathlib import Path
    from safetensors.torch import save_file

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fname = f"artifacts_layer_{layer_idx:02d}.safetensors"

    flat = {}
    for (name, e_idx), art in artifacts.items():
        keys = artifact_keys(name, e_idx)
        flat[keys["codes"]] = art.codes.contiguous()
        flat[keys["scales"]] = art.scales.contiguous()
        flat[keys["codes"].replace(".codes", ".global_scale")] = \
            art.global_scale.contiguous()

    tmp = artifact_dir / (fname + ".tmp")
    save_file(flat, str(tmp))
    os.replace(tmp, artifact_dir / fname)
    return fname


def load_artifact(artifact_dir, file: str, name: str, expert_index,
                  block_size: int, shape) -> QuantizedTensorArtifact:
    """Load one tensor's exact artifact back from its shard."""
    from pathlib import Path
    from safetensors import safe_open

    keys = artifact_keys(name, expert_index)
    gs_key = keys["codes"].replace(".codes", ".global_scale")
    with safe_open(str(Path(artifact_dir) / file), framework="pt",
                   device="cpu") as f:
        codes = f.get_tensor(keys["codes"])
        scales = f.get_tensor(keys["scales"])
        global_scale = (f.get_tensor(gs_key) if gs_key in f.keys()
                        else torch.ones(1, dtype=torch.float32))
    return QuantizedTensorArtifact(
        codes=codes, scales=scales,
        block_size=block_size, shape=tuple(shape),
        global_scale=global_scale,
    )


# ── The P0.5 tensor disposition manifest ──────────────────────────────────────

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Every quantized/fallback tensor record must carry these fields; Stage 7
# hard-errors if any are missing (fail closed).
REQUIRED_TENSOR_FIELDS = (
    "name", "param", "kind", "layer_index", "projection", "expert_index",
    "orig_shape", "orientation", "orig_dtype", "requested_format",
    "disposition", "reason", "gptq_blocksize", "scale_block_size",
    "loss", "normalized_loss", "hessian_nsamples", "artifact",
)

DISPOSITIONS = ("GPTQ_NVFP4", "RTN_NVFP4", "BF16_FALLBACK",
                "EXCLUDED_BY_POLICY", "FAILED_INVALID_ARTIFACT")


def write_quant_manifest(artifact_dir, records: list, run_config: dict,
                         excluded: list) -> str:
    """Write the complete tensor-disposition manifest (P0.5).

    Args:
        artifact_dir: quant_artifacts/ directory (manifest lives beside shards).
        records:      list of per-tensor record dicts (REQUIRED_TENSOR_FIELDS).
        run_config:   quantization settings (format, blocksize, calib, seed, …).
        excluded:     [{"param": ..., "reason": ...}] for every model parameter
                      not covered by a record (embeddings, norms, biases,
                      router, lm_head, …).

    Returns:
        Path to the written manifest.
    """
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    for r in records:
        missing = [f for f in REQUIRED_TENSOR_FIELDS if f not in r]
        if missing:
            raise ValueError(f"record {r.get('name')} missing fields: {missing}")
        if r["disposition"] not in DISPOSITIONS:
            raise ValueError(
                f"record {r.get('name')}: unknown disposition {r['disposition']}")
        needs_artifact = r["disposition"] in ("GPTQ_NVFP4", "RTN_NVFP4")
        if needs_artifact and not r["artifact"]:
            raise ValueError(
                f"record {r['name']}: disposition {r['disposition']} requires "
                f"an artifact reference")

    counts = {}
    for r in records:
        counts[r["disposition"]] = counts.get(r["disposition"], 0) + 1

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_config": run_config,
        "counts": counts,
        "n_records": len(records),
        "tensors": records,
        "excluded": excluded,
    }

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, path)
    return str(path)


def read_quant_manifest(manifest_path):
    """Load and validate the manifest; hard-error on schema problems (P0.5).

    Returns the manifest dict. Raises RuntimeError with a precise message on
    any missing/invalid field — Stage 7 must never fall back to guessing.
    """
    import json
    from pathlib import Path

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise RuntimeError(
            f"Quantization manifest not found: {manifest_path}. Stage 7 "
            f"refuses to pack without a complete Stage 5 manifest (P0.5 — no "
            f"fail-open packing)."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Manifest schema_version "
            f"{manifest.get('schema_version')} != {MANIFEST_SCHEMA_VERSION} "
            f"at {manifest_path} — regenerate with the current Stage 5."
        )
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise RuntimeError(f"Manifest at {manifest_path} has no tensor records.")
    for r in tensors:
        missing = [f for f in REQUIRED_TENSOR_FIELDS if f not in r]
        if missing:
            raise RuntimeError(
                f"Manifest record {r.get('name', '<unnamed>')} is missing "
                f"required fields {missing} — refusing to pack (P0.5)."
            )
    return manifest
