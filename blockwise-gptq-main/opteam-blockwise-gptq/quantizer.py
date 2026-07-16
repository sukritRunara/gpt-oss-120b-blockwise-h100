"""
Pluggable quantizer classes for GPTQ.

Provides per-channel (per-row) symmetric quantization to different number formats.
Each quantizer implements find_params() to compute scales and quantize_dequantize()
for the round-trip used during GPTQ error compensation.
"""

import torch
from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    """Abstract base class for weight quantizers."""

    def __init__(self, device="cpu"):
        self.device = device
        self.scale = None
        self.maxq = None

    @abstractmethod
    def find_params(self, x, weight=True):
        """Compute per-channel quantization parameters from weight tensor.

        Args:
            x: Weight tensor [out_features, in_features].
            weight: If True, compute per-channel (per-row) scales.
        """
        pass

    @abstractmethod
    def quantize_dequantize(self, x, col_idx=None, col_start=0):
        """Quantize then immediately dequantize a weight column/slice.

        Args:
            x: Weight slice to quantize, shape [out_features] or [out_features, k].
            col_idx: Optional column index in the original weight matrix.
                     Used by group quantizers to select the correct scale group.
            col_start: Starting column offset when x is a slice of the full
                       weight matrix. Used by block/group quantizers to select
                       the correct pre-computed scales.

        Returns:
            Dequantized weight in float32, same shape as input.
        """
        pass

    def ready(self):
        """Whether quantization parameters have been computed."""
        return self.scale is not None

    @abstractmethod
    def get_format_name(self):
        """Return string identifier for this format."""
        pass


class FP8E4M3Quantizer(BaseQuantizer):
    """Per-channel symmetric quantization to float8_e4m3fn.

    scale[i] = amax(|W[i,:]|) / 448.0
    The actual cast to float8_e4m3fn ensures the round-trip matches
    what H100 Tensor Cores will see.
    """

    _FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max

    def find_params(self, x, weight=True):
        if weight:
            # Per-channel (per-row) scale
            amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        else:
            amax = x.abs().amax().clamp(min=1e-12).unsqueeze(0)
        self.scale = amax / self._FP8_MAX
        self.scale = self.scale.to(self.device)

    def quantize_dequantize(self, x, col_idx=None, col_start=0):
        scaled = x / self.scale
        clamped = scaled.clamp(-self._FP8_MAX, self._FP8_MAX)
        # Cast to FP8 and back to get true quantization error
        fp8 = clamped.to(torch.float8_e4m3fn)
        dequant = fp8.to(torch.float32) * self.scale
        return dequant

    def get_format_name(self):
        return "fp8_e4m3"


class Int8SymQuantizer(BaseQuantizer):
    """Per-channel symmetric 8-bit integer quantization.

    scale[i] = amax(|W[i,:]|) / 127.0
    """

    _INT8_MAX = 127

    def find_params(self, x, weight=True):
        if weight:
            amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        else:
            amax = x.abs().amax().clamp(min=1e-12).unsqueeze(0)
        self.scale = amax / self._INT8_MAX
        self.scale = self.scale.to(self.device)

    def quantize_dequantize(self, x, col_idx=None, col_start=0):
        scaled = x / self.scale
        rounded = scaled.round().clamp(-128, 127)
        dequant = rounded * self.scale
        return dequant

    def get_format_name(self):
        return "int8_sym"


class Int4SymGroupQuantizer(BaseQuantizer):
    """Per-channel, per-group symmetric 4-bit integer quantization.

    Groups of `group_size` columns share one scale per output channel:
        scale[i, g] = amax(|W[i, g*gs:(g+1)*gs]|) / 7.0

    With only 16 quantization levels, group quantization is essential
    for maintaining quality at 4-bit precision.
    """

    _INT4_MAX = 7

    def __init__(self, group_size=128, device="cpu"):
        super().__init__(device=device)
        self.group_size = group_size

    def find_params(self, x, weight=True):
        out_features, in_features = x.shape
        gs = self.group_size

        # Pad columns to a multiple of group_size
        if in_features % gs != 0:
            pad = gs - (in_features % gs)
            x = torch.nn.functional.pad(x, (0, pad), value=0.0)

        n_groups = x.shape[1] // gs
        # Reshape to [out_features, n_groups, group_size]
        x_grouped = x.reshape(out_features, n_groups, gs)
        amax = x_grouped.abs().amax(dim=2).clamp(min=1e-12)  # [out_features, n_groups]
        self.scale = (amax / self._INT4_MAX).to(self.device)
        self._in_features = in_features

    def quantize_dequantize(self, x, col_idx=None, col_start=0):
        gs = self.group_size
        if col_idx is not None:
            # GPTQ per-column path: x is [out_features, 1]
            group_idx = col_idx // gs
            s = self.scale[:, group_idx].unsqueeze(1)  # [out_features, 1]
            scaled = x / s
            rounded = scaled.round().clamp(-8, self._INT4_MAX)
            return rounded * s
        else:
            # Full-matrix / block-wise GPTQ path.
            # col_start indicates the starting column offset in the original
            # weight matrix, so we select the correct pre-computed group scales.
            out_features, in_features = x.shape
            col_end = col_start + in_features

            # Pad to align end to group boundary
            aligned_end = ((col_end + gs - 1) // gs) * gs
            pad_right = aligned_end - col_end
            if pad_right > 0:
                x_padded = torch.nn.functional.pad(x, (0, pad_right), value=0.0)
            else:
                x_padded = x

            # Pad to align start to group boundary
            pad_left = col_start % gs
            if pad_left > 0:
                x_padded = torch.nn.functional.pad(x_padded, (pad_left, 0), value=0.0)

            n_groups = x_padded.shape[1] // gs
            group_start = (col_start - pad_left) // gs
            group_end = group_start + n_groups
            x_grouped = x_padded.reshape(out_features, n_groups, gs)
            s = self.scale[:, group_start:group_end].unsqueeze(2)  # [out_features, n_groups, 1]
            scaled = x_grouped / s
            rounded = scaled.round().clamp(-8, self._INT4_MAX)
            dequant = (rounded * s).reshape(out_features, -1)
            return dequant[:, pad_left:pad_left + in_features]

    def get_format_name(self):
        return "int4_sym_group"


class Int4SymQuantizer(BaseQuantizer):
    """Per-channel symmetric 4-bit integer quantization.

    scale[i] = amax(|W[i,:]|) / 7.0
    Quantizes to [-8, 7] range (signed 4-bit).
    """

    _INT4_MAX = 7
    _INT4_MIN = -8

    def find_params(self, x, weight=True):
        if weight:
            amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        else:
            amax = x.abs().amax().clamp(min=1e-12).unsqueeze(0)
        self.scale = amax / self._INT4_MAX
        self.scale = self.scale.to(self.device)

    def quantize_dequantize(self, x):
        scaled = x / self.scale
        rounded = scaled.round().clamp(self._INT4_MIN, self._INT4_MAX)
        dequant = rounded * self.scale
        return dequant

    def get_format_name(self):
        return "int4_sym"


class MXInt4Quantizer(BaseQuantizer):
    """MXINT4 quantizer: blocks of 32 elements with E8M0 shared exponent.

    Per OCP MX specification. Each block of 32 contiguous columns shares
    a power-of-2 scale factor (E8M0 format). Element values are symmetric
    INT4 [-7, 7].

    Key differences from Int4SymGroupQuantizer:
    - Fixed block size of 32 (MX standard)
    - Scale is power-of-2 only: 2^floor(log2(amax)) / 7.0
    - Symmetric clamp to [-7, 7] (not [-8, 7])
    """

    MX_BLOCK_SIZE = 32
    MXINT4_MAX = 7

    def __init__(self, device="cpu"):
        super().__init__(device=device)

    def find_params(self, x, weight=True):
        out_features, in_features = x.shape
        bs = self.MX_BLOCK_SIZE

        # Pad columns to a multiple of MX_BLOCK_SIZE
        if in_features % bs != 0:
            pad = bs - (in_features % bs)
            x = torch.nn.functional.pad(x, (0, pad), value=0.0)

        n_blocks = x.shape[1] // bs
        x_blocked = x.reshape(out_features, n_blocks, bs)
        amax = x_blocked.abs().amax(dim=2).clamp(min=1e-12)  # [out_features, n_blocks]

        # E8M0: power-of-2 scale = 2^floor(log2(amax)) / 7.0
        e8m0_amax = torch.exp2(torch.floor(torch.log2(amax)))
        self.scale = (e8m0_amax / self.MXINT4_MAX).to(self.device)
        self._in_features = in_features

    def quantize_dequantize(self, x, col_idx=None, col_start=0):
        bs = self.MX_BLOCK_SIZE
        if col_idx is not None:
            # GPTQ per-column path: x is [out_features, 1]
            block_idx = col_idx // bs
            s = self.scale[:, block_idx].unsqueeze(1)  # [out_features, 1]
            scaled = x / s
            rounded = scaled.round().clamp(-self.MXINT4_MAX, self.MXINT4_MAX)
            return rounded * s
        else:
            # Full matrix / block-wise GPTQ path.
            # col_start indicates the starting column offset in the original
            # weight matrix, so we select the correct pre-computed MX scales.
            out_features, in_features = x.shape
            col_end = col_start + in_features

            # Pad to align end to MX block boundary
            aligned_end = ((col_end + bs - 1) // bs) * bs
            pad_right = aligned_end - col_end
            if pad_right > 0:
                x_padded = torch.nn.functional.pad(x, (0, pad_right), value=0.0)
            else:
                x_padded = x

            # Pad to align start to MX block boundary
            pad_left = col_start % bs
            if pad_left > 0:
                x_padded = torch.nn.functional.pad(x_padded, (pad_left, 0), value=0.0)

            n_blocks = x_padded.shape[1] // bs
            block_start = (col_start - pad_left) // bs
            block_end = block_start + n_blocks
            s = self.scale[:, block_start:block_end].unsqueeze(2)  # [out_features, n_blocks, 1]

            x_blocked = x_padded.reshape(out_features, n_blocks, bs)
            scaled = x_blocked / s
            rounded = scaled.round().clamp(-self.MXINT4_MAX, self.MXINT4_MAX)
            dequant = (rounded * s).reshape(out_features, -1)

            # Remove padding to return original shape
            return dequant[:, pad_left:pad_left + in_features]

    def get_format_name(self):
        return "mxint4"


class NVFP4Quantizer(BaseQuantizer):
    """NVIDIA FP4 (E2M1) quantizer with FP8 E4M3 microscaling.

    Per NVIDIA Blackwell specification:
    - 4-bit E2M1 weights: 1 sign, 2 exponent, 1 mantissa bit
    - Representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    - Shared FP8 E4M3 scale per block of `block_size` columns (default 16)

    The `requires_per_block_params = True` flag tells fasterquant_blockwise
    to call find_params() once per GPTQ block (on the current W_block) rather
    than once globally. This ensures each microscaling block gets a scale
    derived from the error-compensated weights at quantization time.

    Scale computation (ModelOpt / vLLM W4A16 convention, D-010):
        global_scale = amax(|W_tensor|) / (6.0 * 448.0)        # fp32, per tensor
        raw_scale    = amax(|W_micro_block|) / 6.0
        scale_fp8    = quantize_to_fp8_e4m3(raw_scale / global_scale)
        effective    = scale_fp8 * global_scale                # used for QDQ
        dequant      = E2M1(code) * scale_fp8 * global_scale

    Normalizing block scales by a per-tensor global scale keeps them in the
    upper fp8-e4m3 range (near 448) instead of the subnormal region, and is
    the exact form vLLM's Marlin W4A16 kernel consumes (weight_scale_2).
    The global scale must be FIXED before GPTQ starts (computed from the
    original tensor) because blocks are quantized progressively on
    error-compensated weights. Set it via set_global_scale()/
    set_global_scale_from(); when unset, a global scale of 1.0 is used
    (legacy behavior — raw fp8 block scales).
    """

    # E2M1 representable magnitudes (sign is handled separately)
    _E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    _E2M1_MAX = 6.0
    _FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max

    # Signals fasterquant_blockwise to call find_params() per GPTQ block
    requires_per_block_params = True

    def __init__(self, block_size=16, device="cpu"):
        """
        Args:
            block_size: Number of columns sharing one FP8 scale (hardware-fixed at 16).
            device: Target device.
        """
        super().__init__(device=device)
        self.block_size = block_size
        # Keep grid on the target device to avoid repeated .to() calls
        self._grid = self._E2M1_GRID.to(device)
        # Exact-artifact capture state (P0.6) — see begin_capture()
        self._cap = None
        # Per-tensor global scale (D-010). None → 1.0 (legacy raw fp8 scales).
        self.global_scale = None
        # fp8-stored form of the current block scales (what capture records)
        self._scale_fp8 = None

    # ── Per-tensor global scale (D-010) ───────────────────────────────────────

    def set_global_scale(self, value: float):
        """Pin the per-tensor global scale (fp32 scalar).

        Use one shared value across tensors that a serving engine fuses into
        a single GEMM (e.g. q/k/v): vLLM applies max(weight_scale_2) to the
        fused weight WITHOUT rescaling the fp8 group scales.
        """
        self.global_scale = float(max(value, 1e-12))

    def set_global_scale_from(self, w):
        """Compute the ModelOpt global scale amax/(6·448) from a weight tensor."""
        self.set_global_scale(w.detach().abs().amax().item() / (6.0 * self._FP8_MAX))

    # ── Exact-artifact capture (P0.6) ─────────────────────────────────────────
    # GPTQ's per-block scales are computed on error-compensated weights and
    # then discarded; they CANNOT be recovered from the QDQ output. Capture
    # records the exact codes and scales as quantize_dequantize() produces
    # them, so Stage 7 can serialize the model GPTQ actually optimized.

    def begin_capture(self, out_features: int, in_features: int):
        """Start recording exact codes/scales for one [out, in] tensor.

        The subsequent quantize_dequantize(x, col_start=…) calls must tile the
        full column range exactly once (fasterquant_blockwise's left-to-right
        sweep, or a single full-width RTN call). end_capture() verifies
        coverage.
        """
        if in_features % self.block_size != 0:
            raise ValueError(
                f"capture requires in_features % block_size == 0 "
                f"(got {in_features} % {self.block_size})"
            )
        self._cap = {
            "nibbles": torch.zeros(out_features, in_features, dtype=torch.uint8),
            "scales":  torch.zeros(out_features, in_features // self.block_size,
                                   dtype=torch.float8_e4m3fn),
            "covered": torch.zeros(in_features, dtype=torch.bool),
            "shape":   (out_features, in_features),
        }

    def end_capture(self):
        """Finish capture and return a QuantizedTensorArtifact.

        Raises if any column was never quantized (incomplete sweep).
        """
        from quant_artifacts import QuantizedTensorArtifact, pack_nibbles

        cap, self._cap = self._cap, None
        if cap is None:
            raise RuntimeError("end_capture() without begin_capture()")
        if not bool(cap["covered"].all()):
            missing = int((~cap["covered"]).sum())
            raise RuntimeError(
                f"capture incomplete: {missing} columns were never quantized"
            )
        s2 = self.global_scale if self.global_scale is not None else 1.0
        return QuantizedTensorArtifact(
            codes=pack_nibbles(cap["nibbles"]),
            scales=cap["scales"],
            block_size=self.block_size,
            shape=cap["shape"],
            global_scale=torch.tensor([s2], dtype=torch.float32),
        )

    def abort_capture(self):
        """Drop any in-progress capture state (e.g. on quantization failure)."""
        self._cap = None

    def find_params(self, x, weight=True):
        """Compute FP8 E4M3 microscaling factors for blocks of `block_size` columns.

        Called per GPTQ block in fasterquant_blockwise, so x is typically
        [out_features, gptq_blocksize]. Also works on the full weight matrix
        (e.g. for RTN fallback).

        The raw scale is quantized to FP8 E4M3 to match hardware behavior —
        using an unquantized float32 scale would underestimate the true
        quantization error during GPTQ optimization.

        Args:
            x: Weight tensor [out_features, in_features].
            weight: Unused; NVFP4 always uses block microscaling.
        """
        out_features, in_features = x.shape
        bs = self.block_size

        # Pad columns to a multiple of block_size
        pad = (bs - in_features % bs) % bs
        if pad:
            x = torch.nn.functional.pad(x, (0, pad), value=0.0)

        n_blocks = x.shape[1] // bs
        x_blocked = x.reshape(out_features, n_blocks, bs)

        # Raw scale: map block max to E2M1 max
        amax = x_blocked.abs().amax(dim=2).clamp(min=1e-12)  # [out_features, n_blocks]
        raw_scale = amax / self._E2M1_MAX

        # Store the block scales in FP8 E4M3 — normalized by the per-tensor
        # global scale when set (D-010 / ModelOpt convention: keeps fp8
        # values in the high range instead of the coarse subnormal region),
        # raw otherwise (legacy). The effective per-block scale used for QDQ
        # is always fp8_value * global, so QDQ, capture, and the serving
        # kernel see the identical quantization.
        s2 = self.global_scale if self.global_scale is not None else 1.0
        self._scale_fp8 = (raw_scale / s2).clamp(max=self._FP8_MAX) \
                                          .to(torch.float8_e4m3fn)
        self.scale = (self._scale_fp8.to(torch.float32) * s2).to(self.device)

    def _round_to_e2m1(self, x_scaled, return_nibbles=False):
        """Snap each value to the nearest point in the E2M1 grid (sign-preserving).

        Args:
            x_scaled: Tensor already divided by its block scale (arbitrary shape).
            return_nibbles: If True, also return the 4-bit codes
                (grid_index | sign_bit << 3) as uint8 — used by artifact capture.

        Returns:
            Tensor of same shape with values snapped to ±{0, 0.5, ..., 6.0},
            or (values, nibbles) when return_nibbles=True.
        """
        grid = self._grid.to(x_scaled.device)
        sign = x_scaled.sign()
        abs_x = x_scaled.abs()
        # Compute L1 distance to each of the 8 grid points: [..., 8]
        dists = (abs_x.unsqueeze(-1) - grid).abs()
        idx = dists.argmin(dim=-1)
        values = sign * grid[idx]
        if not return_nibbles:
            return values
        # Sign bit from strict negativity: -0.0 inputs get idx 0, sign bit 1 —
        # dequantizes to -0.0, which compares equal to +0.0.
        nibbles = (idx | ((x_scaled < 0).long() << 3)).to(torch.uint8)
        return values, nibbles

    def quantize_dequantize(self, x, col_idx=None, col_start=0):
        """Quantize to E2M1 and dequantize using the FP8 microscaling factors.

        Since find_params() is called per GPTQ block, self.scale always
        corresponds to the current x — col_start is not used for scale
        indexing (unlike group quantizers that hold global pre-computed scales).

        When a capture is active (begin_capture), the exact nibble codes and
        FP8 scales for columns [col_start, col_start+in_features) are recorded
        so the artifact matches this call's output bit-for-bit (P0.6).

        Args:
            x: Weight block [out_features, in_features].
            col_idx: Unused for NVFP4 (per-block params make column indexing
                     unnecessary); must be None while capturing.
            col_start: Column offset of x within the full weight matrix. Not
                       used for scale indexing; required for capture placement.

        Returns:
            Dequantized weights in float32, same shape as x.
        """
        out_features, in_features = x.shape
        bs = self.block_size

        # Pad to multiple of block_size (mirrors find_params padding)
        pad = (bs - in_features % bs) % bs
        if pad:
            x_padded = torch.nn.functional.pad(x, (0, pad), value=0.0)
        else:
            x_padded = x

        n_blocks = x_padded.shape[1] // bs
        x_blocked = x_padded.reshape(out_features, n_blocks, bs)

        # Scale: [out_features, n_blocks] → [out_features, n_blocks, 1]
        s = self.scale[:, :n_blocks].unsqueeze(2)

        # Scale, snap to E2M1 grid, dequantize
        if self._cap is not None:
            if col_idx is not None:
                raise RuntimeError(
                    "artifact capture supports the blockwise path only "
                    "(col_idx per-column calls cannot be captured)"
                )
            if pad:
                raise RuntimeError(
                    "artifact capture requires block-aligned widths "
                    f"(got {in_features} with block_size {bs})"
                )
            x_quant, nibbles = self._round_to_e2m1(x_blocked / s,
                                                   return_nibbles=True)
            self._cap["nibbles"][:, col_start:col_start + in_features] = \
                nibbles.reshape(out_features, -1)[:, :in_features].cpu()
            sb = col_start // bs
            # Record the fp8-stored form (normalized by global_scale when
            # set) — exactly what find_params produced for this block sweep.
            self._cap["scales"][:, sb:sb + n_blocks] = \
                self._scale_fp8[:, :n_blocks].cpu()
            self._cap["covered"][col_start:col_start + in_features] = True
        else:
            x_quant = self._round_to_e2m1(x_blocked / s)
        x_dequant = (x_quant * s).reshape(out_features, -1)

        # Strip padding to return original shape
        return x_dequant[:, :in_features]

    def get_format_name(self):
        return "nvfp4_e2m1"


QUANTIZER_REGISTRY = {
    "fp8": FP8E4M3Quantizer,
    "int8": Int8SymQuantizer,
    "int4": Int4SymGroupQuantizer,
    "int4_perchannel": Int4SymQuantizer,
    "mxint4": MXInt4Quantizer,
    "nvfp4": NVFP4Quantizer,
}