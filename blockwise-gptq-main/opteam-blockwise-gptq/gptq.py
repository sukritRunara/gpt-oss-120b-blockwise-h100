"""
GPTQ algorithm implementation.

Adapted from ist-daslab/gptq. Uses second-order (Hessian) information from
calibration data to optimally quantize weight matrices, minimizing output
error via error propagation through the inverse Hessian.

One GPTQ instance per nn.Linear layer.
"""

import torch
import torch.nn as nn
import math


class GPTQ:
    """GPTQ quantizer for a single nn.Linear layer.

    Accumulates the Hessian from calibration inputs, then runs the
    fasterquant algorithm to produce optimally quantized weights.
    """

    def __init__(self, layer):
        """Initialize GPTQ for a linear layer.

        Args:
            layer: nn.Linear module to quantize.
        """
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone().float()
        self.rows = W.shape[0]  # out_features
        self.columns = W.shape[1]  # in_features
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        """Accumulate Hessian from a calibration batch.

        Args:
            inp: Input activations, shape [batch, seq, in_features] or [batch, in_features].
            out: Output activations (unused, kept for hook compatibility).
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        batch_size = inp.shape[0]

        inp = inp.float().to(self.dev)

        # Incremental Hessian update
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_scaled = math.sqrt(2.0 / self.nsamples) * inp.t()
        self.H += inp_scaled @ inp_scaled.t()

    def fasterquant(self, blocksize=128, percdamp=0.01):
        """Run the GPTQ fasterquant algorithm.

        Quantizes the layer's weights using inverse-Hessian-based error
        compensation, processing columns in blocks for efficiency.

        Args:
            blocksize: Number of columns to process per block.
            percdamp: Damping factor as percentage of mean diagonal.

        Returns:
            Total squared quantization loss.
        """
        W = self.layer.weight.data.clone().float()
        H = self.H

        # Compute per-channel scales
        self.quantizer.find_params(W)

        # Zero out dead columns (no calibration signal)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Damp diagonal for numerical stability
        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += damp

        # Cholesky-based inverse Hessian (computed on CPU to avoid GPU Cholesky bug #136405)
        H_cpu = H.cpu()
        H_cpu = torch.linalg.cholesky(H_cpu)
        H_cpu = torch.cholesky_inverse(H_cpu)
        H_cpu = torch.linalg.cholesky(H_cpu, upper=True)
        Hinv = H_cpu.to(self.dev)
        del H_cpu

        total_loss = torch.zeros(self.rows, device=self.dev, dtype=torch.float32)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            Hinv_block = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W_block[:, i]
                d = Hinv_block[i, i]

                q = self.quantizer.quantize_dequantize(w.unsqueeze(1), col_idx=i1 + i).squeeze(1)
                Q_block[:, i] = q
                err = (w - q) / d
                Err_block[:, i] = err

                total_loss += err ** 2

                # Propagate error to remaining columns in block
                W_block[:, i:] -= err.unsqueeze(1) * Hinv_block[i, i:].unsqueeze(0)

            # Write quantized block back
            W[:, i1:i2] = Q_block

            # Propagate block error to all remaining columns
            if i2 < self.columns:
                W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]

        # Write dequantized weights back to layer
        self.layer.weight.data = W.to(self.layer.weight.dtype)

        total_loss = total_loss.sum().item()
        return total_loss

    def fasterquant_blockwise(self, blocksize=128, percdamp=0.01, log_condition=False):
        """Run block-wise GPTQ (Algorithm 3).

        Quantizes all columns in a block simultaneously, then uses the full
        BxB inverse-Hessian sub-block for error compensation. No sequential
        within-block error propagation.

        Supports microscaling formats (e.g. nvfp4) via the quantizer's
        `requires_per_block_params` flag. When True, find_params() is called
        per block so each block gets its own FP8 scale. For all other formats,
        find_params() is called once globally.

        Args:
            blocksize: Number of columns to process per block (B). For nvfp4,
                must be a multiple of the microscaling block size (16).
            percdamp: Damping factor as percentage of mean diagonal.
            log_condition: If True, log condition number of each BxB sub-block.

        Returns:
            Total squared quantization loss. If log_condition=True, returns
            (total_loss, condition_numbers_list).
        """
        W = self.layer.weight.data.clone().float()
        H = self.H

        # Formats with microscaling (e.g. nvfp4) need find_params per block
        # because each block gets its own scale. All other formats compute
        # global params once here.
        per_block_params = getattr(self.quantizer, "requires_per_block_params", False)
        if not per_block_params:
            self.quantizer.find_params(W)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += damp

        # Cholesky-based inverse Hessian (computed on CPU to avoid GPU Cholesky bug #136405)
        H_cpu = H.cpu()
        H_cpu = torch.linalg.cholesky(H_cpu)
        H_cpu = torch.cholesky_inverse(H_cpu)
        H_cpu = torch.linalg.cholesky(H_cpu, upper=True)
        Hinv = H_cpu.to(self.dev)
        del H_cpu

        total_loss = torch.zeros(self.rows, device=self.dev, dtype=torch.float32)
        condition_numbers = []

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            W_block = W[:, i1:i2].clone()
            Hinv_block = Hinv[i1:i2, i1:i2]

            if log_condition:
                cond = torch.linalg.cond(Hinv_block).item()
                condition_numbers.append(cond)

            # For microscaling formats: compute block scale from the current
            # (error-compensated) W_block, not the original weights.
            # This ensures the FP8 scale reflects the actual values being quantized
            # after prior-block error propagation has updated W.
            if per_block_params:
                self.quantizer.find_params(W_block)

            # Quantize entire block simultaneously (Algorithm 3, line 4)
            Q_block = self.quantizer.quantize_dequantize(W_block, col_start=i1)

            # Block error via triangular solve (Algorithm 3, line 5)
            # E * C_B = (W - Q), so C_B^T * E^T = (W - Q)^T
            # C_B is upper-triangular, C_B^T is lower-triangular
            raw_err = W_block - Q_block
            Err_block = torch.linalg.solve_triangular(
                Hinv_block.T, raw_err.T, upper=False
            ).T

            total_loss += (Err_block ** 2).sum(dim=1)

            W[:, i1:i2] = Q_block

            # Propagate block error to remaining columns (Algorithm 3, line 6)
            if i2 < self.columns:
                W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]

        self.layer.weight.data = W.to(self.layer.weight.dtype)

        total_loss = total_loss.sum().item()

        if log_condition:
            return total_loss, condition_numbers
        return total_loss

    def free(self):
        """Free Hessian memory."""
        self.H = None
        torch.cuda.empty_cache()
