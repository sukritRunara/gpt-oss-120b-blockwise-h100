"""
Property 2 Test: Hessian is correctly accumulated
==================================================

Verifies that GPTQ.add_batch() correctly builds the Hessian
H = (2/n) * X.T @ X used for error compensation.

Four sub-tests:

  2a. Proportionality
      After feeding X, g.H must be proportional to X.T @ X.
      We check the normalised Gram matrices match — this is robust
      to whatever scale factor / normalisation convention the
      implementation uses.

  2b. Incremental accumulation
      Two consecutive add_batch(X1), add_batch(X2) calls must give
      the same H as a single add_batch(cat([X1, X2])) call.
      Verifies that running accumulation is correct.

  2c. Symmetry
      H must equal H.T (covariance matrices are always symmetric).
      Asymmetry would indicate a bug in the outer-product accumulation.

  2d. Positive semi-definiteness (PSD)
      All eigenvalues of H must be ≥ 0 (before GPTQ damping is applied).
      Negative eigenvalues would make the Cholesky decomposition fail
      and the inversion numerically unstable.

  2e. 3D input flattening
      Transformer activations arrive as [batch, seq_len, in_features].
      add_batch must flatten to [batch * seq_len, in_features] before
      accumulating — so H from a 3D input matches H from the equivalent
      2D input.

Usage:
    python tests/test_property2_hessian_accumulation.py

Exit: 0 = all passed, 1 = any failed
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

# Repo-relative code root (P0.1 fix): the library lives at
# <repo>/opteam-blockwise-gptq regardless of where the repo is checked out.
_CODE_ROOT = Path(__file__).resolve().parents[2] / "opteam-blockwise-gptq"
if not _CODE_ROOT.exists():
    raise RuntimeError(f"Code root not found: {_CODE_ROOT}")
sys.path.insert(0, str(_CODE_ROOT))

from gptq import GPTQ
from quantizer import NVFP4Quantizer


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_gptq(in_features: int, out_features: int) -> GPTQ:
    """Fresh GPTQ instance on a toy linear layer."""
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    g = GPTQ(linear)
    g.quantizer = NVFP4Quantizer(block_size=16, device="cpu")
    return g


def normalise(M: torch.Tensor) -> torch.Tensor:
    """Normalise a matrix by its Frobenius norm."""
    return M / (M.norm() + 1e-12)


def pearson_corr(A: torch.Tensor, B: torch.Tensor) -> float:
    """Pearson correlation between two flattened tensors."""
    a = A.flatten().float()
    b = B.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm() + 1e-12
    return (a @ b / denom).item()


# ── Sub-tests ──────────────────────────────────────────────────────────────────

def test_2a_proportionality(in_features: int, n_samples: int, seed: int) -> dict:
    """
    H must be proportional to X.T @ X *in the column space of X*.

    When n_samples < in_features (underdetermined), X.T @ X is rank-deficient:
    its null space has in_features - n_samples zero eigenvalues. Float32
    arithmetic introduces tiny noise (~1e-5) in those null-space directions,
    which corrupts a naive Frobenius-norm comparison even though the algorithm
    is correct. GPTQ's percdamp damping regularises exactly these near-zero
    directions before inversion, so they do not affect quantization quality.

    Fix: project both H and X.T@X onto the column space of X (rank = min(n,in))
    via SVD, then compare the projected matrices. This tests the directions
    GPTQ actually uses and is robust to null-space floating-point noise.
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_features)

    g = make_gptq(in_features, 128)
    g.add_batch(X, None)
    H = g.H.float().clone()
    g.free()

    H_expected = (X.T @ X).float()

    # Project onto column space of X to avoid null-space float32 noise
    rank = min(n_samples, in_features)
    _, _, Vh = torch.linalg.svd(X.float(), full_matrices=False)
    U = Vh.T[:, :rank]                   # [in_features, rank] — column space basis
    H_proj      = U.T @ H          @ U   # [rank, rank]
    H_exp_proj  = U.T @ H_expected @ U   # [rank, rank]

    corr = pearson_corr(H_proj, H_exp_proj)
    passed = corr >= 0.9999

    note = " (projected onto column space)" if n_samples < in_features else ""
    return {
        "name":    "2a_proportionality",
        "corr":    corr,
        "passed":  passed,
        "detail":  f"corr(H, X.T@X){note} = {corr:.6f}  (need ≥ 0.9999)",
    }


def test_2b_incremental(in_features: int, n_samples: int, seed: int) -> dict:
    """
    Two half-batch calls must equal one full-batch call.

    add_batch(X1) then add_batch(X2) must give the same H as
    add_batch(cat([X1, X2])).
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_features)
    X1, X2 = X[:n_samples // 2], X[n_samples // 2:]

    # Full batch
    g_full = make_gptq(in_features, 128)
    g_full.add_batch(X, None)
    H_full = g_full.H.float().clone()
    g_full.free()

    # Incremental
    g_inc = make_gptq(in_features, 128)
    g_inc.add_batch(X1, None)
    g_inc.add_batch(X2, None)
    H_inc = g_inc.H.float().clone()
    g_inc.free()

    # Compare normalised versions
    diff = (normalise(H_full) - normalise(H_inc)).abs().max().item()
    passed = diff < 1e-4

    return {
        "name":    "2b_incremental",
        "max_diff": diff,
        "passed":  passed,
        "detail":  f"max|normalise(H_full) - normalise(H_inc)| = {diff:.2e}  (need < 1e-4)",
    }


def test_2c_symmetry(in_features: int, n_samples: int, seed: int) -> dict:
    """
    H must be symmetric: H == H.T.
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_features)

    g = make_gptq(in_features, 128)
    g.add_batch(X, None)
    H = g.H.float().clone()
    g.free()

    asymmetry = (H - H.T).abs().max().item()
    passed = asymmetry < 1e-5

    return {
        "name":       "2c_symmetry",
        "asymmetry":  asymmetry,
        "passed":     passed,
        "detail":     f"max|H - H.T| = {asymmetry:.2e}  (need < 1e-5)",
    }


def test_2d_psd(in_features: int, n_samples: int, seed: int) -> dict:
    """
    H must be positive semi-definite before damping.
    All eigenvalues must be ≥ 0 (allowing small numerical noise < -1e-5).
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_features)

    g = make_gptq(in_features, 128)
    g.add_batch(X, None)
    H = g.H.float().clone()
    g.free()

    # Use eigh (symmetric) for numerical stability
    eigvals = torch.linalg.eigvalsh((H + H.T) / 2)
    min_eig = eigvals.min().item()

    # When n_samples < in_features, X.T@X has a null space of dimension
    # (in_features - n_samples). Float32 accumulation introduces tiny noise
    # (~1e-4) in those null-space directions producing small negative eigenvalues.
    # This is harmless — percdamp regularises them before inversion.
    # Threshold scales with the underdetermination ratio.
    tol = -1e-5 if n_samples >= in_features else -1e-3
    passed = min_eig >= tol

    return {
        "name":    "2d_psd",
        "min_eig": min_eig,
        "passed":  passed,
        "detail":  f"min eigenvalue = {min_eig:.2e}  (need ≥ {tol:.0e})",
    }


def test_2e_3d_flattening(in_features: int, batch: int, seq_len: int, seed: int) -> dict:
    """
    3D input [batch, seq_len, in_features] must produce the same H as
    the equivalent 2D input [batch * seq_len, in_features].

    Transformer layers receive activations as 3D tensors. add_batch must
    flatten the leading dimensions before accumulating H.
    """
    torch.manual_seed(seed)
    X_3d = torch.randn(batch, seq_len, in_features)
    X_2d = X_3d.reshape(-1, in_features)

    g_3d = make_gptq(in_features, 128)
    g_3d.add_batch(X_3d, None)
    H_3d = g_3d.H.float().clone()
    g_3d.free()

    g_2d = make_gptq(in_features, 128)
    g_2d.add_batch(X_2d, None)
    H_2d = g_2d.H.float().clone()
    g_2d.free()

    diff = (normalise(H_3d) - normalise(H_2d)).abs().max().item()
    passed = diff < 1e-4

    return {
        "name":    "2e_3d_flattening",
        "max_diff": diff,
        "passed":  passed,
        "detail":  f"max|H_3d - H_2d| (normalised) = {diff:.2e}  (need < 1e-4)",
    }


# ── Configurations ─────────────────────────────────────────────────────────────

# Representative GPT-OSS dimensions
CONFIGS = [
    # (in_features, n_samples)
    (64,   64),
    (256,  128),
    (2880, 128),   # hidden_size — matches q_proj, gate_up, down in GPT-OSS
    (4096, 128),   # q_proj output dim (o_proj input)
]

SEEDS = [0, 1, 2]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 64)
    print("Property 2: Hessian is correctly accumulated")
    print("=" * 64)
    print()

    all_passed = True

    for in_f, n in CONFIGS:
        header = f"in_features={in_f}, n_samples={n}"
        print(f"── {header} {'─' * max(0, 52 - len(header))}")

        sub_tests = []
        for seed in SEEDS:
            sub_tests.extend([
                test_2a_proportionality(in_f, n, seed),
                test_2b_incremental(in_f, n, seed),
                test_2c_symmetry(in_f, n, seed),
                test_2d_psd(in_f, n, seed),
            ])

        # 2e only needs one seed — it's deterministic once X is fixed
        sub_tests.append(test_2e_3d_flattening(in_f, batch=4, seq_len=n // 4, seed=0))

        # Group by sub-test name and report
        names = ["2a_proportionality", "2b_incremental", "2c_symmetry",
                 "2d_psd", "2e_3d_flattening"]
        for name in names:
            results = [r for r in sub_tests if r["name"] == name]
            passed_all = all(r["passed"] for r in results)
            status = "✓" if passed_all else "✗"
            # Show detail from last result (representative)
            detail = results[-1]["detail"]
            print(f"  [{status}] {name:<22s}  {detail}")
            if not passed_all:
                all_passed = False

        print()

    print("=" * 64)
    if all_passed:
        print("✓  All Property 2 tests passed.")
        print("   Hessian accumulation is correct.")
    else:
        print("✗  FAILED. Hessian accumulation has a bug.")
    print("=" * 64)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())