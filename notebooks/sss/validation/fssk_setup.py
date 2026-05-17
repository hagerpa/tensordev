"""
fssk_setup.py — FSSK parameter generation and benchmark regimes.

  random_fssk(...)  — generate a random FSSK kernel via FSSK.from_jordan
  REGIMES           — dict of SMALL / MEDIUM / LARGE RegimeConfig instances
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # tensordev/src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

from tensordev.sss.kernel import FSSK
from _validation_util.regime_config import RegimeConfig


# ===========================================================================
# 1. Parameter generation
# ===========================================================================

def random_fssk(
    *,
    q: int,
    R: int,
    m: int,
    d: int,
    seed: int = 0,
    eig_min: float = 0.0,
    eig_max: float = 1.0,
    normalise_b: bool = True,
    dtype=np.float64,
    # Legacy parameters kept for call-site compatibility but no longer used.
    jordan_alpha: float = 0.25,   # noqa: ignored — JordanLambda always uses 1
    spectral_radius: float | None = None,  # noqa: ignored
    as_jordan: bool = False,      # noqa: ignored — output is always Jordan form
) -> FSSK:
    """Return a random FSSK kernel built from real 2×2 Jordan blocks.

    The state matrix Lambda is assembled as a block-diagonal of real Jordan
    blocks of size 2 (plus one size-1 block when R is odd).  Each block has a
    randomly drawn eigenvalue ``lam_i ~ Uniform(eig_min, eig_max)`` and the
    standard Jordan off-diagonal of 1 (as used by :class:`JordanLambda`).

    The kernel is constructed via :meth:`FSSK.from_jordan` so that the
    returned object carries a structured :class:`JordanLambda` operator.

    Parameters
    ----------
    q : int
        Number of FSSK components.
    R : int
        Total state-space dimension.  Decomposed into ``R // 2`` blocks of
        size 2 plus one block of size 1 when R is odd.
    m : int
        Latent path dimension (A has shape ``(q, m, d)``).
    d : int
        Input path dimension.
    seed : int
        NumPy RNG seed.  Default 0.
    eig_min, eig_max : float
        Uniform range for the block eigenvalues.  Default [0.0, 1.0].
    normalise_b : bool
        If True (default), rescale ``b`` to unit Frobenius norm (``‖b‖_F = 1``),
        independent of the eigenvalue scale.  This ensures the kernel has
        meaningful discriminative power regardless of how small the eigenvalues are.
    dtype : numpy dtype
        Output dtype.  Default float64.
    """
    rng = np.random.default_rng(seed)

    # Build R//2 Jordan blocks of size 2, plus one size-1 block if R is odd.
    n_pairs   = R // 2
    n_singles = R % 2
    n_blocks  = n_pairs + n_singles

    lam        = rng.uniform(eig_min, eig_max, size=n_blocks)
    real_rates = lam
    real_sizes = (2,) * n_pairs + (1,) * n_singles

    # Semi-orthogonal projection matrices A of shape (q, m, d).
    A = np.empty((q, m, d), dtype=dtype)
    for p in range(q):
        if m <= d:
            G = rng.normal(size=(d, m))
            Qp, _ = np.linalg.qr(G)
            A[p] = Qp.T
        else:
            G = rng.normal(size=(m, d))
            Qp, _ = np.linalg.qr(G)
            A[p] = Qp

    b = rng.normal(size=(q, R))

    if normalise_b:
        # Normalise b to unit Frobenius norm, independent of eigenvalue scale.
        bnorm = np.linalg.norm(b, ord="fro")
        if bnorm > 0:
            b /= bnorm
    else:
        # Column-wise L1 normalisation (each R-column has L1-norm 1 across q).
        b /= np.sum(np.abs(b), axis=0, keepdims=True)

    return FSSK.from_jordan(
        A=jnp.asarray(A, dtype=dtype),
        b=jnp.asarray(b, dtype=dtype),
        real_rates=jnp.asarray(real_rates, dtype=jnp.float64),
        real_sizes=real_sizes,
    )


# ===========================================================================
# 2. Benchmark regimes
# ===========================================================================

def _fssk_workload(m_ref: int):
    """FSSK workload: log(J) + N·log(m_ref) + 2·log(R)."""
    def fn(J: np.ndarray, N: np.ndarray, R: np.ndarray) -> np.ndarray:
        return np.log(J) + N * np.log(m_ref) + 2.0 * np.log(R)
    return fn


def _fssk_postprocess(df):
    df = df.copy()
    df["family"] = np.where(df["q"] == 1, "q1", "qgt1")
    return df


# SMALL: smoke-test
SMALL = RegimeConfig(
    name="SMALL",
    n_repeats=3,
    seed=20260508,
    sampled_ranges={"J": (32, 256),  "N": (3, 7),  "R": (1, 6)},
    crossed_ranges={"q": (1, 3),     "m": (2, 2),  "d": (2, 2)},
    ref_point={"J": 128, "N": 5, "R": 4},
    workload_fn=_fssk_workload(2),
    exact_n_samples=50,
    flop_n_samples=100,
    postprocess_fn=_fssk_postprocess,
)

# MEDIUM: local development
MEDIUM = RegimeConfig(
    name="MEDIUM",
    n_repeats=5,
    seed=20260508,
    sampled_ranges={"J": (32, 1024), "N": (5, 11), "R": (5, 12)},
    crossed_ranges={"q": (1, 4),     "m": (3, 3),  "d": (3, 3)},
    ref_point={"J": 512, "N": 11, "R": 3},
    workload_fn=_fssk_workload(3),
    exact_n_samples=300,
    flop_n_samples=300,
    postprocess_fn=_fssk_postprocess,
)

# LARGE: production benchmarks
LARGE = RegimeConfig(
    name="LARGE",
    n_repeats=10,
    seed=20260508,
    sampled_ranges={"J": (32, 2048), "N": (3, 16), "R": (1, 16)},
    crossed_ranges={"q": (1, 8),     "m": (2, 5),  "d": (2, 4)},
    ref_point={"J": 1024, "N": 10, "R": 10},
    workload_fn=_fssk_workload(4),
    exact_n_samples=1000,
    flop_n_samples=4000,
    postprocess_fn=_fssk_postprocess,
)

REGIMES: dict[str, RegimeConfig] = {
    "SMALL":  SMALL,
    "MEDIUM": MEDIUM,
    "LARGE":  LARGE,
}
