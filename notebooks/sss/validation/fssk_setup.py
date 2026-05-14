"""
fssk_setup.py — FSSK parameter generation and benchmark regimes.

  random_fssk(...)  — generate random (Lambda, A, b) kernel parameters
  REGIMES           — dict of SMALL / MEDIUM / LARGE RegimeConfig instances
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

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
    jordan_alpha: float = 0.25,
    spectral_radius: float | None = None,
    normalise_b: bool = True,
    as_jordan: bool = False,
    dtype=np.float64,
):
    """Return random (Lambda, A, b) parameters for an FSSK kernel.

    Lambda : (R, R)  by default a random matrix similar to a near-Jordan
             form: eigenvalues drawn uniformly from [eig_min, eig_max] with
             a super-diagonal of *jordan_alpha*, then conjugated by a random
             orthogonal Q so Lambda = Q J Q^T.  When *as_jordan* is True the
             Q step is skipped and Lambda is the near-diagonal J itself.

    A      : (q, m, d)  semi-orthogonal projection matrices.

    b      : (q, R)  random coefficients.  If *normalise_b* is True
             (default), b is rescaled so that ‖Λ‖_2 / ‖b‖_F = 1.

    Parameters
    ----------
    spectral_radius : float or None
        If given, rescale Lambda to this spectral radius before normalising b.
    normalise_b : bool
        Normalise b as described above.  Default True.
    as_jordan : bool
        If True, return Lambda in the near-diagonal Jordan basis (no Q
        conjugation).  Useful for benchmarks where a structured Lambda is
        preferred.  Default False.
    dtype : numpy dtype
        Output dtype for all three arrays.  Default float64.
    """
    rng = np.random.default_rng(seed)

    eigs = rng.uniform(eig_min, eig_max, size=R)
    J = np.diag(eigs)
    for i in range(R - 1):
        J[i, i + 1] = jordan_alpha

    if as_jordan:
        Lambda = J
    else:
        G = rng.normal(size=(R, R))
        Q, _ = np.linalg.qr(G)
        Lambda = Q @ J @ Q.T

    if spectral_radius is not None:
        sr = np.max(np.abs(np.linalg.eigvals(Lambda)))
        if sr > 0:
            Lambda *= spectral_radius / sr

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
        L = np.linalg.norm(Lambda, ord=2)
        bnorm = np.linalg.norm(b, ord="fro")
        if bnorm > 0:
            b *= L / bnorm
    else:
        b /= np.sum(np.abs(b), axis=0, keepdims=True)

    return Lambda.astype(dtype), A.astype(dtype), b.astype(dtype)


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
