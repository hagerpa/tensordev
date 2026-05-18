"""
fssk_setup.py — FSSK parameter generation and benchmark regimes.

  random_fssk(...)  — generate a random FSSK kernel via FSSK.from_jordan
                      with configurable Jordan block sizes and optional
                      dense materialisation (as_dense=True).
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
    eig_min: float = 0.1,
    eig_max: float = 1.0,
    freq_min: float = 0.1,
    freq_max: float = 2.0,
    min_block_size: int = 1,
    max_block_size: int = 4,
    normalise_b: bool = True,
    as_dense: bool = False,
    dtype=np.float64,
) -> FSSK:
    """Return a random FSSK kernel mixing real and oscillatory Jordan blocks.

    The state matrix Lambda is assembled by greedy packing of Jordan blocks into
    ``R`` state dimensions.  At each step a block type is chosen at random:

    - **Real block** (chain-order ``s``, uses ``s`` dims): purely real eigenvalue
      ``λ ~ U(eig_min, eig_max)``.
    - **Oscillatory block** (chain-order ``s``, uses ``2s`` dims): complex conjugate
      pair ``λ ± iω`` with ``λ ~ U(eig_min, eig_max)`` and
      ``ω ~ U(freq_min, freq_max)``.

    If both types can fit, each is chosen with probability ½.  If only real fits
    (i.e. ``remaining < 2 * min_block_size``), a real block is always used.
    Any leftover dims after packing become a single real padding block.

    The kernel is constructed via :meth:`FSSK.from_jordan`.  When
    ``as_dense=True`` the Jordan structure is materialised as a dense matrix
    and the kernel is returned with a :class:`DenseLambda` operator.

    Parameters
    ----------
    q : int
        Number of FSSK components.
    R : int
        Total state-space dimension.
    m : int
        Latent path dimension (``A`` has shape ``(q, m, d)``).
    d : int
        Input path dimension.
    seed : int
        NumPy RNG seed.  Default 0.
    eig_min, eig_max : float
        Uniform range for the real part (decay) of each eigenvalue.
        Default [0.1, 1.0].
    freq_min, freq_max : float
        Uniform range for the imaginary part (frequency) of oscillatory
        eigenvalues.  Must satisfy ``0 < freq_min <= freq_max``.
        Default [0.1, 2.0].
    min_block_size, max_block_size : int
        Minimum and maximum Jordan chain order.  Default 1 / 4.
    normalise_b : bool
        If True (default), rescale ``b`` to unit Frobenius norm.
    as_dense : bool
        If True, materialise the Jordan structure into a dense matrix and
        return an FSSK with a :class:`DenseLambda` operator.  Default False.
    dtype : numpy dtype
        Output dtype.  Default float64.
    """
    if min_block_size < 1:
        raise ValueError("min_block_size must be >= 1.")
    if max_block_size < min_block_size:
        raise ValueError("max_block_size must be >= min_block_size.")
    if freq_min <= 0:
        raise ValueError("freq_min must be strictly positive.")

    rng = np.random.default_rng(seed)

    # ---- pack Jordan blocks into R ------------------------------------------
    # For each block slot randomly choose real (s dims) or oscillatory (2s dims).
    # Fall back to real when the remaining budget can't fit an oscillatory block.
    real_block_sizes: list[int] = []
    osc_block_sizes:  list[int] = []
    remaining = R

    while remaining >= min_block_size:
        can_osc = remaining >= 2 * min_block_size

        if can_osc and bool(rng.integers(0, 2)):
            # Oscillatory block
            max_s = min(max_block_size, remaining // 2)
            s = int(rng.integers(min_block_size, max_s + 1))
            osc_block_sizes.append(s)
            remaining -= 2 * s
        else:
            # Real block
            max_s = min(max_block_size, remaining)
            s = int(rng.integers(min_block_size, max_s + 1))
            real_block_sizes.append(s)
            remaining -= s

    # Any leftover dims (< min_block_size) become one real padding block.
    if remaining > 0:
        real_block_sizes.append(remaining)

    real_sizes = tuple(real_block_sizes)
    osc_sizes  = tuple(osc_block_sizes)
    n_real     = len(real_block_sizes)
    n_osc      = len(osc_block_sizes)

    real_rates = rng.uniform(eig_min, eig_max, size=n_real)
    osc_decays = rng.uniform(eig_min, eig_max, size=n_osc)
    osc_freqs  = rng.uniform(freq_min, freq_max, size=n_osc)

    # ---- semi-orthogonal projection matrices A of shape (q, m, d) ----------
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
        bnorm = np.linalg.norm(b, ord="fro")
        if bnorm > 0:
            b /= bnorm
    else:
        # Column-wise L1 normalisation (each R-column has L1-norm 1 across q).
        b /= np.sum(np.abs(b), axis=0, keepdims=True)

    fssk = FSSK.from_jordan(
        A=jnp.asarray(A, dtype=dtype),
        b=jnp.asarray(b, dtype=dtype),
        real_rates=jnp.asarray(real_rates, dtype=jnp.float64),
        real_sizes=real_sizes,
        osc_decays=jnp.asarray(osc_decays, dtype=jnp.float64),
        osc_freqs=jnp.asarray(osc_freqs, dtype=jnp.float64),
        osc_sizes=osc_sizes,
    )

    if as_dense:
        return FSSK.from_matrix(
            Lambda=fssk.Lambda.matrix(),
            A=fssk.A,
            b=fssk.b,
        )
    return fssk


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
