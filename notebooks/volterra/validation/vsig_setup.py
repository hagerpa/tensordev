"""
vsig_setup.py — vsig parameter spaces and benchmark regimes.

  REGIMES  — dict of SMALL / MEDIUM / LARGE RegimeConfig instances

Sampled axes (workload-filtered, uniform random): J, N
Crossed axes (exhaustive): q  (number of kernel components = kernel.q = A.shape[0])

Fixed constants across all regimes: d=3, m=3, beta=0.6, order=2, dyadic_order=0.

  d — input path dimension     = kernel.path_dim = A.shape[2]  (fixed at D=3)
  m — Volterra path dimension  = kernel.m        = A.shape[1]  (fixed at M=D=3)
  q — number of kernel components = kernel.q    = A.shape[0]  (swept 1..4)

Workload proxy (log-quadratic, for the more expensive quadratic scheme):

    W(J, N) = 2 · log(J) + N · log(m)

which is the log of the dominant quadratic-scheme cost J² · N^q · m^N.
The proxy filters (J, N) samples so the most expensive (J, N, q_max)
combination stays at or below a reference cost.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

from _validation_util.regime_config import RegimeConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D          = 3      # input path dimension   d = kernel.path_dim (fixed)
M          = D      # Volterra path dimension m = kernel.m        (fixed = D in this sweep)
BETA       = 0.6    # fractional exponent (fixed)
ORDER      = 2      # quadrature order (fixed)
DYADIC     = 0      # dyadic refinement (fixed)


# ---------------------------------------------------------------------------
# Workload proxy
# ---------------------------------------------------------------------------

def _vsig_workload(m: int):
    """Log-quadratic workload proxy: 2·log(J) + N·log(m).

    Approximates log of the dominant quadratic-scheme cost J² · N · m^N.
    Used to filter (J, N) samples so the most expensive config stays at or
    below the reference cost.
    """
    def fn(J: np.ndarray, N: np.ndarray) -> np.ndarray:
        return 2.0 * np.log(np.maximum(np.asarray(J, dtype=np.float64), 1.0)) \
               + np.asarray(N, dtype=np.float64) * np.log(m)
    return fn


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

# SMALL: smoke-test / fast CI
SMALL = RegimeConfig(
    name="SMALL",
    n_repeats=1,
    seed=20260601,
    sampled_ranges={"J": (32, 256),  "N": (2, 7)},
    crossed_ranges={"q": (1, 2)},
    ref_point={"J": 128, "N": 5},
    workload_fn=_vsig_workload(m=M),
    exact_n_samples=5,
    flop_n_samples=5,
)

# MEDIUM: local development / paper validation
MEDIUM = RegimeConfig(
    name="MEDIUM",
    n_repeats=1,
    seed=20260601,
    sampled_ranges={"J": (2**7, 2**10),  "N": (5, 12)},
    crossed_ranges={"q": (1, 4)},
    ref_point={"J": 512, "N": 9},
    workload_fn=_vsig_workload(m=M),
    exact_n_samples=200,
    flop_n_samples=200,
)

# LARGE: production benchmarks
LARGE = RegimeConfig(
    name="LARGE",
    n_repeats=1,
    seed=20260601,
    sampled_ranges={"J": (32, 1024), "N": (2, 11)},
    crossed_ranges={"q": (1, 4)},
    ref_point={"J": 512, "N": 9},
    workload_fn=_vsig_workload(m=M),
    exact_n_samples=500,
    flop_n_samples=1000,
)

REGIMES: dict[str, RegimeConfig] = {
    "SMALL":  SMALL,
    "MEDIUM": MEDIUM,
    "LARGE":  LARGE,
}
