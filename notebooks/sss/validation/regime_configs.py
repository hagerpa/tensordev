"""
regime_configs.py — Centralized benchmark regime configurations
================================================================

Each regime is defined by parameter ranges [low, high] (inclusive integers)
and a sample count.

Sampling strategy
-----------------
J, N, R are sampled **without replacement** from the full discrete grid.
Any (J, N, R) triple whose logarithmic workload cost

    cost(J, N, R) = log(J) + N·log(m_ref) + 2·log(R)

exceeds the regime reference cost

    cost_ref = log(J_ref) + N_ref·log(m_ref) + 2·log(R_ref)

is discarded and replaced by the next draw from the shuffled grid.

The accepted (J, N, R) samples are then **crossed with ALL values** of q, m, d
(and dyadic_order for Euler benchmarks), so no parameter is ever randomly
sampled — every value in its range appears in the design.

Three regimes:
- SMALL:  smoke-test
- MEDIUM: local dev
- LARGE:  production

Usage
-----
    from regime_configs import REGIMES

    cfg = REGIMES["MEDIUM"]
    df_exact = cfg.sample_exact()   # → run_timings.py
    df_euler = cfg.sample_euler()   # → validate_euler_convergence.py
    df_flop  = cfg.sample_flop()    # → sweep_flop_scaling.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product as iterproduct
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """Range-based benchmark regime.

    Call .sample_exact() / .sample_euler() / .sample_flop() to obtain a
    reproducible design DataFrame.

    Sampling
    --------
    * ``exact_n_samples`` / ``euler_n_samples`` / ``flop_n_samples`` are the
      number of distinct **(J, N, R)** triples to draw (before workload
      filtering; may be fewer if the valid grid is smaller than requested).
    * Each accepted (J, N, R) triple is cross-joined with **all** integer
      values in q_range, m_range, d_range (and dyadic_range for Euler).

    Workload filter
    ---------------
    A triple (J, N, R) is accepted iff

        log(J) + N · log(m_ref) + 2 · log(R)
            ≤ log(J_ref) + N_ref · log(m_ref) + 2 · log(R_ref)
    """

    name: str
    n_repeats: int
    seed: int           # master RNG seed; each sample method uses seed+offset

    # ── Parameter ranges [low, high] inclusive ────────────────────────────
    q_range:      tuple  # number of kernel components
    J_range:      tuple  # path nodes including t = 0
    R_range:      tuple  # state-space dimension
    N_range:      tuple  # signature truncation level
    dyadic_range: tuple  # Euler dyadic refinement order  (Euler only)
    m_range:      tuple  # latent-path dim
    d_range:      tuple  # input-path dim

    # ── Sample counts: number of distinct (J, N, R) triples ───────────────
    exact_n_samples: int
    euler_n_samples: int
    flop_n_samples:  int

    # ── Workload reference ────────────────────────────────────────────────
    J_ref: int   # reference J for workload cap
    N_ref: int   # reference N for workload cap
    R_ref: int   # reference R for workload cap
    m_ref: int   # reference m used in cost formula

    # ── Fixed settings ────────────────────────────────────────────────────
    n_paths: int = 2   # path-batch size for all benchmarks

    # ─────────────────────────────────────────────────────────────────────

    def _rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def _all_values(self, lo_hi: tuple) -> List[int]:
        """All integers in [lo_hi[0], lo_hi[1]] inclusive."""
        return list(range(lo_hi[0], lo_hi[1] + 1))

    # ── Workload helpers ──────────────────────────────────────────────────

    def _workload_cost(
        self,
        J: np.ndarray,
        N: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """Logarithmic workload proxy: log(J) + N·log(m_ref) + 2·log(R)."""
        return np.log(J) + N * np.log(self.m_ref) + 2.0 * np.log(R)

    def _reference_cost(self) -> float:
        """Workload cost at the reference point (J_ref, N_ref, R_ref)."""
        return float(
            np.log(self.J_ref)
            + self.N_ref * np.log(self.m_ref)
            + 2.0 * np.log(self.R_ref)
        )

    # ── Core (J, N, R) sampler ────────────────────────────────────────────

    def _sample_JNR(self, rng: np.random.Generator, n: int) -> pd.DataFrame:
        """Sample up to *n* distinct (J, N, R) triples without replacement.

        The full integer grid is enumerated, triples exceeding the workload
        reference are discarded, the remainder is shuffled and the first *n*
        are returned.  A warning is issued when fewer than *n* valid triples
        exist.

        Returns a DataFrame with columns J, N, R.
        """
        J_vals = np.arange(self.J_range[0], self.J_range[1] + 1, dtype=np.int64)
        N_vals = np.arange(self.N_range[0], self.N_range[1] + 1, dtype=np.int64)
        R_vals = np.arange(self.R_range[0], self.R_range[1] + 1, dtype=np.int64)

        J_grid, N_grid, R_grid = np.meshgrid(J_vals, N_vals, R_vals, indexing="ij")
        J_flat = J_grid.ravel()
        N_flat = N_grid.ravel()
        R_flat = R_grid.ravel()

        costs    = self._workload_cost(J_flat, N_flat, R_flat)
        ref_cost = self._reference_cost()
        mask     = costs <= ref_cost

        J_valid = J_flat[mask]
        N_valid = N_flat[mask]
        R_valid = R_flat[mask]

        n_avail = len(J_valid)
        if n_avail == 0:
            raise ValueError(
                f"[{self.name}] No (J, N, R) triple passes the workload filter "
                f"(ref_cost={ref_cost:.3f}).  "
                f"Consider raising J_ref / N_ref / R_ref."
            )

        n_take = min(n, n_avail)
        if n_take < n:
            warnings.warn(
                f"[{self.name}] Only {n_avail} valid (J, N, R) triples available; "
                f"requested {n}.  Using all {n_avail}.",
                stacklevel=3,
            )

        idx = rng.permutation(n_avail)[:n_take]
        return pd.DataFrame({
            "J": J_valid[idx],
            "N": N_valid[idx],
            "R": R_valid[idx],
        })

    # ── Cross-join helper ─────────────────────────────────────────────────

    def _cross_with(self, jnr: pd.DataFrame, extra_ranges: dict) -> pd.DataFrame:
        """Cross-join a (J, N, R) DataFrame with all combinations of the
        extra parameters given as ``{col_name: (lo, hi)}`` pairs."""
        if not extra_ranges:
            return jnr.copy()

        names  = list(extra_ranges.keys())
        values = [self._all_values(extra_ranges[k]) for k in names]
        combos = pd.DataFrame(list(iterproduct(*values)), columns=names)

        jnr             = jnr.copy()
        jnr["_key"]     = 1
        combos["_key"]  = 1
        df = jnr.merge(combos, on="_key").drop(columns="_key")
        return df.reset_index(drop=True)

    # ── Public design methods ─────────────────────────────────────────────

    def sample_exact(self) -> pd.DataFrame:
        """Design for run_timings.py.

        Draws ``exact_n_samples`` distinct (J, N, R) triples (workload-filtered,
        without replacement), then crosses with **all** values of q, m, d.

        Returns a DataFrame with columns:
            family, q, J, R, N, m, d, n_paths
        """
        rng = self._rng(0)
        jnr = self._sample_JNR(rng, self.exact_n_samples)
        df  = self._cross_with(jnr, {
            "q": self.q_range,
            "m": self.m_range,
            "d": self.d_range,
        })
        df["family"]  = np.where(df["q"] == 1, "q1", "qgt1")
        df["n_paths"] = self.n_paths
        return df

    def sample_euler(self) -> pd.DataFrame:
        """Design for validate_euler_convergence.py.

        Draws ``euler_n_samples`` distinct (J, N, R) triples (workload-filtered,
        without replacement), then crosses with **all** values of q, m, d,
        dyadic_order.

        Returns a DataFrame with columns:
            family, q, J, R, N, dyadic_order, m, d, n_paths
        """
        rng = self._rng(1)
        jnr = self._sample_JNR(rng, self.euler_n_samples)
        df  = self._cross_with(jnr, {
            "q":            self.q_range,
            "m":            self.m_range,
            "d":            self.d_range,
            "dyadic_order": self.dyadic_range,
        })
        df["family"]  = np.where(df["q"] == 1, "q1", "qgt1")
        df["n_paths"] = self.n_paths
        return df

    def sample_flop(self) -> pd.DataFrame:
        """Design for sweep_flop_scaling.py.

        Draws ``flop_n_samples`` distinct (J, N, R) triples (workload-filtered,
        without replacement), then crosses with **all** values of q, m, d.

        Returns a DataFrame with columns:
            family, q, J, R, N, m, d, n_paths
        """
        rng = self._rng(2)
        jnr = self._sample_JNR(rng, self.flop_n_samples)
        df  = self._cross_with(jnr, {
            "q": self.q_range,
            "m": self.m_range,
            "d": self.d_range,
        })
        df["family"]  = np.where(df["q"] == 1, "q1", "qgt1")
        df["n_paths"] = self.n_paths
        return df


# ---------------------------------------------------------------------------
# SMALL regime: smoke-test
# ---------------------------------------------------------------------------

SMALL = RegimeConfig(
    name="SMALL",
    n_repeats=3,
    seed=20260508,

    q_range      = (1, 3),
    J_range      = (32, 256),
    R_range      = (1, 6),
    N_range      = (3, 7),
    dyadic_range = (0, 4),
    m_range      = (2, 2),
    d_range      = (2, 2),

    exact_n_samples = 50,
    euler_n_samples = 50,
    flop_n_samples  = 100,

    # Workload reference: log(128) + 5·log(2) + 2·log(4) ≈ 11.09
    J_ref = 128,
    N_ref = 5,
    R_ref = 4,
    m_ref = 2,

    n_paths = 2,
)


# ---------------------------------------------------------------------------
# MEDIUM regime: local development
# ---------------------------------------------------------------------------

MEDIUM = RegimeConfig(
    name="MEDIUM",
    n_repeats=5,
    seed=20260508,

    q_range      = (1, 4),
    J_range      = (32, 1024),
    R_range      = (5, 12),
    N_range      = (5, 11),
    dyadic_range = (0, 5),
    m_range      = (3, 3),
    d_range      = (3, 3),

    exact_n_samples = 300,
    euler_n_samples = 300,
    flop_n_samples  = 300,

    # Workload reference: log(J_ref) + N_ref·log(m_ref) + 2·log(R_ref)
    J_ref = 512,
    N_ref = 11,
    R_ref = 3,
    m_ref = 3,

    n_paths = 2,
)


# ---------------------------------------------------------------------------
# LARGE regime: production benchmarks
# ---------------------------------------------------------------------------

LARGE = RegimeConfig(
    name="LARGE",
    n_repeats=10,
    seed=20260508,

    q_range      = (1, 8),
    J_range      = (32, 2048),
    R_range      = (1, 16),
    N_range      = (3, 16),
    dyadic_range = (0, 7),
    m_range      = (2, 5),
    d_range      = (2, 4),

    exact_n_samples = 1000,
    euler_n_samples = 1000,
    flop_n_samples  = 4000,

    # Workload reference: log(1024) + 10·log(4) + 2·log(10) ≈ 25.40
    J_ref = 1024,
    N_ref = 10,
    R_ref = 10,
    m_ref = 4,

    n_paths = 2,
)


# ---------------------------------------------------------------------------
# REGIMES dict
# ---------------------------------------------------------------------------

REGIMES: dict[str, RegimeConfig] = {
    "SMALL":  SMALL,
    "MEDIUM": MEDIUM,
    "LARGE":  LARGE,
}


# ---------------------------------------------------------------------------
# Helper: estimate (or compute) run counts
# ---------------------------------------------------------------------------

def estimate_run_counts(regime_name: str) -> dict:
    """Return actual post-sampling row counts by running the samplers."""
    cfg   = REGIMES[regime_name]
    exact = len(cfg.sample_exact())
    euler = len(cfg.sample_euler())
    flop  = len(cfg.sample_flop())
    return {
        "exact_total": exact,
        "euler_total": euler,
        "flop_total":  flop,
        "grand_total": exact + euler + flop,
    }


def _jnr_counts(regime_name: str) -> dict:
    """Return the number of accepted (J, N, R) triples per sampler."""
    cfg = REGIMES[regime_name]
    rng = cfg._rng(0)
    ne  = len(cfg._sample_JNR(rng, cfg.exact_n_samples))
    rng = cfg._rng(1)
    nu  = len(cfg._sample_JNR(rng, cfg.euler_n_samples))
    rng = cfg._rng(2)
    nf  = len(cfg._sample_JNR(rng, cfg.flop_n_samples))
    return {"exact_JNR": ne, "euler_JNR": nu, "flop_JNR": nf}


# ---------------------------------------------------------------------------
# __main__: print summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 70)
    print("Regime Configuration Summary")
    print("═" * 70)

    for regime_name in ["SMALL", "MEDIUM", "LARGE"]:
        cfg    = REGIMES[regime_name]
        counts = estimate_run_counts(regime_name)
        jnr    = _jnr_counts(regime_name)
        ref    = cfg._reference_cost()

        def _r(lo_hi):
            return f"[{lo_hi[0]}, {lo_hi[1]}]"

        print(f"\n{regime_name} Regime  (seed={cfg.seed}, n_repeats={cfg.n_repeats})")
        print("─" * 70)
        print(f"  Parameter ranges:")
        print(f"    q      : {_r(cfg.q_range)}  ({cfg.q_range[1]-cfg.q_range[0]+1} values, all used)")
        print(f"    J      : {_r(cfg.J_range)}  (sampled without replacement)")
        print(f"    R      : {_r(cfg.R_range)}  (sampled without replacement)")
        print(f"    N      : {_r(cfg.N_range)}  (sampled without replacement)")
        print(f"    dyadic : {_r(cfg.dyadic_range)}  ({cfg.dyadic_range[1]-cfg.dyadic_range[0]+1} values, all used — Euler only)")
        print(f"    m      : {_r(cfg.m_range)}  ({cfg.m_range[1]-cfg.m_range[0]+1} values, all used)")
        print(f"    d      : {_r(cfg.d_range)}  ({cfg.d_range[1]-cfg.d_range[0]+1} values, all used)")
        print()
        print(f"  Workload filter:  log(J) + N·log(m_ref) + 2·log(R)  ≤  {ref:.3f}")
        print(f"    reference: J_ref={cfg.J_ref}, N_ref={cfg.N_ref}, "
              f"R_ref={cfg.R_ref}, m_ref={cfg.m_ref}")
        print()
        print(f"  (J,N,R) triples accepted:")
        print(f"    Exact SSS  : {jnr['exact_JNR']:5d}  (requested {cfg.exact_n_samples})")
        print(f"    Euler      : {jnr['euler_JNR']:5d}  (requested {cfg.euler_n_samples})")
        print(f"    FLOP sweep : {jnr['flop_JNR']:5d}  (requested {cfg.flop_n_samples})")
        print()
        print(f"  Total rows after cross-product:")
        print(f"    Exact SSS  : {counts['exact_total']:8d}")
        print(f"    Euler      : {counts['euler_total']:8d}")
        print(f"    FLOP sweep : {counts['flop_total']:8d}")
        print(f"    GRAND TOTAL: {counts['grand_total']:8d}")

    print()
    print("═" * 70)

