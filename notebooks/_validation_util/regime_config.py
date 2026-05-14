"""
regime_config.py — Generic benchmark regime configuration.

RegimeConfig defines a parameter space, a workload-cost cap, and reproducible
sampling helpers that are reusable across validation sets.

Design
------
Parameters are split into two groups:

* ``sampled_ranges`` — the "expensive" axes whose combinations are filtered by
  a workload cap and sampled without replacement (e.g. J, N for path length
  and truncation level).

* ``crossed_ranges`` — the "cheap" axes whose values are exhaustively crossed
  with every accepted sample (e.g. q, m, d for number of components, latent
  dim, and input dim).

The workload formula is fully injected via ``workload_fn``, which receives the
sampled-parameter arrays as **keyword arguments** matching the keys of
``sampled_ranges``.  A triple is accepted iff its cost does not exceed the
cost at the reference point ``ref_point``.

Usage
-----
    from _validation_util.regime_config import RegimeConfig
    import numpy as np

    SMALL = RegimeConfig(
        name="SMALL",
        n_repeats=3,
        seed=42,
        sampled_ranges={"J": (32, 256), "N": (3, 7)},
        crossed_ranges={"q": (1, 3), "m": (2, 4)},
        ref_point={"J": 128, "N": 5},
        workload_fn=lambda J, N: np.log(J) + N * np.log(4),
        exact_n_samples=50,
        flop_n_samples=100,
    )
    df = SMALL.sample_exact()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product as iterproduct
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeConfig:
    """Range-based benchmark regime with a fully injectable parameter space.

    Parameters
    ----------
    name : str
    n_repeats : int
        Default number of hot timed calls.
    seed : int
        Master RNG seed; exact/flop methods use seed+0 and seed+2.
    sampled_ranges : dict[str, tuple[int, int]]
        Parameter axes subject to workload filtering.  Each key names a
        parameter; the value is a ``(lo, hi)`` inclusive integer range.
        These are the axes over which the design grid is built and filtered.
    crossed_ranges : dict[str, tuple[int, int]]
        Parameter axes exhaustively crossed with every accepted sample.
        Each key names a parameter; the value is a ``(lo, hi)`` inclusive
        integer range.
    ref_point : dict[str, int]
        Reference values for every key in ``sampled_ranges``.  A combination
        is accepted iff ``workload_fn(**combo) <= workload_fn(**ref_point)``.
    workload_fn : Callable[..., np.ndarray]
        Vectorised workload function.  Receives 1-D float64 arrays as keyword
        arguments whose names match the keys of ``sampled_ranges`` and returns
        a 1-D float64 cost array of the same length.
    exact_n_samples : int
        Number of sampled-range combinations to draw for exact-timing designs.
    flop_n_samples : int
        Number of sampled-range combinations to draw for FLOP-scaling designs.
    n_paths : int
        Path batch size added to every design row (default 2).
    postprocess_fn : Callable[[pd.DataFrame], pd.DataFrame] | None
        Optional transform applied to the design DataFrame after sampling and
        crossing.  Use this for experiment-specific derived columns (e.g.
        a ``family`` grouping column).
    """

    name: str
    n_repeats: int
    seed: int

    sampled_ranges: dict   # {param_name: (lo, hi)}
    crossed_ranges: dict   # {param_name: (lo, hi)}
    ref_point: dict        # {param_name: int}

    workload_fn: Callable[..., np.ndarray]

    exact_n_samples: int
    flop_n_samples: int

    n_paths: int = 2
    postprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    # -----------------------------------------------------------------------

    def _rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def _all_values(self, lo_hi: tuple) -> list:
        return list(range(lo_hi[0], lo_hi[1] + 1))

    def _reference_cost(self) -> float:
        ref_arrays = {
            k: np.array([self.ref_point[k]], dtype=np.float64)
            for k in self.sampled_ranges
        }
        return float(self.workload_fn(**ref_arrays)[0])

    def _sample_filtered(self, rng: np.random.Generator, n: int) -> pd.DataFrame:
        """Sample up to *n* combinations from ``sampled_ranges`` without replacement.

        The full integer grid over all sampled parameters is enumerated.
        Combinations whose workload exceeds the reference cost are discarded.
        The remainder is shuffled and the first *n* are returned.
        """
        param_names = list(self.sampled_ranges.keys())
        axes = [
            np.arange(self.sampled_ranges[k][0], self.sampled_ranges[k][1] + 1, dtype=np.int64)
            for k in param_names
        ]

        grids = np.meshgrid(*axes, indexing="ij")
        flat_int = {k: g.ravel() for k, g in zip(param_names, grids)}
        flat_f64 = {k: v.astype(np.float64) for k, v in flat_int.items()}

        costs = self.workload_fn(**flat_f64)
        mask  = costs <= self._reference_cost()

        filtered = {k: v[mask] for k, v in flat_int.items()}
        n_avail  = int(mask.sum())

        if n_avail == 0:
            raise ValueError(
                f"[{self.name}] No combination passes the workload filter "
                f"(ref_cost={self._reference_cost():.3f}). "
                f"Consider raising ref_point values."
            )
        n_take = min(n, n_avail)
        if n_take < n:
            warnings.warn(
                f"[{self.name}] Only {n_avail} valid combinations available; "
                f"requested {n}. Using all {n_avail}.",
                stacklevel=3,
            )

        idx = rng.permutation(n_avail)[:n_take]
        return pd.DataFrame({k: v[idx] for k, v in filtered.items()})

    def _cross_with(self, sampled: pd.DataFrame) -> pd.DataFrame:
        """Cross-join *sampled* with all combinations from ``crossed_ranges``."""
        if not self.crossed_ranges:
            return sampled.copy()

        names  = list(self.crossed_ranges.keys())
        values = [self._all_values(self.crossed_ranges[k]) for k in names]
        combos = pd.DataFrame(list(iterproduct(*values)), columns=names)

        sampled        = sampled.copy()
        sampled["_key"] = 1
        combos["_key"]  = 1
        return sampled.merge(combos, on="_key").drop(columns="_key").reset_index(drop=True)

    def _build_design(self, sampled: pd.DataFrame) -> pd.DataFrame:
        df = self._cross_with(sampled)
        df["n_paths"] = self.n_paths
        if self.postprocess_fn is not None:
            df = self.postprocess_fn(df)
        return df

    def sample_exact(self) -> pd.DataFrame:
        """Design for exact-timing benchmarks.

        Draws ``exact_n_samples`` combinations from ``sampled_ranges``
        (workload-filtered, without replacement), then crosses with all
        values from ``crossed_ranges``.
        """
        rng     = self._rng(0)
        sampled = self._sample_filtered(rng, self.exact_n_samples)
        return self._build_design(sampled)

    def sample_flop(self) -> pd.DataFrame:
        """Design for FLOP-scaling benchmarks.

        Draws ``flop_n_samples`` combinations from ``sampled_ranges``
        (workload-filtered, without replacement), then crosses with all
        values from ``crossed_ranges``.
        """
        rng     = self._rng(2)
        sampled = self._sample_filtered(rng, self.flop_n_samples)
        return self._build_design(sampled)
