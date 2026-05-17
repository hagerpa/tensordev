"""
sweep_timing.py — FSSK exact-signature timing benchmark sweep.

For each configuration in the regime design the script constructs a
StateSpaceSignature, times the hot vsig calls, and records wall-clock,
process-CPU, and OS-resource timings.

Output (written to --output-dir):
    fssk_exact_scaling_timings_{regime}.pkl
    fssk_exact_scaling_timings_{regime}.csv

Usage
-----
    python sweep_timing.py --regime MEDIUM
    python sweep_timing.py --regime LARGE --repeats 10
    python sweep_timing.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/
sys.path.insert(0, str(Path(__file__).resolve().parent))     # validation/

from tensordev.sss import StateSpaceSignature
from tensordev.util.random_paths import unit_speed_paths

from _validation_util.timing_utils import (
    time_call,
    time_warmup,
    time_hot_calls,
    aggregate_timing,
)
from _validation_util.benchmark_runner import (
    IDENTITY_FIELDS,
    TIMING_FIELDS,
    XLA_FIELDS,
    ERROR_FIELDS,
    make_empty_row,
    run_benchmark,
)

from fssk_setup import REGIMES, random_fssk


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

RESULT_SCHEMA: OrderedDict = OrderedDict([
    # ── run identity (generic) ────────────────────────────────────────────
    *IDENTITY_FIELDS.items(),
    # ── kernel / path hyperparameters (FSSK-specific) ─────────────────────
    ("q",          ("int",   "number of FSSK kernel components")),
    ("J",          ("int",   "number of path grid points (including t=0)")),
    ("intervals",  ("int",   "J - 1; number of path increments")),
    ("R",          ("int",   "state-space dimension")),
    ("m",          ("int",   "latent-path dimension")),
    ("d",          ("int",   "input-path dimension")),
    ("N",          ("int",   "signature truncation level")),
    ("n_paths",    ("int",   "batch size of paths fed to vsig")),
    ("n_repeats",  ("int",   "number of hot timed calls")),
    # ── error metrics (generic, placeholder) ─────────────────────────────
    *ERROR_FIELDS.items(),
    # ── wall-clock / CPU / OS-resource timings (generic) ─────────────────
    *TIMING_FIELDS.items(),
    # ── XLA cost and memory (generic) ────────────────────────────────────
    *XLA_FIELDS.items(),
    # ── theoretical cost proxies (FSSK-specific) ──────────────────────────
    ("q1_proxy",   ("float", "(J-1) · R² · m^N  (n=1 family)")),
    ("qgt1_proxy", ("float", "(J-1) · R² · N · m^N  (n>1 family)")),
])


def _empty_row() -> dict:
    return make_empty_row(RESULT_SCHEMA)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_exact_components(*, X, dt, fssk, N, n_repeats) -> dict:
    """Time StateSpaceSignature construction and vsig calls."""
    sss, rec_construct = time_call(
        StateSpaceSignature,
        kernel=fssk, trunc=N,
    )

    def call():
        return sss.vsig(X, dt=dt, axis=-2)

    _, rec_first = time_call(call)
    time_warmup(call, n_warmup=1)
    hot_records = time_hot_calls(call, n_calls=n_repeats)
    hot_agg = aggregate_timing(hot_records)

    return dict(
        wall_construct_s=rec_construct.wall_s,
        wall_first_call_s=rec_first.wall_s,
        **hot_agg,
        cpu_construct_s=rec_construct.cpu_s,
        cpu_first_call_s=rec_first.cpu_s,
        ru_construct_utime_s=rec_construct.ru_utime_s,
        ru_construct_stime_s=rec_construct.ru_stime_s,
        ru_first_call_utime_s=rec_first.ru_utime_s,
        ru_first_call_stime_s=rec_first.ru_stime_s,
    )


def _make_row_fn(regime_name: str, n_repeats: int, seed0: int, total_runs: int) -> Callable:
    def row_fn(run_id: int, cfg_row: dict) -> dict:
        q       = int(cfg_row["q"])
        J       = int(cfg_row["J"])
        R       = int(cfg_row["R"])
        N       = int(cfg_row["N"])
        m       = int(cfg_row["m"])
        n_paths = int(cfg_row["n_paths"])

        print(
            f"[{run_id + 1:03d}/{total_runs}]  "
            f"{cfg_row['family']:>4} | "
            f"q={q}  J={J}  R={R}  N={N}",
            flush=True,
        )

        dt = 1.0 / (J - 1)
        X = unit_speed_paths(
            dt=dt, dt_fine=dt / 8, n_paths=n_paths, dim=3,
            seed=seed0 + 10_000 + run_id,
        )
        fssk = random_fssk(
            q=q, R=R, m=m, d=3, seed=seed0 + run_id,
            eig_min=0.1, eig_max=1.5, normalise_b=False,
        )

        timings    = time_exact_components(X=X, dt=dt, fssk=fssk, N=N, n_repeats=n_repeats)
        intervals  = J - 1
        q1_proxy   = intervals * (R ** 2) * (m ** N)
        qgt1_proxy = q1_proxy * N

        row = _empty_row()
        row.update(
            run_id=run_id,
            regime=regime_name,
            method="fssk_exact_vsig",
            family=cfg_row["family"],
            design="random",
            q=q,
            J=J,
            intervals=intervals,
            R=R,
            m=m,
            d=3,
            N=N,
            n_paths=n_paths,
            n_repeats=n_repeats,
            **timings,
            q1_proxy=q1_proxy,
            qgt1_proxy=qgt1_proxy,
        )
        return row
    return row_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSSK exact-signature timing benchmark")
    p.add_argument("--regime", choices=["SMALL", "MEDIUM", "LARGE"], default="MEDIUM")
    p.add_argument("--repeats", type=int, default=None,
                   help="Override n_repeats from the regime config")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print design and exit without timing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = REGIMES[args.regime]
    n_repeats = args.repeats if args.repeats is not None else cfg.n_repeats

    df_design = cfg.sample_exact()

    print(f"Regime    : {args.regime}")
    print(f"n_repeats : {n_repeats}")
    print(f"Total runs: {len(df_design)}")
    print()
    print(df_design.groupby("family").size().rename("n_runs").to_string())
    print()

    regime_tag = args.regime.lower()
    row_fn = _make_row_fn(args.regime, n_repeats, cfg.seed, total_runs=len(df_design))

    df = run_benchmark(
        df_design,
        row_fn,
        output_dir=args.output_dir,
        output_stem=f"fssk_exact_scaling_timings_{regime_tag}",
        dry_run=args.dry_run,
        column_order=list(RESULT_SCHEMA.keys()),
    )

    if not args.dry_run:
        print(df[[
            "method", "family", "design", "q", "J", "R", "N",
            "wall_hot_median_s", "wall_hot_mean_s", "wall_hot_std_s",
        ]].to_string())


if __name__ == "__main__":
    main()