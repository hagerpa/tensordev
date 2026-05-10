"""
run_timings.py — FSS-kernel exact scaling benchmark
====================================================

Timing uses MEDIAN over n_repeats hot calls, which is robust to scheduling noise.

Usage
-----
    python run_timings.py                  # defaults to MEDIUM regime
    python run_timings.py --regime MEDIUM
    python run_timings.py --regime LARGE
    python run_timings.py --regime MEDIUM --repeats 10
    python run_timings.py --dry-run        # print design, exit without timing

Result schema
-------------
Every row written to the output pickle / CSV conforms to RESULT_SCHEMA (see
section 3 below).  Fields that are not yet populated are stored as None so
that downstream analysis code can always rely on the full column set being
present.
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

import argparse
import itertools
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensordev.sss import StateSpaceSignature
from tensordev.util.random_paths import unit_speed_paths
from timing_utils import (
    time_call,
    time_warmup,
    time_hot_calls,
    aggregate_timing,
)
from regime_configs import REGIMES

# ---------------------------------------------------------------------------
# 2. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSS-kernel exact scaling benchmark")
    p.add_argument(
        "--regime",
        choices=["SMALL", "MEDIUM", "LARGE"],
        default="MEDIUM",
        help="Regime key controlling grid sizes (default: MEDIUM)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override n_repeats from the regime config",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "validation_outputs",
        help="Directory for .pkl / .csv output",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print design and estimated counts, then exit without running",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# 3. Result schema
# ---------------------------------------------------------------------------
# Each entry: field_name -> (type_hint_str, description)
# The ordering here defines the column order in the output DataFrame.
# Fields marked "placeholder" are always written as None until the
# corresponding measurement is implemented.

RESULT_SCHEMA: OrderedDict = OrderedDict([
    # ── run identity ─────────────────────────────────────────────────────
    ("run_id",                  ("int",         "sequential index within the current run")),
    ("regime",                  ("str",         "'SMALL', 'MEDIUM' or 'LARGE'")),
    ("method",                  ("str",         "algorithm tag, e.g. 'fssk_exact_vsig'")),
    ("family",                  ("str",         "'q1' or 'qgt1'")),
    ("design",                  ("str",         "'random'")),

    # ── kernel / path hyperparameters ────────────────────────────────────
    ("q",                       ("int",         "number of FSSK kernel components")),
    ("J",                       ("int",         "number of path grid points (including t=0)")),
    ("intervals",               ("int",         "J - 1; number of path increments")),
    ("R",                       ("int",         "state-space dimension")),
    ("m",                       ("int",         "latent-path dimension")),
    ("d",                       ("int",         "input-path dimension")),
    ("N",                       ("int",         "signature truncation level")),
    ("n_paths",                 ("int",         "batch size of paths fed to vsig")),
    ("n_repeats",               ("int",         "number of hot timed calls")),

    # ── error metrics (optional; None when not computed) ─────────────────
    ("error_metric_name",       ("str|None",    "placeholder: name of the error metric, e.g. 'max_abs'")),
    ("error_value",             ("float|None",  "placeholder: scalar error against a reference")),
    ("error_reference_tag",     ("str|None",    "placeholder: tag identifying the reference, e.g. 'euler_dyadic6'")),

    # ── wall-clock timings (seconds) ─────────────────────────────────────
    ("wall_construct_s",        ("float",       "wall time for StateSpaceSignature.from_matrix(...)")),
    ("wall_first_call_s",       ("float",       "wall time for the first vsig call (includes JAX JIT)")),
    ("wall_hot_median_s",       ("float",       "median wall time over n_repeats hot vsig calls")),
    ("wall_hot_mean_s",         ("float",       "mean wall time over n_repeats hot vsig calls")),
    ("wall_hot_std_s",          ("float",       "std dev of wall time over n_repeats hot vsig calls")),
    ("wall_hot_iqr_s",          ("float",       "IQR of wall time over n_repeats hot vsig calls")),
    ("wall_hot_min_s",          ("float",       "minimum wall time over n_repeats hot vsig calls")),
    ("wall_hot_max_s",          ("float",       "maximum wall time over n_repeats hot vsig calls")),

    # ── CPU-process timings (seconds) ────────────────────────────────────
    ("cpu_construct_s",         ("float|None",  "process CPU time for construction (process_time_ns)")),
    ("cpu_first_call_s",        ("float|None",  "process CPU time for first vsig call (process_time_ns)")),
    ("cpu_hot_median_s",        ("float|None",  "median process CPU time over hot calls")),
    ("cpu_hot_mean_s",          ("float|None",  "mean process CPU time over hot calls")),
    ("cpu_hot_std_s",           ("float|None",  "std dev of process CPU time over hot calls")),
    ("cpu_hot_iqr_s",           ("float|None",  "IQR of process CPU time over hot calls")),

    # ── OS-level resource counters (resource.getrusage; None on Windows) ─
    ("ru_construct_utime_s",    ("float|None",  "user CPU time for construction (getrusage)")),
    ("ru_construct_stime_s",    ("float|None",  "sys CPU time for construction (getrusage)")),
    ("ru_first_call_utime_s",   ("float|None",  "user CPU time for first vsig call (getrusage)")),
    ("ru_first_call_stime_s",   ("float|None",  "sys CPU time for first vsig call (getrusage)")),
    ("ru_hot_utime_median_s",   ("float|None",  "median user CPU time over hot calls (getrusage)")),
    ("ru_hot_utime_mean_s",     ("float|None",  "mean user CPU time over hot calls (getrusage)")),
    ("ru_hot_utime_std_s",      ("float|None",  "std dev of user CPU time over hot calls (getrusage)")),
    ("ru_hot_utime_iqr_s",      ("float|None",  "IQR of user CPU time over hot calls (getrusage)")),
    ("ru_hot_stime_median_s",   ("float|None",  "median sys CPU time over hot calls (getrusage)")),
    ("ru_hot_stime_mean_s",     ("float|None",  "mean sys CPU time over hot calls (getrusage)")),
    ("ru_hot_stime_std_s",      ("float|None",  "std dev of sys CPU time over hot calls (getrusage)")),
    ("ru_hot_stime_iqr_s",      ("float|None",  "IQR of sys CPU time over hot calls (getrusage)")),
    ("ru_hot_nvcsw_total",      ("int|None",    "total voluntary context switches over hot calls")),
    ("ru_hot_nivcsw_total",     ("int|None",    "total involuntary context switches over hot calls")),
    ("ru_hot_minflt_total",     ("int|None",    "total minor page faults over hot calls")),
    ("ru_hot_majflt_total",     ("int|None",    "total major page faults over hot calls")),

    # ── XLA timing ───────────────────────────────────────────────────────
    ("xla_lower_time_s",        ("float|None",  "wall seconds for .lower() — HLO emission")),
    ("xla_compile_time_s",      ("float|None",  "wall seconds for .compile() — XLA compilation")),

    # ── XLA cost-model scalars (from compiled.cost_analysis()) ───────────
    ("xla_flops",               ("float|None",  "total FLOPs")),
    ("xla_bytes_accessed",      ("float|None",  "total bytes read + written")),
    ("xla_transcendentals",     ("float|None",  "transcendental-op count")),

    # ── XLA memory scalars (from compiled.memory_analysis()) ─────────────
    ("xla_argument_bytes",      ("int|None",    "HBM for input buffers")),
    ("xla_output_bytes",        ("int|None",    "HBM for output buffers")),
    ("xla_temp_bytes",          ("int|None",    "HBM for temporaries / scratch")),
    ("xla_alias_bytes",         ("int|None",    "HBM for aliased buffers")),
    ("xla_peak_memory_bytes",   ("int|None",    "peak HBM across the computation")),
    ("xla_generated_code_bytes", ("int|None",   "compiled kernel binary size")),

    # ── raw XLA metadata ─────────────────────────────────────────────────
    ("xla_hlo_path",            ("str|None",    "path to the serialised HLO proto file")),
    ("xla_profile_path",        ("str|None",    "path to the XLA execution profile")),
    ("cost_analysis_raw",       ("str|None",    "JSON of compiled.cost_analysis() output")),
    ("memory_analysis_raw",     ("str|None",    "JSON of compiled.memory_analysis() fields")),

    # ── theoretical cost proxies ─────────────────────────────────────────
    ("q1_proxy",                ("float",       "J * R^2 * m^N  (theoretical cost for q=1 family)")),
    ("qgt1_proxy",              ("float",       "J * R^2 * N * m^N  (theoretical cost for q>1 family)")),
])


def _empty_row() -> dict:
    """Return a dict with every RESULT_SCHEMA field initialised to None."""
    return dict.fromkeys(RESULT_SCHEMA, None)


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------



def random_fssk_params(
    *, q: int, R: int, m: int, d: int, seed: int,
    eig_min: float = 0.1, eig_max: float = 1.5, jordan_alpha: float = 0.25,
    dtype=np.float64,
):
    rng = np.random.default_rng(seed)
    eigs = rng.uniform(eig_min, eig_max, size=R)
    Jmat = np.diag(eigs)
    for i in range(R - 1):
        Jmat[i, i + 1] = jordan_alpha
    G = rng.normal(size=(R, R))
    Q, _ = np.linalg.qr(G)
    Lambda = Q @ Jmat @ Q.T
    A = np.empty((q, m, d), dtype=dtype)
    for p in range(q):
        G = rng.normal(size=(d, m) if m <= d else (m, d))
        Qp, _ = np.linalg.qr(G)
        A[p] = Qp.T if m <= d else Qp
    b = rng.normal(size=(q, R))
    b /= np.sum(np.abs(b), axis=0, keepdims=True)
    return (
        jnp.asarray(Lambda.astype(dtype)),
        jnp.asarray(A.astype(dtype)),
        jnp.asarray(b.astype(dtype)),
    )


def time_exact_components(*, X, dt, Lambda, A, b, N, n_repeats) -> dict:
    """Time StateSpaceSignature construction and vsig calls.

    Uses :mod:`timing_utils` for all measurements:
    - Wall time          : ``time.perf_counter()``
    - Process CPU time   : ``time.process_time_ns()``
    - OS resource usage  : ``resource.getrusage(RUSAGE_SELF)``

    ``block_until_ready`` is called on every JAX output before clocks stop.

    Returns a dict whose keys match RESULT_SCHEMA fields directly.
    """
    # ── Construction ─────────────────────────────────────────────────────
    sss, rec_construct = time_call(
        StateSpaceSignature.from_matrix,
        Lambda=Lambda, A=A, b=b, trunc=N,
    )

    def call():
        return sss.vsig(X, dt=dt, axis=-2)

    # ── First call (includes JAX JIT compilation) ─────────────────────────
    _, rec_first = time_call(call)

    # ── Warmup: one extra call to evict cold-cache effects, not recorded ──
    time_warmup(call, n_warmup=1)

    # ── Hot calls ────────────────────────────────────────────────────────
    hot_records = time_hot_calls(call, n_calls=n_repeats)
    hot_agg = aggregate_timing(hot_records)

    return dict(
        # wall-clock
        wall_construct_s=rec_construct.wall_s,
        wall_first_call_s=rec_first.wall_s,
        **hot_agg,          # wall_hot_{median,mean,std,min,max}_s
        # process CPU
        cpu_construct_s=rec_construct.cpu_s,
        cpu_first_call_s=rec_first.cpu_s,
        # cpu_hot_{median,mean,std}_s already in hot_agg
        # getrusage — construction
        ru_construct_utime_s=rec_construct.ru_utime_s,
        ru_construct_stime_s=rec_construct.ru_stime_s,
        # getrusage — first call
        ru_first_call_utime_s=rec_first.ru_utime_s,
        ru_first_call_stime_s=rec_first.ru_stime_s,
        # ru_hot_* already in hot_agg
    )

# ---------------------------------------------------------------------------
# 5. Design matrix
# ---------------------------------------------------------------------------

# build_design is replaced by cfg.sample_exact() from regime_configs.

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = REGIMES[args.regime]
    n_repeats = args.repeats if args.repeats is not None else cfg.n_repeats

    seed0 = cfg.seed

    df_design = cfg.sample_exact()

    print(f"Regime          : {args.regime}")
    print(f"n_repeats       : {n_repeats}")
    print(f"Total runs      : {len(df_design)}")
    print()
    print(df_design.groupby("family").size().rename("n_runs").to_string())
    print()

    if args.dry_run:
        print("Dry-run mode — exiting without running timings.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime_tag = args.regime.lower()
    out_pkl = args.output_dir / f"fssk_exact_scaling_timings_{regime_tag}.pkl"
    out_csv = args.output_dir / f"fssk_exact_scaling_timings_{regime_tag}.csv"

    rows = []

    for run_id, design_row in df_design.iterrows():
        cfg_row = design_row.to_dict()
        q, J, R, N = int(cfg_row["q"]), int(cfg_row["J"]), int(cfg_row["R"]), int(cfg_row["N"])
        m       = int(cfg_row["m"])
        d       = int(cfg_row["d"])
        n_paths = int(cfg_row["n_paths"])

        print(
            f"[{run_id + 1:03d}/{len(df_design)}]  "
            f"{cfg_row['family']:>4} | "
            f"q={q}  J={J}  R={R}  N={N}",
            flush=True,
        )

        dt = 1.0 / (J - 1)
        X = unit_speed_paths(dt=dt, dt_fine=dt / 8, n_paths=n_paths, dim=3, seed=seed0 + 10_000 + run_id)
        Lambda, A, b = random_fssk_params(q=q, R=R, m=m, d=3, seed=seed0 + run_id)

        timings = time_exact_components(
            X=X, dt=dt, Lambda=Lambda, A=A, b=b, N=N, n_repeats=n_repeats,
        )

        intervals = J - 1
        q1_proxy   = intervals * (R ** 2) * (m ** N)
        qgt1_proxy = q1_proxy * N

        # Start from a fully-None skeleton so every schema field is always present.
        row = _empty_row()
        row.update(
            # ── identity ────────────────────────────────────────────────
            run_id=run_id,
            regime=args.regime,
            method="fssk_exact_vsig",
            family=cfg_row["family"],
            design="random",
            # ── hyperparameters ─────────────────────────────────────────
            q=q,
            J=J,
            intervals=intervals,
            R=R,
            m=m,
            d=3,
            N=N,
            n_paths=n_paths,
            n_repeats=n_repeats,
            # ── wall-clock timings (from time_exact_components) ─────────
            **timings,
            # ── theoretical cost proxies ────────────────────────────────
            q1_proxy=q1_proxy,
            qgt1_proxy=qgt1_proxy,
            # error metrics, CPU times, XLA fields remain None (placeholders)
        )
        rows.append(row)

        # checkpoint every 25 runs
        if (run_id + 1) % 25 == 0:
            pd.DataFrame(rows).to_pickle(out_pkl)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  checkpoint → {out_pkl}", flush=True)

    df_timings = pd.DataFrame(rows)
    # Enforce canonical column order from RESULT_SCHEMA.
    df_timings = df_timings[list(RESULT_SCHEMA.keys())]
    df_timings.to_pickle(out_pkl)
    df_timings.to_csv(out_csv, index=False)

    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")
    print(df_timings[[
        "method", "family", "design", "q", "J", "R", "N",
        "wall_hot_median_s", "wall_hot_mean_s", "wall_hot_std_s",
    ]].to_string())


if __name__ == "__main__":
    main()
