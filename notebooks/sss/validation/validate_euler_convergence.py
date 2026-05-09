"""
validate_euler_convergence.py — Euler convergence validation with comprehensive timing
=======================================================================================

Validates that Euler scheme converges to exact SSS as dyadic_order increases.
Records error metrics, wall-time/CPU-time statistics, XLA costs, and diagnostic counters.

Usage
-----
    python validate_euler_convergence.py --regime MEDIUM
    python validate_euler_convergence.py --regime LARGE
    python validate_euler_convergence.py --dry-run
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

import argparse
import itertools
import sys
from collections import OrderedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensordev.sss import StateSpaceSignature
from euler_general import fssk_euler_vsig

from timing_utils import (
    time_call,
    time_warmup,
    time_hot_calls,
    aggregate_timing,
)
from xla_utils import compile_and_profile, abstract_like
from regime_configs import REGIMES

# ---------------------------------------------------------------------------
# 2. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Euler convergence validation benchmark")
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

RESULT_SCHEMA: OrderedDict = OrderedDict([
    # ── run identity ─────────────────────────────────────────────────────
    ("run_id",                  ("int",         "sequential index within the current run")),
    ("regime",                  ("str",         "'SMALL', 'MEDIUM' or 'LARGE'")),
    ("method",                  ("str",         "'euler_validation'")),
    ("family",                  ("str",         "'q1' or 'qgt1'")),

    # ── kernel / path hyperparameters ────────────────────────────────────
    ("q",                       ("int",         "number of FSSK kernel components")),
    ("J",                       ("int",         "number of path grid points (including t=0)")),
    ("intervals",               ("int",         "J - 1; number of path increments")),
    ("R",                       ("int",         "state-space dimension")),
    ("m",                       ("int",         "latent-path dimension")),
    ("d",                       ("int",         "input-path dimension")),
    ("N",                       ("int",         "signature truncation level")),
    ("n_paths",                 ("int",         "batch size of paths")),
    ("dyadic_order",            ("int",         "Euler dyadic refinement level")),
    ("n_repeats",               ("int",         "number of hot timed calls")),

    # ── error metrics ────────────────────────────────────────────────────
    ("error_max_abs_global",    ("float",       "max absolute error across all levels and paths")),
    ("error_mean_abs_global",   ("float",       "mean absolute error across all levels and paths")),
    ("error_max_rel_global",    ("float",       "max relative error across all levels and paths")),
    ("error_mean_rel_global",   ("float",       "mean relative error across all levels and paths")),
    ("error_max_abs_by_level",  ("str",         "JSON: per-level max absolute errors")),
    ("error_mean_abs_by_level", ("str",         "JSON: per-level mean absolute errors")),

    # ── exact SSS timings ────────────────────────────────────────────────
    ("exact_wall_construct_s",      ("float",       "wall time for exact SSS construction")),
    ("exact_wall_first_call_s",     ("float",       "wall time for exact SSS first call (JIT)")),
    ("exact_wall_hot_median_s",     ("float",       "median wall time over hot calls")),
    ("exact_wall_hot_mean_s",       ("float",       "mean wall time over hot calls")),
    ("exact_wall_hot_std_s",        ("float",       "std wall time over hot calls")),
    ("exact_wall_hot_iqr_s",        ("float",       "IQR wall time over hot calls")),
    ("exact_wall_hot_min_s",        ("float",       "min wall time over hot calls")),
    ("exact_wall_hot_max_s",        ("float",       "max wall time over hot calls")),
    ("exact_cpu_construct_s",       ("float|None",  "CPU time for exact SSS construction")),
    ("exact_cpu_first_call_s",      ("float|None",  "CPU time for exact SSS first call")),
    ("exact_cpu_hot_median_s",      ("float|None",  "median CPU time over hot calls")),
    ("exact_cpu_hot_mean_s",        ("float|None",  "mean CPU time over hot calls")),
    ("exact_cpu_hot_std_s",         ("float|None",  "std CPU time over hot calls")),
    ("exact_cpu_hot_iqr_s",         ("float|None",  "IQR CPU time over hot calls")),
    ("exact_ru_construct_utime_s",  ("float|None",  "user CPU time for construction")),
    ("exact_ru_construct_stime_s",  ("float|None",  "system CPU time for construction")),
    ("exact_ru_first_call_utime_s", ("float|None",  "user CPU time for first call")),
    ("exact_ru_first_call_stime_s", ("float|None",  "system CPU time for first call")),
    ("exact_ru_hot_utime_median_s", ("float|None",  "median user CPU time over hot calls")),
    ("exact_ru_hot_utime_mean_s",   ("float|None",  "mean user CPU time over hot calls")),
    ("exact_ru_hot_utime_std_s",    ("float|None",  "std user CPU time over hot calls")),
    ("exact_ru_hot_utime_iqr_s",    ("float|None",  "IQR user CPU time over hot calls")),
    ("exact_ru_hot_stime_median_s", ("float|None",  "median system CPU time over hot calls")),
    ("exact_ru_hot_stime_mean_s",   ("float|None",  "mean system CPU time over hot calls")),
    ("exact_ru_hot_stime_std_s",    ("float|None",  "std system CPU time over hot calls")),
    ("exact_ru_hot_stime_iqr_s",    ("float|None",  "IQR system CPU time over hot calls")),
    ("exact_ru_hot_nvcsw_total",    ("int|None",    "total voluntary context switches")),
    ("exact_ru_hot_nivcsw_total",   ("int|None",    "total involuntary context switches")),
    ("exact_ru_hot_minflt_total",   ("int|None",    "total minor page faults")),
    ("exact_ru_hot_majflt_total",   ("int|None",    "total major page faults")),

    # ── Euler timings ────────────────────────────────────────────────────
    ("euler_wall_first_call_s",     ("float",       "wall time for Euler first call (JIT)")),
    ("euler_wall_hot_median_s",     ("float",       "median wall time over hot calls")),
    ("euler_wall_hot_mean_s",       ("float",       "mean wall time over hot calls")),
    ("euler_wall_hot_std_s",        ("float",       "std wall time over hot calls")),
    ("euler_wall_hot_iqr_s",        ("float",       "IQR wall time over hot calls")),
    ("euler_wall_hot_min_s",        ("float",       "min wall time over hot calls")),
    ("euler_wall_hot_max_s",        ("float",       "max wall time over hot calls")),
    ("euler_cpu_first_call_s",      ("float|None",  "CPU time for Euler first call")),
    ("euler_cpu_hot_median_s",      ("float|None",  "median CPU time over hot calls")),
    ("euler_cpu_hot_mean_s",        ("float|None",  "mean CPU time over hot calls")),
    ("euler_cpu_hot_std_s",         ("float|None",  "std CPU time over hot calls")),
    ("euler_cpu_hot_iqr_s",         ("float|None",  "IQR CPU time over hot calls")),
    ("euler_ru_first_call_utime_s", ("float|None",  "user CPU time for first call")),
    ("euler_ru_first_call_stime_s", ("float|None",  "system CPU time for first call")),
    ("euler_ru_hot_utime_median_s", ("float|None",  "median user CPU time over hot calls")),
    ("euler_ru_hot_utime_mean_s",   ("float|None",  "mean user CPU time over hot calls")),
    ("euler_ru_hot_utime_std_s",    ("float|None",  "std user CPU time over hot calls")),
    ("euler_ru_hot_utime_iqr_s",    ("float|None",  "IQR user CPU time over hot calls")),
    ("euler_ru_hot_stime_median_s", ("float|None",  "median system CPU time over hot calls")),
    ("euler_ru_hot_stime_mean_s",   ("float|None",  "mean system CPU time over hot calls")),
    ("euler_ru_hot_stime_std_s",    ("float|None",  "std system CPU time over hot calls")),
    ("euler_ru_hot_stime_iqr_s",    ("float|None",  "IQR system CPU time over hot calls")),
    ("euler_ru_hot_nvcsw_total",    ("int|None",    "total voluntary context switches")),
    ("euler_ru_hot_nivcsw_total",   ("int|None",    "total involuntary context switches")),
    ("euler_ru_hot_minflt_total",   ("int|None",    "total minor page faults")),
    ("euler_ru_hot_majflt_total",   ("int|None",    "total major page faults")),

    # ── XLA profiling (exact SSS) ────────────────────────────────────────
    ("exact_xla_lower_time_s",      ("float|None",  "XLA lowering time for exact SSS")),
    ("exact_xla_compile_time_s",    ("float|None",  "XLA compilation time for exact SSS")),
    ("exact_xla_flops",             ("float|None",  "total FLOPs for exact SSS")),
    ("exact_xla_bytes_accessed",    ("float|None",  "total bytes accessed for exact SSS")),
    ("exact_xla_transcendentals",   ("float|None",  "transcendental ops for exact SSS")),
    ("exact_xla_argument_bytes",    ("int|None",    "input buffer bytes for exact SSS")),
    ("exact_xla_output_bytes",      ("int|None",    "output buffer bytes for exact SSS")),
    ("exact_xla_temp_bytes",        ("int|None",    "temp buffer bytes for exact SSS")),
    ("exact_xla_peak_memory_bytes", ("int|None",    "peak memory for exact SSS")),

    # ── XLA profiling (Euler) ────────────────────────────────────────────
    ("euler_xla_lower_time_s",      ("float|None",  "XLA lowering time for Euler")),
    ("euler_xla_compile_time_s",    ("float|None",  "XLA compilation time for Euler")),
    ("euler_xla_flops",             ("float|None",  "total FLOPs for Euler")),
    ("euler_xla_bytes_accessed",    ("float|None",  "total bytes accessed for Euler")),
    ("euler_xla_transcendentals",   ("float|None",  "transcendental ops for Euler")),
    ("euler_xla_argument_bytes",    ("int|None",    "input buffer bytes for Euler")),
    ("euler_xla_output_bytes",      ("int|None",    "output buffer bytes for Euler")),
    ("euler_xla_temp_bytes",        ("int|None",    "temp buffer bytes for Euler")),
    ("euler_xla_peak_memory_bytes", ("int|None",    "peak memory for Euler")),

    # ── theoretical cost proxies ─────────────────────────────────────────
    ("exact_proxy",                 ("float",       "J * R^2 * m^N (exact cost proxy)")),
    ("euler_proxy",                 ("float",       "J * 2^dyadic * R * m^N (Euler cost proxy)")),
])


def _empty_row() -> dict:
    """Return a dict with every RESULT_SCHEMA field initialised to None."""
    return dict.fromkeys(RESULT_SCHEMA, None)


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------

def brownian_paths(*, n_paths: int, J: int, d: int, seed: int):
    """Generate Brownian paths for testing."""
    key = jax.random.PRNGKey(seed)
    dt = 1.0 / (J - 1)
    dW = jnp.sqrt(dt) * jax.random.normal(key, shape=(n_paths, J - 1, d), dtype=jnp.float64)
    X = jnp.concatenate(
        [jnp.zeros((n_paths, 1, d), dtype=jnp.float64), jnp.cumsum(dW, axis=1)],
        axis=1,
    )
    return X, dt


def random_fssk_params(
    *, q: int, R: int, m: int, d: int, seed: int,
    eig_min: float = 0.1, eig_max: float = 1.5, jordan_alpha: float = 0.25,
    dtype=np.float64,
):
    """Generate random FSSK parameters."""
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


def compute_errors(exact_sig, euler_sig):
    """Compute error metrics between exact and Euler signatures."""
    import json

    max_abs_per_level = []
    mean_abs_per_level = []
    max_rel_per_level = []
    mean_rel_per_level = []

    for n, (S_exact, S_euler) in enumerate(zip(exact_sig, euler_sig)):
        abs_err = jnp.abs(S_exact - S_euler)
        max_abs = float(jnp.max(abs_err))
        mean_abs = float(jnp.mean(abs_err))
        max_abs_per_level.append(max_abs)
        mean_abs_per_level.append(mean_abs)

        # Relative error (avoid division by zero)
        denom = jnp.maximum(jnp.abs(S_exact), 1e-15)
        rel_err = abs_err / denom
        max_rel = float(jnp.max(rel_err))
        mean_rel = float(jnp.mean(rel_err))
        max_rel_per_level.append(max_rel)
        mean_rel_per_level.append(mean_rel)

    return {
        "error_max_abs_global": max(max_abs_per_level),
        "error_mean_abs_global": np.mean(mean_abs_per_level),
        "error_max_rel_global": max(max_rel_per_level),
        "error_mean_rel_global": np.mean(mean_rel_per_level),
        "error_max_abs_by_level": json.dumps(max_abs_per_level),
        "error_mean_abs_by_level": json.dumps(mean_abs_per_level),
    }


def time_exact_sss(*, sss, X, dt, n_repeats):
    """Time exact SSS vsig with comprehensive statistics."""
    def call():
        return sss.vsig(X, dt=dt, axis=-2)

    # First call (JIT compilation)
    _, rec_first = time_call(call)

    # Warmup
    time_warmup(call, n_warmup=1)

    # Hot calls
    hot_records = time_hot_calls(call, n_calls=n_repeats)
    hot_agg = aggregate_timing(hot_records)

    return {
        "wall_first_call_s": rec_first.wall_s,
        **{k: v for k, v in hot_agg.items() if k.startswith("wall_hot_")},
        "cpu_first_call_s": rec_first.cpu_s,
        **{k: v for k, v in hot_agg.items() if k.startswith("cpu_hot_")},
        "ru_first_call_utime_s": rec_first.ru_utime_s,
        "ru_first_call_stime_s": rec_first.ru_stime_s,
        **{k: v for k, v in hot_agg.items() if k.startswith("ru_hot_")},
    }


def time_euler(*, X, kernel, dt, N, dyadic_order, n_repeats):
    """Time Euler vsig with comprehensive statistics."""
    def call():
        return fssk_euler_vsig(X, kernel=kernel, dt=dt, trunc=N, axis=-2, dyadic_order=dyadic_order)

    # First call (JIT compilation)
    _, rec_first = time_call(call)

    # Warmup
    time_warmup(call, n_warmup=1)

    # Hot calls
    hot_records = time_hot_calls(call, n_calls=n_repeats)
    hot_agg = aggregate_timing(hot_records)

    return {
        "wall_first_call_s": rec_first.wall_s,
        **{k: v for k, v in hot_agg.items() if k.startswith("wall_hot_")},
        "cpu_first_call_s": rec_first.cpu_s,
        **{k: v for k, v in hot_agg.items() if k.startswith("cpu_hot_")},
        "ru_first_call_utime_s": rec_first.ru_utime_s,
        "ru_first_call_stime_s": rec_first.ru_stime_s,
        **{k: v for k, v in hot_agg.items() if k.startswith("ru_hot_")},
    }


def profile_xla(*, sss=None, kernel=None, X_spec, dt, N, dyadic_order=None, method="exact"):
    """Profile XLA compilation and costs."""
    if method == "exact":
        def fn(X):
            return sss.vsig(X, dt=dt, axis=-2)
    else:  # euler
        def fn(X):
            return fssk_euler_vsig(X, kernel=kernel, dt=dt, trunc=N, axis=-2, dyadic_order=dyadic_order)

    try:
        profile = compile_and_profile(fn, X_spec)
        return {
            "xla_lower_time_s": profile.get("xla_lower_time_s"),
            "xla_compile_time_s": profile.get("xla_compile_time_s"),
            "xla_flops": profile.get("xla_flops"),
            "xla_bytes_accessed": profile.get("xla_bytes_accessed"),
            "xla_transcendentals": profile.get("xla_transcendentals"),
            "xla_argument_bytes": profile.get("xla_argument_bytes"),
            "xla_output_bytes": profile.get("xla_output_bytes"),
            "xla_temp_bytes": profile.get("xla_temp_bytes"),
            "xla_peak_memory_bytes": profile.get("xla_peak_memory_bytes"),
        }
    except Exception as e:
        print(f"  WARNING: XLA profiling failed for {method}: {e}")
        return {k: None for k in [
            "xla_lower_time_s", "xla_compile_time_s", "xla_flops",
            "xla_bytes_accessed", "xla_transcendentals", "xla_argument_bytes",
            "xla_output_bytes", "xla_temp_bytes", "xla_peak_memory_bytes",
        ]}


# ---------------------------------------------------------------------------
# 5. Design matrix
# ---------------------------------------------------------------------------

# build_design is replaced by cfg.sample_euler() from regime_configs.

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = REGIMES[args.regime]
    n_repeats = args.repeats if args.repeats is not None else cfg.n_repeats

    seed0 = cfg.seed

    df_design = cfg.sample_euler()

    print(f"Regime          : {args.regime}")
    print(f"n_repeats       : {n_repeats}")
    print(f"Total runs      : {len(df_design)}")
    print()
    print(df_design.groupby(["family", "dyadic_order"]).size().rename("n_runs").to_string())
    print()

    if args.dry_run:
        print("Dry-run mode — exiting without running validation.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime_tag = args.regime.lower()
    out_pkl = args.output_dir / f"euler_convergence_{regime_tag}.pkl"
    out_csv = args.output_dir / f"euler_convergence_{regime_tag}.csv"

    rows = []

    for run_id, design_row in df_design.iterrows():
        cfg_row = design_row.to_dict()
        q = int(cfg_row["q"])
        J = int(cfg_row["J"])
        R = int(cfg_row["R"])
        N = int(cfg_row["N"])
        dyadic_order = int(cfg_row["dyadic_order"])
        m       = int(cfg_row["m"])
        d       = int(cfg_row["d"])
        n_paths = int(cfg_row["n_paths"])

        print(
            f"[{run_id + 1:04d}/{len(df_design)}]  "
            f"{cfg_row['family']:>4} | q={q}  J={J}  R={R}  N={N}  dyadic={dyadic_order}",
            flush=True,
        )

        # Generate data
        X, dt = brownian_paths(n_paths=n_paths, J=J, d=d, seed=seed0 + 10_000 + run_id)
        Lambda, A, b = random_fssk_params(q=q, R=R, m=m, d=d, seed=seed0 + run_id)

        # Construct exact SSS
        sss, rec_construct = time_call(
            StateSpaceSignature.from_matrix,
            Lambda=Lambda, A=A, b=b, trunc=N,
        )

        # Compute reference (exact) signature
        S_exact = sss.vsig(X, dt=dt, axis=-2)

        # Compute Euler signature
        S_euler = fssk_euler_vsig(
            X, kernel=sss.kernel, dt=dt, trunc=N, axis=-2, dyadic_order=dyadic_order
        )

        # Compute errors
        errors = compute_errors(S_exact, S_euler)

        # Time exact SSS
        exact_timings = time_exact_sss(sss=sss, X=X, dt=dt, n_repeats=n_repeats)

        # Time Euler
        euler_timings = time_euler(
            X=X, kernel=sss.kernel, dt=dt, N=N,
            dyadic_order=dyadic_order, n_repeats=n_repeats
        )

        # XLA profiling
        X_spec = jax.ShapeDtypeStruct(X.shape, X.dtype)
        exact_xla = profile_xla(sss=sss, X_spec=X_spec, dt=dt, N=N, method="exact")
        euler_xla = profile_xla(
            kernel=sss.kernel, X_spec=X_spec, dt=dt, N=N,
            dyadic_order=dyadic_order, method="euler"
        )

        # Cost proxies
        intervals = J - 1
        exact_proxy = intervals * (R ** 2) * (m ** N)
        euler_proxy = intervals * (1 << dyadic_order) * R * (m ** N)

        # Build result row
        row = _empty_row()
        row.update(
            run_id=run_id,
            regime=args.regime,
            method="euler_validation",
            family=cfg_row["family"],
            q=q,
            J=J,
            intervals=intervals,
            R=R,
            m=m,
            d=d,
            N=N,
            n_paths=n_paths,
            dyadic_order=dyadic_order,
            n_repeats=n_repeats,
            # Errors
            **errors,
            # Exact SSS timings
            exact_wall_construct_s=rec_construct.wall_s,
            exact_cpu_construct_s=rec_construct.cpu_s,
            exact_ru_construct_utime_s=rec_construct.ru_utime_s,
            exact_ru_construct_stime_s=rec_construct.ru_stime_s,
            **{f"exact_{k}": v for k, v in exact_timings.items()},
            # Euler timings
            **{f"euler_{k}": v for k, v in euler_timings.items()},
            # XLA profiling
            **{f"exact_{k}": v for k, v in exact_xla.items()},
            **{f"euler_{k}": v for k, v in euler_xla.items()},
            # Cost proxies
            exact_proxy=exact_proxy,
            euler_proxy=euler_proxy,
        )
        rows.append(row)

        # Checkpoint every 25 runs
        if (run_id + 1) % 25 == 0:
            pd.DataFrame(rows).to_pickle(out_pkl)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  checkpoint → {out_pkl.name}", flush=True)

    df_results = pd.DataFrame(rows)
    df_results = df_results[list(RESULT_SCHEMA.keys())]
    df_results.to_pickle(out_pkl)
    df_results.to_csv(out_csv, index=False)

    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")
    print()
    print("Error summary by dyadic_order:")
    print(df_results.groupby("dyadic_order").agg({
        "error_max_abs_global": ["mean", "max"],
        "error_mean_abs_global": "mean",
    }).to_string())


if __name__ == "__main__":
    main()



