"""
sweep_flop_scaling.py — FSSK XLA FLOP and memory-cost scaling sweep.

Uses abstract inputs (jax.ShapeDtypeStruct) to profile the FSSK recursion
with scalar coefficients precomputed outside the traced function. This matches
the cost model in the paper, where coefficient families are assumed available
before the online state recursion. The script also profiles scalar coefficient
construction separately.

Usage
-----
    python sweep_flop_scaling.py --regime MEDIUM
    python sweep_flop_scaling.py --regime SMALL --dry-run
"""

import argparse
import itertools
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensordev.sss import StateSpaceSignature
from tensordev.sss.state_update import fssk_state_from_coef
from xla_utils import compile_and_profile
from regime_configs import REGIMES

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSSK FLOP-scaling sweep")
    p.add_argument(
        "--regime",
        choices=["SMALL", "MEDIUM", "LARGE"],
        default="MEDIUM",
        help="Sweep regime (default: MEDIUM)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "validation_outputs",
        help="Output directory for results",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print design and exit without profiling",
    )
    p.add_argument(
        "--coef-runtime-repeats",
        type=int,
        default=0,
        help=(
            "Optional: additionally execute scalar coefficient construction this "
            "many hot repetitions and store wall/CPU timing summaries. Default 0 "
            "keeps this script as compile/cost-analysis only."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_fssk_params(
    *, q: int, R: int, m: int, d: int, seed: int,
    eig_min: float = 0.1, eig_max: float = 1.5, jordan_alpha: float = 0.25,
    dtype=np.float64,
):
    """Generate random FSSK parameters (Lambda, A, b)."""
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


def build_design(cfg: "RegimeConfig") -> pd.DataFrame:
    """Delegate to cfg.sample_flop() — uniform random sampling over ranges."""
    return cfg.sample_flop()


def _profile_with_prefix(profile: dict, prefix: str) -> dict:
    """Prefix every field returned by compile_and_profile."""
    return {f"{prefix}_{k}": v for k, v in profile.items()}


def _runtime_summary_with_prefix(summary: dict, prefix: str) -> dict:
    """Prefix timing_utils.aggregate_timing output."""
    return {f"{prefix}_{k}": v for k, v in summary.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = REGIMES[args.regime]
    df_design = build_design(cfg)

    print(f"Regime          : {args.regime}")
    print(f"Total configs   : {len(df_design)}")
    print()
    print("Design summary:")
    print(f"  q   : {sorted(df_design['q'].unique())}")
    print(f"  J   : {sorted(df_design['J'].unique())}")
    print(f"  R   : {sorted(df_design['R'].unique())}")
    print(f"  N   : {sorted(df_design['N'].unique())}")
    print(f"  m   : {sorted(df_design['m'].unique())}")
    print(f"  d   : {sorted(df_design['d'].unique())}")
    print()

    if args.dry_run:
        print("Dry-run mode — exiting without profiling.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime_tag = args.regime.lower()
    out_pkl = args.output_dir / f"fssk_flop_scaling_{regime_tag}.pkl"
    out_csv = args.output_dir / f"fssk_flop_scaling_{regime_tag}.csv"

    rows = []
    seed0 = 20260507

    for idx, row in df_design.iterrows():
        q, J, R, N, m, d = (
            int(row["q"]), int(row["J"]), int(row["R"]),
            int(row["N"]), int(row["m"]), int(row["d"]),
        )
        n_paths = int(row["n_paths"])

        print(
            f"[{idx + 1:04d}/{len(df_design)}]  "
            f"q={q}  J={J}  R={R}  N={N}  m={m}  d={d}",
            flush=True,
        )

        # Generate FSSK parameters (concrete arrays for Lambda, A, b).
        Lambda, A, b = random_fssk_params(
            q=q, R=R, m=m, d=d, seed=seed0 + idx
        )

        # Construct the StateSpaceSignature object.
        # This step does NOT run expensive JAX ops; it only stores parameters.
        sss = StateSpaceSignature.from_matrix(Lambda=Lambda, A=A, b=b, trunc=N)

        # ------------------------------------------------------------------
        # Main profiled object: FSSK recursion with scalar coefficients
        # precomputed outside the traced function. This matches the online
        # recursion cost model in the paper.
        # ------------------------------------------------------------------
        S = J - 1
        dt = 1.0 / S
        dt_scalar = jnp.asarray(dt, dtype=jnp.float64)

        # Precompute scalar coefficients once. The recursion profile below
        # captures these coefficient arrays as constants and only traces the
        # projected-increment update/readout.
        coef = sss.kernel.coef(dt_scalar, trunc=N, dtype=jnp.float64)

        if q == 1:
            y_spec = jax.ShapeDtypeStruct((n_paths, S, m), jnp.float64)
            y_axis = -2
        else:
            y_spec = jax.ShapeDtypeStruct((n_paths, S, q, m), jnp.float64)
            y_axis = -3

        def recursion_fn(y):
            # Profile only the state recursion.  The terminal readout is lower
            # order for the FSSK cost proposition and, more importantly, XLA
            # cost_analysis() reports the lax.scan body cost rather than
            # multiplying by the static trip count S.  We therefore store both
            # the scan-body estimate and the total online-work estimate S times
            # the body cost below.
            return fssk_state_from_coef(
                y,
                coef=coef,
                axis=y_axis,
                block_size=None,
                accumulate=True,
                initial_state=sss.state,
                output_starting_state=False,
            )

        recursion_profile = compile_and_profile(recursion_fn, y_spec)

        def _scale_scan_cost(value, multiplier):
            if value is None:
                return value
            try:
                if not np.isfinite(float(value)):
                    return value
                return float(value) * float(multiplier)
            except Exception:
                return value

        # XLA currently reports the cost of the compiled scan body for this
        # recursion, not the cost multiplied by the number of time steps.  The
        # following total estimates are the quantities to compare with the
        # paper's J-dependent cost model.  Memory-size quantities are *not*
        # multiplied here; only arithmetic / traffic counters are.
        recursion_total_estimates = {
            "xla_flops": _scale_scan_cost(recursion_profile.get("xla_flops"), S),
            "xla_bytes_accessed": _scale_scan_cost(recursion_profile.get("xla_bytes_accessed"), S),
            "xla_transcendentals": _scale_scan_cost(recursion_profile.get("xla_transcendentals"), S),
        }

        # ------------------------------------------------------------------
        # Secondary profiled object: scalar coefficient construction. Here dt
        # is a dynamic scalar input, so XLA cannot simply constant-fold the
        # whole coefficient object for this configuration.
        # ------------------------------------------------------------------
        dt_spec = jax.ShapeDtypeStruct((), jnp.float64)

        def coef_fn(dt_arg):
            return sss.kernel.coef(dt_arg, trunc=N, dtype=jnp.float64)

        coef_profile = compile_and_profile(coef_fn, dt_spec)

        # Optional actual hot runtime for coefficient construction. Disabled
        # by default because this sweep is mainly a compile/cost-analysis
        # experiment.
        coef_runtime = {}
        if args.coef_runtime_repeats > 0:
            from timing_utils import aggregate_timing, time_hot_calls, time_warmup

            coef_jit = jax.jit(coef_fn)
            time_warmup(coef_jit, dt_scalar, n_warmup=1)
            records = time_hot_calls(
                coef_jit, dt_scalar, n_calls=args.coef_runtime_repeats
            )
            coef_runtime = _runtime_summary_with_prefix(
                aggregate_timing(records), "coef_runtime"
            )

        # Build the result row. For backward compatibility with the analysis
        # script, the unprefixed xla_* columns refer to the main recursion
        # profile. Prefixed columns keep recursion and coefficient profiles
        # side by side.
        result = dict(
            config_id=idx,
            regime=args.regime,
            design=row.get("design", "random"),
            q=q,
            J=J,
            R=R,
            N=N,
            m=m,
            d=d,
            n_paths=n_paths,
            S=S,
            profiled_object="precomputed_state_recursion_total_estimate",
            coefficient_profiled_separately=True,
            xla_cost_analysis_semantics="lax_scan_body_cost_scaled_by_S_for_total_online_work",
            y_axis=y_axis,
            y_shape=tuple(y_spec.shape),
            dt_scalar=float(dt),
            # Backward-compatible main XLA profiling results: total online
            # recursion-work estimates.  The corresponding scan-body estimates
            # are stored in recursion_body_xla_* below.
            inputs_are_abstract=recursion_profile["inputs_are_abstract"],
            xla_lower_time_s=recursion_profile["xla_lower_time_s"],
            xla_compile_time_s=recursion_profile["xla_compile_time_s"],
            xla_flops=recursion_total_estimates["xla_flops"],
            xla_bytes_accessed=recursion_total_estimates["xla_bytes_accessed"],
            xla_transcendentals=recursion_total_estimates["xla_transcendentals"],
            xla_argument_bytes=recursion_profile["xla_argument_bytes"],
            xla_output_bytes=recursion_profile["xla_output_bytes"],
            xla_temp_bytes=recursion_profile["xla_temp_bytes"],
            xla_alias_bytes=recursion_profile["xla_alias_bytes"],
            xla_peak_memory_bytes=recursion_profile["xla_peak_memory_bytes"],
            xla_generated_code_bytes=recursion_profile["xla_generated_code_bytes"],
            recursion_body_xla_flops=recursion_profile["xla_flops"],
            recursion_body_xla_bytes_accessed=recursion_profile["xla_bytes_accessed"],
            recursion_body_xla_transcendentals=recursion_profile["xla_transcendentals"],
            cost_analysis_raw=recursion_profile["cost_analysis_raw"],
            memory_analysis_raw=recursion_profile["memory_analysis_raw"],
        )
        result.update(_profile_with_prefix(recursion_profile, "recursion"))
        result.update(_profile_with_prefix(coef_profile, "coef"))
        result.update(coef_runtime)

        # Include any error diagnostics if present.
        for err_key in ["_lower_error", "_compile_error", "_cost_error", "_memory_error"]:
            if err_key in recursion_profile:
                result[f"recursion{err_key}"] = recursion_profile[err_key]
                result[err_key] = recursion_profile[err_key]
            if err_key in coef_profile:
                result[f"coef{err_key}"] = coef_profile[err_key]

        rows.append(result)

        # Checkpoint every 50 configs.
        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_pickle(out_pkl)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  checkpoint → {out_pkl.name}", flush=True)

    df_results = pd.DataFrame(rows)
    df_results.to_pickle(out_pkl)
    df_results.to_csv(out_csv, index=False)

    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")
    print()
    print("Summary of XLA FLOPs:")
    print(df_results[["q", "J", "R", "N", "m", "xla_flops"]].describe())


if __name__ == "__main__":
    main()

