"""
sweep_kernel_flop.py — FSSK kernel XLA FLOP and memory-cost scaling sweep.

Profiles fssk_sigkernel (ETD1 and Heun schemes) using abstract inputs to
measure XLA compile-time cost estimates.  Random Jordan-form FSSKs are sampled
from a filtered design grid.

Expected workload model:  J^2 × R^2  (PDE solver; precomp dominates but scales with q)

Usage
-----
    python sweep_kernel_flop.py --regime MEDIUM
    python sweep_kernel_flop.py --regime SMALL --dry-run
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))              # notebooks/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sss" / "validation"))  # fssk_setup

from tensordev.sss.kernel import FSSK
from tensordev.kernel.fssk import (
    fssk_sigkernel,
    _get_solver, _build_transport_params, _prepare_dt,
    _build_gamma_grid_static,
)
from tensordev.kernel.static_kernels import LinearKernel
from fssk_setup import random_fssk
from _validation_util.xla_utils import compile_and_profile
from _validation_util.regime_config import RegimeConfig


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

def _kernel_workload():
    """Kernel workload: J^2 × R^2  (PDE solver cost, confirmed empirically)."""
    def fn(J: np.ndarray, R: np.ndarray, dyadic: np.ndarray) -> np.ndarray:
        J_eff = J * (2.0 ** dyadic)
        return (J_eff ** 2) * (R ** 2)
    return fn


REGIMES: dict[str, RegimeConfig] = {
    "SMALL": RegimeConfig(
        name="SMALL",
        n_repeats=0,  # FLOP sweep = compile-only, no hot calls
        seed=202624516,
        sampled_ranges={"J": (16, 128), "R": (2, 8), "dyadic": (0, 0)},
        crossed_ranges={"q": (1, 1), "m": (3, 3), "d": (3, 3)},
        ref_point={"J": 48, "R": 3, "dyadic": 0},
        workload_fn=_kernel_workload(),
        exact_n_samples=50,
        flop_n_samples=100,
        n_paths=2,
    ),
    "MEDIUM": RegimeConfig(
        name="MEDIUM",
        n_repeats=0,
        seed=21126416,
        sampled_ranges={"J": (2**8+1, 2**12+1), "R": (6, 13), "dyadic": (0, 0)},
        crossed_ranges={"q": (1, 5), "m": (3, 3), "d": (3, 3)},
        ref_point={"J": 512, "R": 10, "dyadic": 0},
        workload_fn=_kernel_workload(),
        exact_n_samples=150,
        flop_n_samples=100,
        n_paths=2,
    ),
    "LARGE": RegimeConfig(
        name="LARGE",
        n_repeats=0,
        seed=20260516,
        sampled_ranges={"J": (16, 512), "R": (2, 16), "dyadic": (0, 0)},
        crossed_ranges={"q": (1, 1), "m": (3, 3), "d": (3, 3)},
        ref_point={"J": 192, "R": 6, "dyadic": 0},
        workload_fn=_kernel_workload(),
        exact_n_samples=300,
        flop_n_samples=100,
        n_paths=2,
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Kernel FLOP-scaling sweep")
    p.add_argument(
        "--regime",
        choices=list(REGIMES),
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = REGIMES[args.regime]
    df_design = cfg.sample_flop()

    print(f"Regime         : {args.regime}")
    print(f"Total configs  : {len(df_design)}")
    print()
    print("Design summary:")
    print(f"  q      : {sorted(df_design['q'].unique())}")
    print(f"  J      : {sorted(df_design['J'].unique())}")
    print(f"  R      : {sorted(df_design['R'].unique())}")
    print(f"  dyadic : {sorted(df_design['dyadic'].unique())}")
    print(f"  m      : {sorted(df_design['m'].unique())}")
    print(f"  d      : {sorted(df_design['d'].unique())}")
    print()

    if args.dry_run:
        print("Dry-run mode -- exiting without profiling.")
        print()
        print(df_design.head(20).to_string(index=False))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime_tag = args.regime.lower()
    out_pkl = args.output_dir / f"kernel_flop_scaling_{regime_tag}.pkl"
    out_csv = args.output_dir / f"kernel_flop_scaling_{regime_tag}.csv"

    rows = []
    seed0 = cfg.seed

    for idx, row_series in df_design.iterrows():
        row = row_series.to_dict()
        q      = int(row["q"])
        J      = int(row["J"])
        R      = int(row["R"])
        dyadic = int(row["dyadic"])
        m      = int(row["m"])
        d      = int(row["d"])
        n_paths = int(row["n_paths"])

        print(
            f"[{idx + 1:04d}/{len(df_design)}]  "
            f"q={q}  J={J}  R={R}  dyadic={dyadic}  m={m}  d={d}",
            flush=True,
        )

        # ── Generate Jordan-form FSSK ─────────────────────────────────────────
        kernel = random_fssk(
            q=q, R=R, m=m, d=d,
            seed=seed0 + idx,
            eig_min=0.1,
            eig_max=1.5,
            normalise_b=False,
        )


        # ── Compute Jordan block sizes for expected-cost comparison ───────────
        # For a near-Jordan matrix with super-diagonal coupling, we assume the
        # block structure can be recovered from consecutive-eigenvalue grouping.
        # For simplicity, we upper-bound by R itself (single block) or use the
        # exact formula if we track block boundaries.  Here we store R and let
        # the analysis script compute sum(block_sizes^2).
        #
        # Simple upper bound: sum(block_sizes^2) ≤ R^2 (one block).
        # Naive multiple blocks: ≈ R  (many 1×1 blocks).
        # For now we just store R and let the fit determine the empirical exponent.
        jordan_block_metric = R  # placeholder; analysis will fit the exponent

        # ── Abstract input specs ───────────────────────────────────────────────
        dt = 1.0 / (J - 1)
        X_spec = jax.ShapeDtypeStruct((n_paths, J, d), jnp.float64)
        Y_spec = jax.ShapeDtypeStruct((n_paths, J, d), jnp.float64)

        # ── Profile Heun (dyadic_order=0) ──────────────────────────────────────
        def heun_fn(X, Y):
            return fssk_sigkernel(
                X, Y,
                kernel=kernel,
                dt_x=dt,
                dt_y=dt,
                evaluate="terminal",
                pairwise=False,
                backend="scan",
                scheme="heun",
                dyadic_order=0,
            )

        heun_profile = compile_and_profile(heun_fn, X_spec, Y_spec)

        # ── Profile precomputation only (gamma construction from X, Y) ────────
        # Same abstract X, Y specs as the full kernel.  This isolates the cost
        # of _build_gamma_grid_static so the PDE-solver cost can be recovered
        # by subtraction:  pde_flops = kernel_flops - precomp_flops.
        def precomp_fn(X, Y):
            gamma_padded, _ = _build_gamma_grid_static(
                X, Y,
                kernel=kernel,
                static_kernel=LinearKernel(scale=1.0),
                pairwise=False,
                dtype=jnp.float64,
            )
            return gamma_padded

        precomp_profile = compile_and_profile(precomp_fn, X_spec, Y_spec)

        # ── Profile PDE-only (abstract gamma, no projection cost) ─────────────
        # gamma shape: (batch, s_nodes, t_nodes, R, R); s_nodes = t_nodes = J
        gamma_spec = jax.ShapeDtypeStruct((n_paths, J, J, R, R), jnp.float64)
        dt_x_arr, dt_x_uniform = _prepare_dt(dt, J - 1, name="dt_x", dtype=jnp.float64)
        dt_y_arr, dt_y_uniform = _prepare_dt(dt, J - 1, name="dt_y", dtype=jnp.float64)
        transport_params = _build_transport_params(
            kernel.Lambda, dt_x_arr, dt_y_arr,
            dt_x_uniform=dt_x_uniform, dt_y_uniform=dt_y_uniform,
            dtype=jnp.float64, precompute_propagators=False,
        )
        pde_solver = _get_solver(
            backend="scan", scheme="heun",
            precompute_propagators=False,
            dyadic_order=(0, 0), terminal_only=True,
        )

        def pde_fn(gamma):
            eta, *_ = pde_solver(
                gamma, dt_x_arr, dt_y_arr,
                lambda_op=kernel.Lambda, transport_params=transport_params,
            )
            return eta

        pde_profile = compile_and_profile(pde_fn, gamma_spec)

        # ── Build result row ───────────────────────────────────────────────────
        def _safe_sub(a, b):
            """Subtract two nullable float-like FLOP values; returns None if either is absent."""
            try:
                if a is None or b is None:
                    return None
                fa, fb = float(a), float(b)
                if not (np.isfinite(fa) and np.isfinite(fb)):
                    return None
                return fa - fb
            except Exception:
                return None

        result = dict(
            config_id=idx,
            regime=args.regime,
            scheme="heun",
            q=q,
            J=J,
            R=R,
            dyadic=0,
            m=m,
            d=d,
            n_paths=n_paths,
            dt=dt,
            jordan_form=True,
            jordan_block_metric=jordan_block_metric,
            # Full kernel XLA profiling results
            inputs_are_abstract=heun_profile["inputs_are_abstract"],
            xla_lower_time_s=heun_profile["xla_lower_time_s"],
            xla_compile_time_s=heun_profile["xla_compile_time_s"],
            xla_flops=heun_profile["xla_flops"],
            xla_bytes_accessed=heun_profile["xla_bytes_accessed"],
            xla_transcendentals=heun_profile["xla_transcendentals"],
            xla_argument_bytes=heun_profile["xla_argument_bytes"],
            xla_output_bytes=heun_profile["xla_output_bytes"],
            xla_temp_bytes=heun_profile["xla_temp_bytes"],
            xla_alias_bytes=heun_profile["xla_alias_bytes"],
            xla_peak_memory_bytes=heun_profile["xla_peak_memory_bytes"],
            xla_generated_code_bytes=heun_profile["xla_generated_code_bytes"],
            cost_analysis_raw=heun_profile["cost_analysis_raw"],
            memory_analysis_raw=heun_profile["memory_analysis_raw"],
            # Precomputation (gamma construction) XLA profiling
            precomp_xla_flops=precomp_profile["xla_flops"],
            precomp_xla_bytes_accessed=precomp_profile["xla_bytes_accessed"],
            precomp_xla_compile_time_s=precomp_profile["xla_compile_time_s"],
            precomp_lower_error=precomp_profile.get("_lower_error"),
            # PDE-solver cost by subtraction: kernel - precomp
            pde_by_subtraction_xla_flops=_safe_sub(
                heun_profile["xla_flops"], precomp_profile["xla_flops"]
            ),
            pde_by_subtraction_xla_bytes_accessed=_safe_sub(
                heun_profile["xla_bytes_accessed"], precomp_profile["xla_bytes_accessed"]
            ),
            # PDE-only XLA profiling (abstract gamma input, kept for reference)
            pde_xla_flops=pde_profile["xla_flops"],
            pde_xla_compile_time_s=pde_profile["xla_compile_time_s"],
            pde_lower_error=pde_profile.get("_lower_error"),
        )
        # Include any error diagnostics
        for err_key in ["_lower_error", "_compile_error", "_cost_error", "_memory_error"]:
            if err_key in heun_profile:
                result[err_key] = heun_profile[err_key]
            if err_key in precomp_profile:
                result[f"precomp{err_key}"] = precomp_profile[err_key]

        rows.append(result)

        # Checkpoint every 50 configs
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
    print(df_results[["scheme", "J", "R", "dyadic", "xla_flops"]].describe())


if __name__ == "__main__":
    main()

