"""
sweep_vsig_flop_scaling.py — XLA FLOP-count sweep for Volterra vsig.

Samples (J, N) uniformly within a workload-capped region and crosses with
q ∈ {1, 2, 3, 4}.  For each configuration three FLOP counts are profiled via
abstract (device-free) XLA compilation:

  fft_pre_flops   XLA FLOPs for lag-table precomputation (precompute_lag_tables)
  fft_hot_flops   XLA FLOPs for the FFT hot loop with precomputed tables
  quad_flops      XLA FLOPs for the quadratic hot loop

fft_total_flops = fft_pre_flops + fft_hot_flops is derived in the analysis step.

Fixed constants: d=3, m=3, beta=0.6, order=2, dyadic_order=0.

Output (--output-dir):
    vsig_flop_scaling_<regime>.pkl
    vsig_flop_scaling_<regime>.csv

Usage
-----
    python sweep_vsig_flop_scaling.py --regime SMALL
    python sweep_vsig_flop_scaling.py --regime MEDIUM
    python sweep_vsig_flop_scaling.py --regime MEDIUM --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # notebooks/

from tensordev.volterra.kernel import FractionalKernel
from tensordev.volterra.iteration_quad import quadratic_iteration
from tensordev.volterra.iteration_fft import fft_iteration, precompute_lag_tables
from _validation_util.xla_utils import compile_and_profile
from vsig_setup import REGIMES, D, M, BETA, ORDER, DYADIC


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Volterra vsig FLOP-count sweep")
    p.add_argument(
        "--regime",
        choices=["SMALL", "MEDIUM", "LARGE"],
        default="MEDIUM",
        help="Sweep regime (default: MEDIUM)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "validation_outputs",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lag_precomp_pytree(kernel, S, order, trunc):
    """Wrap precompute_lag_tables as a function of h only, returning a flat list."""
    def _fn(h):
        tables = precompute_lag_tables(
            kernel, S=S, h=h, order=order, trunc=trunc, dtype=jnp.float64)
        arrs = []
        for tt in tables.theta_tables:
            for level_w in tt.weights:
                arrs.extend(level_w)
        for level_w in tables.output_table.weights:
            arrs.extend(level_w)
        return arrs
    return _fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = REGIMES[args.regime]
    df_design = cfg.sample_flop()

    print(f"Regime         : {args.regime}")
    print(f"Configurations : {len(df_design)}")
    print(f"J range        : {df_design['J'].min()} – {df_design['J'].max()}")
    print(f"N range        : {df_design['N'].min()} – {df_design['N'].max()}")
    print(f"q values       : {sorted(df_design['q'].unique())}  (kernel.q = #kernel components)")
    print(f"m (fixed)      : {M}  (kernel.m = Volterra path dim)")
    print(f"d (fixed)      : {D}  (kernel.path_dim = input path dim)")
    print(f"beta (fixed)   : {BETA}")
    print(f"order (fixed)  : {ORDER}")
    print()

    if args.dry_run:
        print("Dry-run — exiting.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag     = args.regime.lower()
    out_pkl = args.output_dir / f"vsig_flop_scaling_{tag}.pkl"
    out_csv = args.output_dir / f"vsig_flop_scaling_{tag}.csv"

    dtype = jnp.float64
    rows  = []

    for idx, row in df_design.iterrows():
        J = int(row["J"])
        N = int(row["N"])
        q = int(row["q"])    # kernel.q = number of kernel components = A.shape[0]
        S = J - 1
        h = np.float64(1.0 / S)

        print(f"[{idx + 1:04d}/{len(df_design)}]  J={J}  N={N}  q={q}", flush=True)

        A      = jnp.ones((q, M, D), dtype=dtype) / np.sqrt(D)  # shape (q, m, d)
        kernel = FractionalKernel(
            beta=jnp.array([BETA] * q, dtype=dtype),
            A=A,
        )
        h_jax   = jnp.asarray(h, dtype=dtype)
        dX_spec = jax.ShapeDtypeStruct((1, S, D), dtype)
        h_spec  = jax.ShapeDtypeStruct((), dtype)

        # ── 1. Lag precomputation ─────────────────────────────────────────
        precomp_fn  = _lag_precomp_pytree(kernel, S=S, order=ORDER, trunc=N)
        pre_profile = compile_and_profile(precomp_fn, h_spec)

        # ── 2. FFT hot loop (precomputed tables as closure constants) ─────
        lag = precompute_lag_tables(
            kernel, S=S, h=h_jax, order=ORDER, trunc=N, dtype=dtype)

        def fft_hot_fn(dX, _lag=lag):
            return fft_iteration(dX, kernel=kernel, trunc=N, dt=h_jax,
                                 axis=-2, order=ORDER, lag_tables=_lag)

        fft_hot_profile = compile_and_profile(fft_hot_fn, dX_spec)

        # ── 3. Quadratic hot loop ─────────────────────────────────────────
        # XLA cost_analysis() reports the lax.scan body cost only, not body × S.
        # Multiply by S here to recover the true total online-recursion cost,
        # mirroring the correction applied in the SSS sweep_flop_scaling.py.
        def quad_fn(dX):
            return quadratic_iteration(dX, kernel=kernel, trunc=N, dt=h_jax,
                                       axis=-2, order=ORDER)

        quad_profile = compile_and_profile(quad_fn, dX_spec)

        # XLA cost_analysis for lax.scan reports only the body cost (not body × S).
        # The body itself processes all S source points (masked sum), so body ∝ S.
        # True total = S × body ∝ S². Multiply by S to recover the full online cost.
        def _scale(v):
            try:
                return float(v) * S if v is not None and np.isfinite(float(v)) else v
            except Exception:
                return v

        rows.append(dict(
            regime=args.regime,
            J=J, N=N, q=q, m=M, d=D, S=S,  # q=kernel.q, m=kernel.m=M, d=kernel.path_dim
            order=ORDER, dyadic_order=DYADIC, beta=BETA,
            # FFT: no lax.scan → XLA counts the full unrolled graph correctly.
            fft_pre_flops  = pre_profile.get("xla_flops"),
            fft_hot_flops  = fft_hot_profile.get("xla_flops"),
            # Quadratic: multiply raw scan body cost by S to get true total.
            quad_flops      = _scale(quad_profile.get("xla_flops")),
            quad_flops_body = quad_profile.get("xla_flops"),
            quad_bytes      = _scale(quad_profile.get("xla_bytes_accessed")),
            fft_pre_bytes  = pre_profile.get("xla_bytes_accessed"),
            fft_hot_bytes  = fft_hot_profile.get("xla_bytes_accessed"),
            fft_pre_lower_s    = pre_profile.get("xla_lower_time_s"),
            fft_pre_compile_s  = pre_profile.get("xla_compile_time_s"),
            fft_hot_lower_s    = fft_hot_profile.get("xla_lower_time_s"),
            fft_hot_compile_s  = fft_hot_profile.get("xla_compile_time_s"),
            quad_lower_s       = quad_profile.get("xla_lower_time_s"),
            quad_compile_s     = quad_profile.get("xla_compile_time_s"),
            fft_pre_error  = pre_profile.get("_lower_error") or pre_profile.get("_compile_error"),
            fft_hot_error  = fft_hot_profile.get("_lower_error") or fft_hot_profile.get("_compile_error"),
            quad_error     = quad_profile.get("_lower_error") or quad_profile.get("_compile_error"),
        ))

        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_pickle(out_pkl)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  checkpoint → {out_pkl.name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")
    valid = df.dropna(subset=["fft_hot_flops", "quad_flops"])
    print(f"\nValid rows: {len(valid)} / {len(df)}")
    print(valid[["J", "N", "q", "m", "fft_pre_flops", "fft_hot_flops", "quad_flops"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
