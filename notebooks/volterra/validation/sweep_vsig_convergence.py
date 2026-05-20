"""
sweep_vsig_convergence.py — Convergence sweep for vsig_fft vs predictor-corrector.

Compares vsig_fft (orders 0, 1, 2) and the fractional predictor-corrector scheme
against a high-accuracy vsig_fft reference, sweeping either grid steps or dyadic
refinement order.  Results are saved as pickled DataFrames for analyse_vsig_convergence.py.

Paths: unit-speed paths of dimension --dim (default 3); A = I_dim.

Output (--output-dir):
    vsig_conv_{mode}.pkl  — dict with keys:
        "terminal"  DataFrame: beta, method, order, x_val, level,
                               max_abs_entry, wall_median_s
        "traj"      DataFrame: beta, method, order, x_val, level,
                               max_abs_entry_traj
        "params"    dict of run parameters

Usage
-----
    python sweep_vsig_convergence.py
    python sweep_vsig_convergence.py --betas 0.1 0.6 --convergence-mode steps
    python sweep_vsig_convergence.py --convergence-mode dyadic --vsig-dyadic-max 5
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # notebooks/

from tensordev.util.random_paths import unit_speed_paths, brownian_motion_paths
from tensordev.volterra import ConvolutionKernel, vsig, vsig_fft
from tensordev.volterra.iteration_fft import precompute_lag_tables
from _validation_util.timing_utils import time_warmup, time_hot_calls, aggregate_timing


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_BETAS          = [0.1, 0.6]
_DEFAULT_MODE           = "dyadic"
_DEFAULT_TRUNC          = 6
_DEFAULT_BATCH          = 6
_DEFAULT_SEED           = 7
_DEFAULT_T              = 1.0
_DEFAULT_N_TIMING       = 3

# "steps" mode defaults
_DEFAULT_STEPS_LIST     = [8, 16, 32, 64, 128]
_DEFAULT_REF_STEPS_FACTOR = 4

# "dyadic" mode defaults
_DEFAULT_FIXED_STEPS    = 128
_DEFAULT_VSIG_DYADIC_MAX = 4
_DEFAULT_PC_DYADIC_MAX  = 8
_DEFAULT_REF_DYADIC_EXTRA = 2

_DEFAULT_VSIG_ORDERS    = [0, 1, 2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="vsig convergence sweep")
    p.add_argument("--betas", nargs="+", type=float, default=_DEFAULT_BETAS)
    p.add_argument("--convergence-mode", choices=["steps", "dyadic"],
                   default=_DEFAULT_MODE)
    p.add_argument("--trunc", type=int, default=_DEFAULT_TRUNC)
    p.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--T", type=float, default=_DEFAULT_T)
    p.add_argument("--n-timing-repeats", type=int, default=_DEFAULT_N_TIMING)
    p.add_argument("--vsig-orders", nargs="+", type=int, default=_DEFAULT_VSIG_ORDERS)
    # "steps" mode
    p.add_argument("--steps-list", nargs="+", type=int, default=_DEFAULT_STEPS_LIST)
    p.add_argument("--ref-steps-factor", type=int, default=_DEFAULT_REF_STEPS_FACTOR)
    # "dyadic" mode
    p.add_argument("--fixed-steps", type=int, default=_DEFAULT_FIXED_STEPS)
    p.add_argument("--vsig-dyadic-max", type=int, default=_DEFAULT_VSIG_DYADIC_MAX)
    p.add_argument("--pc-dyadic-max", type=int, default=_DEFAULT_PC_DYADIC_MAX)
    p.add_argument("--ref-dyadic-extra", type=int, default=_DEFAULT_REF_DYADIC_EXTRA)
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--path-type", choices=["unit_speed", "brownian"],
                   default="unit_speed",
                   help="Path type: 'unit_speed' (default) or 'brownian' (Wiener process).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level_errors(ref, got, batch: int) -> list[dict]:
    rows = []
    for level, (a, b) in enumerate(zip(ref, got)):
        fscale = math.factorial(level)
        diff = (jnp.asarray(b).reshape((batch, -1)) - jnp.asarray(a).reshape((batch, -1))) * fscale
        rows.append({"level": level, "max_abs_entry": float(jnp.max(jnp.abs(diff)))})
    return rows


def _traj_errors(ref_traj, out_traj) -> list[dict]:
    rows = []
    for level, (a, b) in enumerate(zip(ref_traj, out_traj)):
        fscale = math.factorial(level)
        diff = (jnp.asarray(b) - jnp.asarray(a)) * fscale
        rows.append({"level": level, "max_abs_entry_traj": float(jnp.max(jnp.abs(diff)))})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode         = args.convergence_mode
    betas        = args.betas
    trunc        = args.trunc
    batch        = args.batch
    T            = args.T
    n_repeats    = args.n_timing_repeats
    vsig_orders  = args.vsig_orders
    dim          = args.dim
    path_type    = args.path_type
    dtype        = jnp.float64

    # ── Mode-specific parameters ─────────────────────────────────────────────
    if mode == "steps":
        steps_list     = sorted(args.steps_list)
        ref_steps      = steps_list[-1] * args.ref_steps_factor
        x_vals_vsig    = steps_list
        x_vals_pc      = steps_list
        x_key          = "steps"
    else:
        fixed_steps    = args.fixed_steps
        x_vals_vsig    = list(range(args.vsig_dyadic_max + 1))
        x_vals_pc      = list(range(args.pc_dyadic_max + 1))
        ref_dyadic_order = args.vsig_dyadic_max + args.ref_dyadic_extra
        x_key          = "dyadic_order"

    # ── Path generation ──────────────────────────────────────────────────────
    print("Generating paths ...")
    if mode == "steps":
        all_steps = sorted(set(steps_list) | {ref_steps})
        X_by_steps: dict[int, jax.Array] = {}
        for s in all_steps:
            dt_s = T / s
            if path_type == "brownian":
                _raw = brownian_motion_paths(dt=dt_s, n_paths=batch, dim=dim,
                                             T=T, seed=args.seed, dtype=float)
            else:
                _raw = unit_speed_paths(dt=dt_s, dt_fine=dt_s / 1024,
                                        n_paths=batch, dim=dim, seed=args.seed, dtype=float)
            X_by_steps[s] = jnp.asarray(_raw, dtype=dtype)
            print(f"  steps={s:5d}: {X_by_steps[s].shape}")
        X_ref   = X_by_steps[ref_steps]
        dt_ref  = T / ref_steps
    else:
        dt_fixed = T / fixed_steps
        if path_type == "brownian":
            _raw = brownian_motion_paths(dt=dt_fixed, n_paths=batch, dim=dim,
                                         T=T, seed=args.seed, dtype=float)
        else:
            _raw = unit_speed_paths(dt=dt_fixed, dt_fine=dt_fixed / 1024,
                                    n_paths=batch, dim=dim, seed=args.seed, dtype=float)
        X_single = jnp.asarray(_raw, dtype=dtype)
        print(f"  steps={fixed_steps}: {X_single.shape}")
        X_ref   = X_single
        dt_ref  = T / fixed_steps

    A = jnp.eye(dim, dtype=dtype)[None, :, :]   # (1, dim, dim)

    rows_terminal: list[dict] = []
    rows_traj:     list[dict] = []

    # ── Beta loop ────────────────────────────────────────────────────────────
    for beta in betas:
        print(f"\n{'═' * 60}")
        print(f"  β = {beta}")
        print(f"{'═' * 60}")

        kernel = ConvolutionKernel.fractional(beta=jnp.array([beta], dtype=dtype), A=A)

        # ── Reference ──────────────────────────────────────────────────────
        if mode == "steps":
            ref_S = ref_steps
            ref_h = T / ref_steps
            ref_d = 0

            print(f"\nPrecomputing reference (S={ref_S}, h={ref_h:.6g}) ...",
                  end=" ", flush=True)
            ref_lag = precompute_lag_tables(kernel, S=ref_S, h=ref_h, order=2,
                                            trunc=trunc, dtype=dtype)
            reference = vsig_fft(X_ref, kernel=kernel, dt=dt_ref, trunc=trunc, axis=1,
                                 increment_input=False, dyadic_order=ref_d, order=2,
                                 lag_tables=ref_lag)
            jax.block_until_ready(reference)
            ref_traj_raw = vsig_fft(X_ref, kernel=kernel, dt=dt_ref, trunc=trunc, axis=1,
                                    increment_input=False, dyadic_order=ref_d, order=2,
                                    lag_tables=ref_lag, block_size=1, accumulate=True,
                                    output_starting_point=True)
            jax.block_until_ready(ref_traj_raw)
            print("done.")
        else:
            # Dyadic mode: Richardson extrapolation from two consecutive levels.
            # Fine and coarse lag tables are computed and freed sequentially so only
            # one large table is live at a time.  Peak memory ≈ one fine-level table
            # (half of what ref_dyadic_extra=2 would need without extrapolation).
            ref_d_fine   = ref_dyadic_order          # vsig_dyadic_max + ref_dyadic_extra
            ref_d_coarse = ref_dyadic_order - 1

            print(f"\nPrecomputing reference via Richardson (λ={ref_d_coarse}→{ref_d_fine}) ...",
                  end=" ", flush=True)

            # Fine level — compute then free the lag table
            _lag = precompute_lag_tables(
                kernel, S=fixed_steps * (1 << ref_d_fine),
                h=dt_fixed / (1 << ref_d_fine), order=2, trunc=trunc, dtype=dtype)
            _V_fine = vsig_fft(X_ref, kernel=kernel, dt=dt_ref, trunc=trunc, axis=1,
                               increment_input=False, dyadic_order=ref_d_fine, order=2,
                               lag_tables=_lag)
            jax.block_until_ready(_V_fine)
            _traj_fine = vsig_fft(X_ref, kernel=kernel, dt=dt_ref, trunc=trunc, axis=1,
                                  increment_input=False, dyadic_order=ref_d_fine, order=2,
                                  lag_tables=_lag, block_size=1, accumulate=True,
                                  output_starting_point=True)
            jax.block_until_ready(_traj_fine)
            del _lag

            # Coarse level — compute then free the lag table
            _lag = precompute_lag_tables(
                kernel, S=fixed_steps * (1 << ref_d_coarse),
                h=dt_fixed / (1 << ref_d_coarse), order=2, trunc=trunc, dtype=dtype)
            _V_coarse = vsig_fft(X_ref, kernel=kernel, dt=dt_ref, trunc=trunc, axis=1,
                                 increment_input=False, dyadic_order=ref_d_coarse, order=2,
                                 lag_tables=_lag)
            jax.block_until_ready(_V_coarse)
            _traj_coarse = vsig_fft(X_ref, kernel=kernel, dt=dt_ref, trunc=trunc, axis=1,
                                    increment_input=False, dyadic_order=ref_d_coarse, order=2,
                                    lag_tables=_lag, block_size=1, accumulate=True,
                                    output_starting_point=True)
            jax.block_until_ready(_traj_coarse)
            del _lag

            # Richardson p=1 per dyadic step: V_rich = 2·V_fine − V_coarse
            reference    = tuple(2.0 * f - c for f, c in zip(_V_fine, _V_coarse))
            ref_traj_raw = tuple(2.0 * f - c for f, c in zip(_traj_fine, _traj_coarse))
            del _V_fine, _V_coarse, _traj_fine, _traj_coarse
            print("done.")

        # ── vsig sweep ─────────────────────────────────────────────────────
        total_vsig = len(vsig_orders) * len(x_vals_vsig)
        print(f"\nSweeping vsig ({total_vsig} runs) ...")
        for order in vsig_orders:
            for idx, x in enumerate(x_vals_vsig):
                run_num = vsig_orders.index(order) * len(x_vals_vsig) + idx + 1
                print(f"  [vsig {run_num:2d}/{total_vsig}] order={order}, {x_key}={x}",
                      end="  ... ", flush=True)

                if mode == "steps":
                    X_x, dt_x, d_x = X_by_steps[x], T / x, 0
                    S_x, h_x = x, T / x
                else:
                    X_x, dt_x, d_x = X_single, T / fixed_steps, x
                    S_x, h_x = fixed_steps * (1 << x), dt_fixed / (1 << x)

                lag = precompute_lag_tables(
                    kernel, S=S_x, h=h_x, order=order, trunc=trunc, dtype=dtype)

                def _vsig_call(X_x=X_x, dt_x=dt_x, d_x=d_x, order=order, lag=lag):
                    return vsig_fft(X_x, kernel=kernel, dt=dt_x, trunc=trunc, axis=1,
                                    dyadic_order=d_x, order=order, lag_tables=lag)

                time_warmup(_vsig_call, n_warmup=1)
                records = time_hot_calls(_vsig_call, n_calls=n_repeats)
                agg = aggregate_timing(records)
                out = _vsig_call()
                jax.block_until_ready(out)
                wall_s = agg["wall_hot_median_s"]
                print(f"{wall_s * 1e3:8.1f} ms")

                for row in _level_errors(reference, out, batch):
                    row.update(beta=beta, method="vsig", order=order, x_val=x,
                               wall_median_s=wall_s)
                    rows_terminal.append(row)

                # trajectory
                out_traj = vsig_fft(X_x, kernel=kernel, dt=dt_x, trunc=trunc, axis=1,
                                    dyadic_order=d_x, order=order, lag_tables=lag,
                                    block_size=1, accumulate=True,
                                    output_starting_point=True)
                if mode == "steps":
                    factor = ref_steps // x
                    ref_sub = tuple(lvl[:, ::factor] for lvl in ref_traj_raw)
                else:
                    ref_sub = ref_traj_raw  # block_size=1 returns coarse grid directly

                for row in _traj_errors(ref_sub, out_traj):
                    row.update(beta=beta, method="vsig", order=order, x_val=x)
                    rows_traj.append(row)

        # ── PC sweep ───────────────────────────────────────────────────────
        print(f"\nSweeping PC ({len(x_vals_pc)} runs) ...")
        for idx, x in enumerate(x_vals_pc, 1):
            print(f"  [PC {idx:2d}/{len(x_vals_pc)}] {x_key}={x}",
                  end="  ... ", flush=True)

            if mode == "steps":
                X_x, dt_x, d_x = X_by_steps[x], T / x, 0
            else:
                X_x, dt_x, d_x = X_single, T / fixed_steps, x

            def _pc_call(X_x=X_x, dt_x=dt_x, d_x=d_x):
                return vsig(
                    X_x, kernel=kernel, trunc=trunc, dt=dt_x, axis=1,
                    increment_input=False, dyadic_order=d_x,
                    scheme="adams", order=1,
                )

            time_warmup(_pc_call, n_warmup=1)
            records = time_hot_calls(_pc_call, n_calls=n_repeats)
            agg = aggregate_timing(records)
            out = _pc_call()
            jax.block_until_ready(out)
            wall_s = agg["wall_hot_median_s"]
            print(f"{wall_s * 1e3:8.1f} ms")

            for row in _level_errors(reference, out, batch):
                row.update(beta=beta, method="pc", order=None, x_val=x,
                           wall_median_s=wall_s)
                rows_terminal.append(row)

            # trajectory
            out_traj = vsig(
                X_x, kernel=kernel, trunc=trunc, dt=dt_x, axis=1,
                increment_input=False, dyadic_order=d_x,
                scheme="adams", order=1,
                block_size=1, accumulate=True, output_starting_point=True,
            )
            if mode == "steps":
                factor = ref_steps // x
                ref_sub = tuple(lvl[:, ::factor] for lvl in ref_traj_raw)
            else:
                ref_sub = ref_traj_raw

            for row in _traj_errors(ref_sub, out_traj):
                row.update(beta=beta, method="pc", order=None, x_val=x)
                rows_traj.append(row)

    # ── Save ─────────────────────────────────────────────────────────────────
    df_terminal = pd.DataFrame(rows_terminal)
    df_traj     = pd.DataFrame(rows_traj)

    params = dict(
        betas=betas, convergence_mode=mode, trunc=trunc, batch=batch,
        seed=args.seed, T=T, n_timing_repeats=n_repeats, vsig_orders=vsig_orders,
        dim=dim, path_type=path_type,
    )
    if mode == "steps":
        params.update(steps_list=steps_list, ref_steps=ref_steps,
                      ref_steps_factor=args.ref_steps_factor)
    else:
        params.update(fixed_steps=fixed_steps, vsig_dyadic_max=args.vsig_dyadic_max,
                      pc_dyadic_max=args.pc_dyadic_max,
                      ref_dyadic_extra=args.ref_dyadic_extra,
                      ref_dyadic_order=ref_dyadic_order)

    out_pkl = args.output_dir / f"vsig_conv_{mode}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"terminal": df_terminal, "traj": df_traj, "params": params}, f)

    print(f"\nSaved: {out_pkl}")
    print(f"  terminal rows: {len(df_terminal)}")
    print(f"  traj rows:     {len(df_traj)}")


if __name__ == "__main__":
    main()