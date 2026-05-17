"""
sweep_fssk_approx.py — FSSK rough approximation vs predictor-corrector reference.

Sweeps the FSSK state dimension R and compares the resulting signature against
a high-accuracy fractional predictor-corrector reference.  Also records a vsig
(order=2) baseline error for comparison.

Paths: unit-speed paths of dimension --dim (default 3); A = I_dim.

Output (--output-dir):
    fssk_approx.pkl  — dict with keys:
        "fssk"    DataFrame: beta, R, level, max_abs_entry
        "vsig"    DataFrame: beta, level, max_abs_entry  (order=2 baseline)
        "params"  dict of run parameters

Usage
-----
    python sweep_fssk_approx.py
    python sweep_fssk_approx.py --betas 0.1 0.6 --R-values 2 4 6 8 10
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # notebooks/

from tensordev.util.random_paths import unit_speed_paths
from tensordev.sss import StateSpaceSignature
from tensordev.sss.rough_approx import fractional_fssk
from tensordev.volterra import ConvolutionKernel, vsig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_BETAS          = [0.1, 0.6]
_DEFAULT_R_VALUES       = [2, 3, 4, 6, 8, 10]
_DEFAULT_TRUNC          = 6
_DEFAULT_BATCH          = 4
_DEFAULT_STEPS          = 32
_DEFAULT_T              = 1.0
_DEFAULT_PC_DYADIC      = 9
_DEFAULT_COEF_QUAD      = 64
_DEFAULT_SEED           = 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSSK approximation sweep")
    p.add_argument("--betas", nargs="+", type=float, default=_DEFAULT_BETAS)
    p.add_argument("--R-values", nargs="+", type=int, default=_DEFAULT_R_VALUES)
    p.add_argument("--trunc", type=int, default=_DEFAULT_TRUNC)
    p.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    p.add_argument("--steps", type=int, default=_DEFAULT_STEPS)
    p.add_argument("--T", type=float, default=_DEFAULT_T)
    p.add_argument("--pc-dyadic-order", type=int, default=_DEFAULT_PC_DYADIC)
    p.add_argument("--coef-quad-order", type=int, default=_DEFAULT_COEF_QUAD)
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--dim", type=int, default=3)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    betas        = args.betas
    R_values     = args.R_values
    trunc        = args.trunc
    batch        = args.batch
    steps        = args.steps
    T            = args.T
    dim          = args.dim
    dtype        = jnp.float64
    dt           = T / steps

    # ── Path generation ──────────────────────────────────────────────────────
    print("Generating paths ...")
    X = jnp.asarray(
        unit_speed_paths(dt=dt, dt_fine=dt / 1024, n_paths=batch,
                         dim=dim, seed=args.seed, dtype=float),
        dtype=dtype,
    )
    print(f"  X.shape: {X.shape}")

    A = jnp.eye(dim, dtype=dtype)[None, :2, :]  # (1, 2, dim): project to m=2

    rows_fssk: list[dict] = []
    rows_vsig: list[dict] = []

    for beta in betas:
        if not (0.5 < beta < 1.0):
            print(f"\n[SKIP] β = {beta} outside (0.5, 1) — fractional_fssk requires positive Hurst index")
            continue

        print(f"\n{'═' * 60}")
        print(f"  β = {beta}")
        print(f"{'═' * 60}")

        exact_kernel = ConvolutionKernel.fractional(
            beta=jnp.array([beta], dtype=dtype), A=A,
        )

        # ── PC reference ────────────────────────────────────────────────────
        print(f"  Computing PC reference (dyadic_order={args.pc_dyadic_order}) ...",
              end=" ", flush=True)
        pc_target = vsig(
            X, kernel=exact_kernel, trunc=trunc, dt=dt, axis=1,
            increment_input=False, dyadic_order=args.pc_dyadic_order,
            scheme="adams", order=1,
        )
        jax.block_until_ready(pc_target)
        print("done.")

        # ── vsig baseline (order=2) ─────────────────────────────────────────
        vsig_out = vsig(
            X, kernel=exact_kernel, trunc=trunc, dt=dt, axis=1,
            increment_input=False, order=2,
        )
        jax.block_until_ready(vsig_out)
        for row in _level_errors(pc_target, vsig_out, batch):
            row["beta"] = beta
            rows_vsig.append(row)

        # ── FSSK R sweep ────────────────────────────────────────────────────
        for R in R_values:
            print(f"  R={R:3d} ...", end=" ", flush=True)
            fssk_kernel = fractional_fssk(
                beta=beta, R=R, A=A, T=T,
                coef_quad_order=args.coef_quad_order, dtype=dtype,
            )
            fssk_sig = StateSpaceSignature(kernel=fssk_kernel, trunc=trunc).vsig(
                X, dt=dt, axis=1, increment_input=False,
            )
            jax.block_until_ready(fssk_sig)
            for row in _level_errors(pc_target, fssk_sig, batch):
                row.update(beta=beta, R=R)
                rows_fssk.append(row)
            max_err = max(r["max_abs_entry"] for r in rows_fssk if r["beta"] == beta and r["R"] == R and r["level"] > 0)
            print(f"max err (lvl>0) = {max_err:.3e}")

    # ── Save ─────────────────────────────────────────────────────────────────
    df_fssk = pd.DataFrame(rows_fssk)
    df_vsig = pd.DataFrame(rows_vsig)
    params = dict(
        betas=betas, R_values=R_values, trunc=trunc, batch=batch,
        steps=steps, T=T, dim=dim, seed=args.seed,
        pc_dyadic_order=args.pc_dyadic_order,
        coef_quad_order=args.coef_quad_order,
    )

    out_pkl = args.output_dir / "fssk_approx.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"fssk": df_fssk, "vsig": df_vsig, "params": params}, f)

    print(f"\nSaved: {out_pkl}")
    print(f"  fssk rows: {len(df_fssk)}")
    print(f"  vsig rows: {len(df_vsig)}")


if __name__ == "__main__":
    main()
