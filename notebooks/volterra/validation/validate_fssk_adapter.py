"""
validate_fssk_adapter.py — Correctness check: SSS vs vsig_fft through FSSKConvolutionKernel.

Compares two code paths for the same FSSK kernel on the same grid:
  1. Direct SSS — StateSpaceSignature.vsig (reference)
  2. vsig_fft through ConvolutionKernel.fssk (adapter)

Prints a per-level, per-order table of max absolute differences (n!-scaled).
Exits with code 1 if any difference exceeds --tolerance.

Paths: 3D unit-speed paths (dim fixed to 3).

Usage
-----
    python validate_fssk_adapter.py
    python validate_fssk_adapter.py --steps 65 --batch 4 --orders 0 1 2
    python validate_fssk_adapter.py --tolerance 1e-6
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd

jax.config.update("jax_enable_x64", True)

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[3] / "src"))          # tensordev/src
sys.path.insert(0, str(_here.parents[2]))                  # notebooks/
sys.path.insert(0, str(_here.parents[2] / "sss" / "validation"))  # fssk_setup

from tensordev.util.random_paths import unit_speed_paths
from tensordev.sss import StateSpaceSignature
from tensordev.volterra import ConvolutionKernel, vsig
from fssk_setup import random_fssk


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_STEPS          = 65
_DEFAULT_BATCH          = 4
_DEFAULT_TRUNC          = 6
_DEFAULT_T              = 1.0
_DEFAULT_ORDERS         = [0, 1, 2]
_DEFAULT_SCHEMES        = ["quadratic"]
_DEFAULT_SEED           = 0
_DEFAULT_TOLERANCE      = 1e-4
_DEFAULT_EIG_MIN        = 0.1
_DEFAULT_EIG_MAX        = 1.0
_DEFAULT_FREQ_MIN       = 0.1
_DEFAULT_FREQ_MAX       = 2.0
_DEFAULT_MIN_BLOCK_SIZE = 1
_DEFAULT_MAX_BLOCK_SIZE = 4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Validate FSSKConvolutionKernel adapter")
    p.add_argument("--steps",    type=int,   default=_DEFAULT_STEPS)
    p.add_argument("--batch",    type=int,   default=_DEFAULT_BATCH)
    p.add_argument("--trunc",    type=int,   default=_DEFAULT_TRUNC)
    p.add_argument("--T",        type=float, default=_DEFAULT_T)
    p.add_argument("--orders",   nargs="+",  type=int, default=_DEFAULT_ORDERS)
    p.add_argument("--schemes",  nargs="+",  type=str, default=_DEFAULT_SCHEMES,
                   help='vsig integration schemes to sweep. Default: quadratic.')
    p.add_argument("--seed",     type=int,   default=_DEFAULT_SEED)
    p.add_argument("--dim",      type=int,   default=3)
    p.add_argument("--q",        type=int,   default=3,
                   help="Number of kernel components (kernel.q = A.shape[0]). Default: 3.")
    p.add_argument("--R",        type=int,   default=4,
                   help="FSSK state dimension. Default: 4.")
    p.add_argument("--m",            type=int,   default=None,
                   help="Volterra path dimension (A.shape[1]). Defaults to --dim.")
    # Jordan block parameters
    p.add_argument("--eig-min",        type=float, default=_DEFAULT_EIG_MIN,
                   help="Min real part (decay) of eigenvalues. Default: 0.1.")
    p.add_argument("--eig-max",        type=float, default=_DEFAULT_EIG_MAX,
                   help="Max real part (decay) of eigenvalues. Default: 1.0.")
    p.add_argument("--freq-min",       type=float, default=_DEFAULT_FREQ_MIN,
                   help="Min imaginary part (frequency) of oscillatory eigenvalues. Default: 0.1.")
    p.add_argument("--freq-max",       type=float, default=_DEFAULT_FREQ_MAX,
                   help="Max imaginary part (frequency) of oscillatory eigenvalues. Default: 2.0.")
    p.add_argument("--min-block-size", type=int,   default=_DEFAULT_MIN_BLOCK_SIZE,
                   help="Minimum Jordan chain order. Default: 1.")
    p.add_argument("--max-block-size", type=int,   default=_DEFAULT_MAX_BLOCK_SIZE,
                   help="Maximum Jordan chain order. Default: 4.")
    p.add_argument("--as-dense",       action="store_true", default=False,
                   help="Materialise Jordan structure into a DenseLambda kernel.")
    p.add_argument("--dyadic-orders", nargs="+", type=int, default=[0],
                   help="Dyadic refinement orders to sweep. Default: 0.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to write fssk_adapter_errors.csv. Skipped if not set.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dtype  = jnp.float64
    steps  = args.steps
    batch  = args.batch
    trunc  = args.trunc
    T      = args.T
    dim    = args.dim
    dt     = T / steps

    # ── Paths ────────────────────────────────────────────────────────────────
    print("Generating paths ...")
    X = jnp.asarray(
        unit_speed_paths(dt=dt, dt_fine=dt / 1024, n_paths=batch,
                         dim=dim, seed=args.seed, dtype=float),
        dtype=dtype,
    )
    m = args.m if args.m is not None else dim
    print(f"  X.shape: {X.shape}")

    # ── Build random FSSK kernel (q components, state dim R) ─────────────────
    fssk = random_fssk(
        q=args.q, R=args.R, m=m, d=dim, seed=args.seed,
        eig_min=args.eig_min, eig_max=args.eig_max,
        freq_min=args.freq_min, freq_max=args.freq_max,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        as_dense=args.as_dense,
    )
    fssk = type(fssk)(Lambda=fssk.Lambda,
                      A=jnp.asarray(fssk.A, dtype=dtype),
                      b=jnp.asarray(fssk.b, dtype=dtype),
                      quad_order=fssk.quad_order)
    ck   = ConvolutionKernel.fssk(fssk)
    print(f"  kernel.q={ck.q}  kernel.m={ck.m}  R={ck.state_dim}")

    # ── Direct SSS reference ─────────────────────────────────────────────────
    print("Computing direct SSS reference ...")
    sss_out = StateSpaceSignature(kernel=fssk, trunc=trunc).vsig(
        X, dt=dt, axis=1, increment_input=False,
    )
    jax.block_until_ready(sss_out)
    print(f"  SSS level shapes: {[z.shape for z in sss_out]}")

    # ── Sweep vsig_fft orders × dyadic orders ───────────────────────────────
    rows = []
    for dyadic_order in args.dyadic_orders:
        for scheme in args.schemes:
            for order in args.orders:
                print(f"  vsig_fft order={order} scheme={scheme} dyadic={dyadic_order} ...", end=" ", flush=True)
                fft_out = vsig(
                    X, kernel=ck, dt=dt, trunc=trunc, axis=1,
                    increment_input=False, order=order, scheme=scheme,
                    dyadic_order=dyadic_order,
                )
                jax.block_until_ready(fft_out)
                for level, (a, b_) in enumerate(zip(sss_out, fft_out)):
                    fscale = math.factorial(level)
                    diff = (jnp.asarray(b_).reshape((batch, -1))
                            - jnp.asarray(a).reshape((batch, -1))) * fscale
                    max_diff = float(jnp.max(jnp.abs(diff)))
                    rows.append({"dyadic_order": dyadic_order, "scheme": scheme,
                                 "order": order, "level": level, "max_abs_diff": max_diff})
                max_all = max(r["max_abs_diff"] for r in rows
                              if r["dyadic_order"] == dyadic_order
                              and r["scheme"] == scheme
                              and r["order"] == order and r["level"] > 0)
                print(f"max diff (lvl>0) = {max_all:.3e}")

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="level", columns=["dyadic_order", "scheme", "order"], values="max_abs_diff")
    print(f"\nPer-level max abs difference (n!-scaled, steps={steps}):")
    print(pivot.to_string())

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_csv = args.output_dir / "fssk_adapter_errors.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
