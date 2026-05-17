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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # notebooks/

from tensordev.util.random_paths import unit_speed_paths
from tensordev.sss import StateSpaceSignature
from tensordev.sss.kernel import FSSK
from tensordev.volterra import ConvolutionKernel, vsig_fft


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_STEPS     = 65
_DEFAULT_BATCH     = 4
_DEFAULT_TRUNC     = 6
_DEFAULT_T         = 1.0
_DEFAULT_ORDERS    = [0, 1, 2]
_DEFAULT_SEED      = 0
_DEFAULT_TOLERANCE = 1e-4


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
    p.add_argument("--seed",     type=int,   default=_DEFAULT_SEED)
    p.add_argument("--tolerance",type=float, default=_DEFAULT_TOLERANCE)
    p.add_argument("--dim",      type=int,   default=3)
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
    A = jnp.eye(dim, dtype=dtype)[None, :, :]
    print(f"  X.shape: {X.shape}")

    # ── Build FSSK (exponential mixture kernel) ───────────────────────────────
    rates      = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=dtype)
    Lambda_mat = jnp.diag(rates)
    b          = jnp.ones((1, len(rates)), dtype=dtype)
    fssk       = FSSK.from_matrix(Lambda=Lambda_mat, A=A, b=b, quad_order=32)
    ck         = ConvolutionKernel.fssk(fssk)

    # ── Direct SSS reference ─────────────────────────────────────────────────
    print("Computing direct SSS reference ...")
    sss_out = StateSpaceSignature(kernel=fssk, trunc=trunc).vsig(
        X, dt=dt, axis=1, increment_input=False,
    )
    jax.block_until_ready(sss_out)
    print(f"  SSS level shapes: {[z.shape for z in sss_out]}")

    # ── Sweep vsig_fft orders ────────────────────────────────────────────────
    rows = []
    for order in args.orders:
        print(f"  vsig_fft order={order} ...", end=" ", flush=True)
        fft_out = vsig_fft(
            X, kernel=ck, dt=dt, trunc=trunc, axis=1,
            increment_input=False, order=order,
        )
        jax.block_until_ready(fft_out)
        for level, (a, b_) in enumerate(zip(sss_out, fft_out)):
            fscale = math.factorial(level)
            diff = (jnp.asarray(b_).reshape((batch, -1))
                    - jnp.asarray(a).reshape((batch, -1))) * fscale
            max_diff = float(jnp.max(jnp.abs(diff)))
            rows.append({"order": order, "level": level, "max_abs_diff": max_diff})
        max_all = max(r["max_abs_diff"] for r in rows if r["order"] == order and r["level"] > 0)
        print(f"max diff (lvl>0) = {max_all:.3e}")

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="level", columns="order", values="max_abs_diff")
    print(f"\nPer-level max abs difference (n!-scaled, steps={steps}):")
    print(pivot.to_string())

    failed = df[(df["level"] > 0) & (df["max_abs_diff"] > args.tolerance)]
    if not failed.empty:
        print(f"\n[FAIL] {len(failed)} entries exceed tolerance={args.tolerance}:")
        print(failed.to_string(index=False))
        sys.exit(1)
    else:
        print(f"\n[PASS] All differences ≤ tolerance={args.tolerance}")


if __name__ == "__main__":
    main()
