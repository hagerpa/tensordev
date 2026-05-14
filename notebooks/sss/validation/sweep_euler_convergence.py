"""
sweep_euler_convergence.py — Compute Euler convergence errors for FSSK.

For each setup (n, R) sweeps dyadic_order 0..max_dyadic and records
    ℓ! · max_{paths,entries} |S_exact^ℓ − S_euler^ℓ|
for each signature level ℓ.

Output (written to --output-dir):
    euler_conv_sweep.pkl   — raw error DataFrame (one row per setup × dyadic × level)

Usage
-----
    python sweep_euler_convergence.py
    python sweep_euler_convergence.py --J 64 --N 8
    python sweep_euler_convergence.py --setups '[{"n":1,"R":2,"seed":42},...]'
"""

from __future__ import annotations

import argparse
import json
import sys
from math import factorial
from pathlib import Path

import jax
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/
sys.path.insert(0, str(Path(__file__).resolve().parent))     # validation/

from tensordev.sss import StateSpaceSignature
from tensordev.util.random_paths import unit_speed_paths
from euler_general import fssk_euler_vsig
from fssk_setup import random_fssk


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_J         = 32
_DEFAULT_N         = 7
_DEFAULT_M         = 2
_DEFAULT_N_PATHS   = 2
_DEFAULT_MAX_DYADIC = 7
_DEFAULT_PATH_SEED = 20226

_DEFAULT_SETUPS = [
    dict(label="n=1, R=2", n=1, R=2, seed=4221),
    dict(label="n=2, R=2", n=2, R=2, seed=4313),
    dict(label="n=4, R=3", n=4, R=3, seed=4133),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_level_scaled_errors(exact: tuple, approx: tuple) -> np.ndarray:
    """ℓ! · max|S_exact^ℓ − S_euler^ℓ| for each level ℓ ≥ 1 (skips level 0)."""
    errors = []
    for k, (e, a) in enumerate(zip(exact, approx)):
        if k == 0:
            continue
        diff = np.abs(np.asarray(e) - np.asarray(a))
        errors.append(float(diff.max()) * factorial(k))
    return np.array(errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Euler convergence sweep")
    p.add_argument("--J",          type=int,  default=_DEFAULT_J)
    p.add_argument("--N",          type=int,  default=_DEFAULT_N)
    p.add_argument("--m",          type=int,  default=_DEFAULT_M)
    p.add_argument("--n-paths",    type=int,  default=_DEFAULT_N_PATHS)
    p.add_argument("--max-dyadic", type=int,  default=_DEFAULT_MAX_DYADIC)
    p.add_argument("--path-seed",  type=int,  default=_DEFAULT_PATH_SEED)
    p.add_argument("--d",          type=int,  default=3,
                   help="Ignored; paths are always 3D unit-speed")
    p.add_argument("--setups", type=str, default=None,
                   help='JSON list, e.g. \'[{"n":1,"R":2,"seed":42},...]\'. '
                        'Optional "label" key per entry.')
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    J       = args.J
    N       = args.N
    m       = args.m
    d       = 3
    n_paths = args.n_paths
    dt      = 1.0 / (J - 1)
    DYADIC_ORDERS = list(range(args.max_dyadic + 1))

    if args.setups is not None:
        raw = json.loads(args.setups)
        SETUPS = [
            dict(
                label=s.get("label", f"n={s['n']}, R={s['R']}"),
                n=s["n"], R=s["R"], seed=s.get("seed", 0),
            )
            for s in raw
        ]
    else:
        SETUPS = _DEFAULT_SETUPS

    X = unit_speed_paths(dt=dt, dt_fine=dt / 8, n_paths=n_paths, dim=3, seed=args.path_seed)

    error_rows: list[dict] = []

    for setup in SETUPS:
        label = setup["label"]
        q     = setup["n"]
        R     = setup["R"]
        seed  = setup["seed"]

        print(f"\n── {label} ──────────────────────────────")

        Lambda_np, A_np, b_np = random_fssk(
            q=q, R=R, m=m, d=d, seed=seed, eig_min=0.0, eig_max=1.0
        )
        sss = StateSpaceSignature.from_matrix(Lambda=Lambda_np, A=A_np, b=b_np, trunc=N)

        S_exact  = sss.vsig(X, dt=dt, axis=-2)
        n_levels = len(S_exact) - 1  # exclude level-0 constant term

        for dyadic in DYADIC_ORDERS:
            S_euler = fssk_euler_vsig(
                X, kernel=sss.kernel, dt=dt, trunc=N, axis=-2, dyadic_order=dyadic
            )
            errs = _per_level_scaled_errors(S_exact, S_euler)
            print(
                f"  dyadic={dyadic}  "
                + "  ".join(f"lvl{k+1}={errs[k]:.2e}" for k in range(n_levels))
            )

            for k, err in enumerate(errs):
                error_rows.append(dict(
                    label=label, q=q, R=R, seed=seed,
                    J=J, N=N, m=m, n_paths=n_paths,
                    dyadic_order=dyadic,
                    level=k + 1,
                    scaled_max_error=err,
                ))

    df = pd.DataFrame(error_rows)
    out_pkl = args.output_dir / "euler_conv_sweep.pkl"
    df.to_pickle(out_pkl)
    print(f"\nSaved: {out_pkl}  ({len(df)} rows)")


if __name__ == "__main__":
    main()