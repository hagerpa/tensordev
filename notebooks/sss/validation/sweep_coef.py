"""
sweep_coef.py — XLA FLOP sweep for FSSK coefficient computation.

Profiles sss.kernel.coef(dt, trunc=N) for both dense and Jordan-form Lambda
over (q, R, N) configurations.  The dimension m plays no role in coefficient
computation and is fixed at m=3.

Predicted work (Prop. cost_fssk_weights in the paper):
    Dense  Lambda:  W = R³ + R²·N^q
    Jordan Lambda:  W = R²·N^q

Output (written to --output-dir):
    fssk_coef_scaling_{regime}.pkl
    fssk_coef_scaling_{regime}.csv

Usage
-----
    python sweep_coef.py --regime MEDIUM
    python sweep_coef.py --regime SMALL --dry-run
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/
sys.path.insert(0, str(Path(__file__).resolve().parent))     # validation/

from tensordev.sss import StateSpaceSignature

from _validation_util.xla_utils import compile_and_profile

from fssk_setup import random_fssk


# ---------------------------------------------------------------------------
# Regime designs  (q, R, N sweep — J, m, d are irrelevant for coef cost)
# ---------------------------------------------------------------------------

_DESIGNS = {
    "SMALL": dict(
        q_vals=[1, 2, 3],
        R_range=(2, 8),
        N_range=(3, 7),
        n_rn=30,
        seed=20260601,
    ),
    "MEDIUM": dict(
        q_vals=[1, 2, 3, 4],
        R_range=(2, 12),
        N_range=(3, 11),
        n_rn=75,
        seed=20260601,
    ),
    "LARGE": dict(
        q_vals=[1, 2, 3, 4, 5, 6],
        R_range=(2, 20),
        N_range=(3, 16),
        n_rn=300,
        seed=20260601,
    ),
}

_FIXED_M = 3  # latent/path dim — does not affect coef computation
_FIXED_D = 3


def _sample_design(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n_rn (R, N) pairs uniformly, then cross with q_vals and forms."""
    R_lo, R_hi = cfg["R_range"]
    N_lo, N_hi = cfg["N_range"]

    R_grid = np.arange(R_lo, R_hi + 1)
    N_grid = np.arange(N_lo, N_hi + 1)
    pairs = np.array([(r, n) for r in R_grid for n in N_grid])
    idx   = rng.choice(len(pairs), size=min(cfg["n_rn"], len(pairs)), replace=False)
    sampled = pd.DataFrame(pairs[idx], columns=["R", "N"])

    rows = []
    for _, row in sampled.iterrows():
        for q in cfg["q_vals"]:
            for form in ["dense", "jordan"]:
                rows.append({"q": int(q), "R": int(row["R"]), "N": int(row["N"]), "form": form})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSSK coefficient FLOP-scaling sweep")
    p.add_argument("--regime", choices=["SMALL", "MEDIUM", "LARGE"], default="MEDIUM")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print design and exit without profiling")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = _DESIGNS[args.regime]
    rng = np.random.default_rng(cfg["seed"])

    df_design = _sample_design(cfg, rng)

    print(f"Regime     : {args.regime}")
    print(f"Total rows : {len(df_design)}")
    print(f"q values   : {sorted(df_design['q'].unique())}")
    print(f"R range    : {df_design['R'].min()} – {df_design['R'].max()}")
    print(f"N range    : {df_design['N'].min()} – {df_design['N'].max()}")
    print(f"Forms      : {sorted(df_design['form'].unique())}")
    print()

    if args.dry_run:
        print("Dry-run — exiting without profiling.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime_tag = args.regime.lower()
    out_pkl = args.output_dir / f"fssk_coef_scaling_{regime_tag}.pkl"
    out_csv = args.output_dir / f"fssk_coef_scaling_{regime_tag}.csv"

    dt_spec  = jax.ShapeDtypeStruct((), jnp.float64)
    seed0    = cfg["seed"]
    rows     = []

    for idx, row in df_design.iterrows():
        q    = int(row["q"])
        R    = int(row["R"])
        N    = int(row["N"])
        form = row["form"]
        as_dense = (form == "dense")

        print(
            f"[{idx + 1:04d}/{len(df_design)}]  "
            f"form={form:6s}  q={q}  R={R:2d}  N={N:2d}",
            flush=True,
        )

        fssk = random_fssk(
            q=q, R=R, m=_FIXED_M, d=_FIXED_D,
            seed=seed0 + idx,
            eig_min=0.1, eig_max=1.5,
            freq_min=0.1, freq_max=2.0,
            normalise_b=False,
            as_dense=as_dense,
        )
        sss = StateSpaceSignature(kernel=fssk, trunc=N)

        def coef_fn(dt_arg):
            return sss.kernel.coef(dt_arg, trunc=N, dtype=jnp.float64)

        profile = compile_and_profile(coef_fn, dt_spec)

        rows.append(dict(
            run_id=idx,
            regime=args.regime,
            form=form,
            q=q,
            R=R,
            N=N,
            xla_flops=profile.get("xla_flops"),
            xla_bytes_accessed=profile.get("xla_bytes_accessed"),
            xla_lower_time_s=profile.get("xla_lower_time_s"),
            xla_compile_time_s=profile.get("xla_compile_time_s"),
        ))

        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_pickle(out_pkl)
            print(f"  checkpoint → {out_pkl.name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")


if __name__ == "__main__":
    main()