"""
validate_lambda0.py — Lambda=0 correctness check for StateSpaceSignature.

Checks the identity:

    SSS_{Lambda=0}(X) == Sig(Y),   Y_t = [sum_p sum_r b[p,r] A[p]] X_t

for q = 1 and q > 1 over truncation levels N = 1 .. N_max.

Usage
-----
    python validate_lambda0.py
"""

import sys
from pathlib import Path

# Make tensordev importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import math
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tensordev import get_default_core
from tensordev.sss import StateSpaceSignature
from tensordev.development import path_signature
from tensordev.util.random_paths import unit_speed_paths

from helpers import random_fssk

JaxCore = get_default_core()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def transformed_lambda0_path(X, A, b):
    """Compute Y_t = [sum_p (sum_r b[p,r]) A[p]] X_t.

    Parameters
    ----------
    X : (B, L, d)
    A : (q, m, d)
    b : (q, R)

    Returns
    -------
    Y : (B, L, m)
    """
    c = jnp.sum(b, axis=1)               # (q,)
    M = jnp.einsum("p,pmd->md", c, A)    # (m, d)
    return jnp.einsum("md,bld->blm", M, X)


def compare_levelwise(sig_a, sig_b):
    """Return per-level error statistics between two signature tuples."""
    rows = []
    for n, (Sa, Sb) in enumerate(zip(sig_a, sig_b)):
        err = np.asarray(jnp.abs(Sa - Sb))
        fac = math.factorial(n)
        rows.append({
            "level":        n,
            "mean_abs":     float(err.mean()),
            "max_abs":      float(err.max()),
            "mean_scaled":  float(fac * err.mean()),
            "max_scaled":   float(fac * err.max()),
        })
    return rows


def lambda0_validation_case(
    *,
    q,
    R,
    m,
    d,
    N_max=10,
    n_paths=8,
    dt=1 / 64,
    dt_fine=1 / 4096,
    seed=0,
):
    """Run the Lambda=0 validation for one (q, R, m, d) configuration."""
    _, A_np, b_np = random_fssk(q=q, R=R, m=m, d=d, seed=seed)

    Lambda = jnp.zeros((R, R), dtype=jnp.float64)
    A = jnp.asarray(A_np, dtype=jnp.float64)
    b = jnp.asarray(b_np, dtype=jnp.float64)

    X = jnp.asarray(
        unit_speed_paths(
            dt=dt,
            dt_fine=dt_fine,
            n_paths=n_paths,
            dim=d,
            seed=10_000 + seed,
            dtype=np.float64,
        )
    )

    Y = transformed_lambda0_path(X, A, b)

    rows = []
    for N in range(1, N_max + 1):
        sss = StateSpaceSignature.from_matrix(Lambda=Lambda, A=A, b=b, trunc=N)
        S_sss = sss.vsig(X, dt=dt, axis=-2)
        S_sig = path_signature(Y, trunc=N, axis=-2)

        for row in compare_levelwise(S_sss, S_sig):
            row.update({"q": q, "R": R, "m": m, "d": d, "N": N})
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df_lam0 = pd.concat(
        [
            lambda0_validation_case(q=1, R=1, m=2, d=2, N_max=10, seed=0),
            lambda0_validation_case(q=3, R=2, m=2, d=2, N_max=10, seed=1),
            lambda0_validation_case(q=3, R=3, m=3, d=3, N_max=8,  seed=2),
        ],
        ignore_index=True,
    )

    df_lam0_summary = (
        df_lam0
        .groupby(["q", "R", "m", "d", "N"], as_index=False)
        .agg(
            max_abs=("max_abs",     "max"),
            max_scaled=("max_scaled", "max"),
            mean_abs=("mean_abs",   "max"),
            mean_scaled=("mean_scaled", "max"),
        )
    )

    print(df_lam0_summary.to_string(index=False))
    print()
    print("Overall max_abs:   ", df_lam0["max_abs"].max())
    print("Overall max_scaled:", df_lam0["max_scaled"].max())
    print("float64 eps:       ", np.finfo(np.float64).eps)


if __name__ == "__main__":
    main()
