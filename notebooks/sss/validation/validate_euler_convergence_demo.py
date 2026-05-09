"""
validate_euler_convergence_demo.py
===================================
Clean convergence-only demo: Euler dyadic refinement → exact FSSK.

No timing, no XLA profiling.  For each kernel setup the script computes the
exact FSSK signature once and then sweeps dyadic_order 0..max_dyadic,
recording the max and mean absolute error at every level.  A linear
regression on the log2 errors gives the per-setup convergence rate.

Outputs (written to --output-dir):
  euler_conv_demo_errors.csv   – raw errors (one row per setup × dyadic)
  euler_conv_demo_rates.csv    – fitted convergence rates (one row per setup)
  euler_conv_demo_plot.pdf     – log-scale error vs dyadic_order

Usage
-----
    python validate_euler_convergence_demo.py                    # defaults
    python validate_euler_convergence_demo.py --J 64 --N 8      # override grid/trunc
    python validate_euler_convergence_demo.py --output-dir /tmp/out
    python validate_euler_convergence_demo.py \\
        --setups '[{"q":1,"R":2,"seed":42},{"q":3,"R":3,"seed":7}]'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import factorial
from pathlib import Path
import jax
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # validation/

import plot_config  # noqa: F401 — applies rcParams; must come after matplotlib.use()
from plot_config import new_fig, savefig_fig, COLORS, MARKERS

from tensordev.sss import StateSpaceSignature
from euler_general import fssk_euler_vsig
from helpers import random_fssk

# ---------------------------------------------------------------------------
# Default experimental design (overridable via CLI)
# ---------------------------------------------------------------------------

_DEFAULT_J = 32
_DEFAULT_N = 7
_DEFAULT_M = 2
_DEFAULT_D = 2
_DEFAULT_N_PATHS = 2
_DEFAULT_MAX_DYADIC = 7
_DEFAULT_PATH_SEED = 20226

_DEFAULT_SETUPS = [
    dict(label="q=1, R=2", q=1, R=2, seed=4221),
    dict(label="q=2, R=2", q=2, R=2, seed=4313),
    dict(label="q=4, R=3", q=4, R=3, seed=4133),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brownian_paths(n_paths: int, J: int, d: int, seed: int, dt: float) -> np.ndarray:
    """Return iid Brownian paths of shape (n_paths, J, d)."""
    rng = np.random.default_rng(seed)
    dX = rng.normal(0.0, np.sqrt(dt), size=(n_paths, J - 1, d))
    X = np.concatenate([np.zeros((n_paths, 1, d)), np.cumsum(dX, axis=1)], axis=1)
    return X.astype(np.float64)


def _per_level_scaled_errors(exact: tuple, approx: tuple) -> np.ndarray:
    """Max error over paths per signature level, scaled by level!

    For level ℓ (1-indexed) returns:
        ℓ! · max_{paths, entries} |S_exact^ℓ − S_euler^ℓ|

    Skips level 0 (the constant term, always 1).
    """
    errors = []
    for k, (e, a) in enumerate(zip(exact, approx)):
        if k == 0:
            continue
        diff = np.abs(np.asarray(e) - np.asarray(a))
        errors.append(float(diff.max()) * factorial(k))
    return np.array(errors)


def _fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit log2(y) = slope * x + intercept via OLS; returns (slope, intercept)."""
    log_y = np.log2(np.where(y > 0, y, np.nan))
    valid = np.isfinite(log_y)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    coeffs = np.polyfit(x[valid], log_y[valid], 1)
    return float(coeffs[0]), float(coeffs[1])


def _label_to_slug(label: str) -> str:
    """Convert a setup label to a safe filename component."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Euler convergence demo")
    p.add_argument("--J", type=int, default=_DEFAULT_J, help="Path grid points (J-1 increments)")
    p.add_argument("--N", type=int, default=_DEFAULT_N, help="Signature truncation level")
    p.add_argument("--m", type=int, default=_DEFAULT_M, help="Latent-path dimension")
    p.add_argument("--d", type=int, default=_DEFAULT_D, help="Input-path dimension")
    p.add_argument("--n-paths", type=int, default=_DEFAULT_N_PATHS, help="Batch size (number of paths)")
    p.add_argument("--max-dyadic", type=int, default=_DEFAULT_MAX_DYADIC, help="Sweep dyadic orders 0..max-dyadic")
    p.add_argument("--path-seed", type=int, default=_DEFAULT_PATH_SEED, help="RNG seed for path generation")
    p.add_argument(
        "--setups",
        type=str,
        default=None,
        help=(
            'JSON list of setup dicts, e.g. \'[{"q":1,"R":2,"seed":42},...]\'. '
            "Each dict may include an optional \"label\" key."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "validation_outputs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve parameters
    J = args.J
    N = args.N
    m = args.m
    d = args.d
    n_paths = args.n_paths
    dt = 1.0 / (J - 1)

    DYADIC_ORDERS = list(range(args.max_dyadic + 1))
    x_dyadic = np.array(DYADIC_ORDERS, dtype=float)

    if args.setups is not None:
        raw = json.loads(args.setups)
        SETUPS = []
        for s in raw:
            q, R, seed = s["q"], s["R"], s.get("seed", 0)
            label = s.get("label", f"q={q}, R={R}")
            SETUPS.append(dict(label=label, q=q, R=R, seed=seed))
    else:
        SETUPS = _DEFAULT_SETUPS

    X = _brownian_paths(n_paths=n_paths, J=J, d=d, seed=args.path_seed, dt=dt)

    # error_rows: one row per (setup, dyadic_order, level)
    error_rows: list[dict] = []
    rate_rows: list[dict] = []
    plot_stems: list[Path] = []

    for setup in SETUPS:
        label = setup["label"]
        q = setup["q"]
        R = setup["R"]
        seed = setup["seed"]

        print(f"\n── {label} ──────────────────────────────")

        # Build kernel
        Lambda_np, A_np, b_np = random_fssk(q=q, R=R, m=m, d=d, seed=seed, eig_min=1.0, eig_max=2.0)
        sss = StateSpaceSignature.from_matrix(Lambda=Lambda_np, A=A_np, b=b_np, trunc=N)

        # Exact signature (computed once per setup)
        S_exact = sss.vsig(X, dt=dt, axis=-2)
        n_levels = len(S_exact) - 1  # skip level-0 constant term

        # per_level_errors[dyadic, level-1]
        per_level_errors = np.full((len(DYADIC_ORDERS), n_levels), np.nan)

        for di, dyadic in enumerate(DYADIC_ORDERS):
            S_euler = fssk_euler_vsig(
                X, kernel=sss.kernel, dt=dt, trunc=N, axis=-2, dyadic_order=dyadic
            )
            errs = _per_level_scaled_errors(S_exact, S_euler)
            per_level_errors[di] = errs
            print(f"  dyadic={dyadic}  " +
                  "  ".join(f"lvl{k + 1}={errs[k]:.2e}" for k in range(n_levels)))

            for k, err in enumerate(errs):
                error_rows.append(dict(
                    label=label, q=q, R=R,
                    dyadic_order=dyadic,
                    level=k + 1,
                    scaled_max_error=err,
                ))

        # Fit slope per level
        for k in range(n_levels):
            slope, intercept = _fit_slope(x_dyadic, per_level_errors[:, k])
            print(f"  level {k + 1} slope: {slope:.3f}")
            rate_rows.append(dict(
                label=label, q=q, R=R,
                level=k + 1,
                slope=slope,
                intercept_log2=intercept,
            ))

        # ── Per-setup plot ─────────────────────────────────────────────────
        fig, ax = new_fig("half")

        # Sequential colormap (viridis) for levels — avoids harsh categorical jumps
        cmap = plt.cm.viridis
        level_colors = [cmap(0.1 + 0.80 * k / max(n_levels - 1, 1)) for k in range(n_levels)]

        for k in range(n_levels):
            col = level_colors[k]
            ax.semilogy(
                x_dyadic, per_level_errors[:, k],
                marker=MARKERS[k % len(MARKERS)],
                color=col,
                label=fr"$n={k + 1}$",
            )
            # Fitted dashed line
            r = next((r for r in rate_rows
                      if r["label"] == label and r["level"] == k + 1), None)
            if r and np.isfinite(r["slope"]):
                y_fit = 2.0 ** (r["intercept_log2"] + r["slope"] * x_dyadic)
                ax.semilogy(x_dyadic, y_fit, color=col, linestyle="--",
                            linewidth=0.8, alpha=0.55)

        ax.set_xlabel("dyadic order $p$")
        ax.set_ylabel(r"$\delta V_{n,p}$")
        ax.set_title(
            fr"FSSK - Euler convergence  $q={q},\,R={R}$")  # fr"($J={J},\,N={N},\,m={m},\,d={d}$)"        ax.set_xticks(DYADIC_ORDERS)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(ncol=min(n_levels, 4), loc="upper right")

        slug = _label_to_slug(label)
        stem = args.output_dir / f"euler_conv_demo_{slug}"
        plot_stems.append(stem)
        savefig_fig(fig, stem, ["pdf", "png"])
        plt.close(fig)

    # ── Save CSVs ──────────────────────────────────────────────────────────
    df_errors = pd.DataFrame(error_rows)
    df_rates = pd.DataFrame(rate_rows)

    path_errors = args.output_dir / "euler_conv_demo_errors.csv"
    path_rates = args.output_dir / "euler_conv_demo_rates.csv"

    df_errors.to_csv(path_errors, index=False)
    df_rates.to_csv(path_rates, index=False)
    print(f"\nSaved:\n  {path_errors}\n  {path_rates}")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n── Convergence rates (slope per setup × level) ──────────────────")
    print(df_rates[["label", "level", "slope"]].to_string(index=False))


if __name__ == "__main__":
    main()
