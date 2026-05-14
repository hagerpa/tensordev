"""
analyse_euler_convergence.py — Fit slopes and plot Euler convergence results.

Reads the pickle produced by sweep_euler_convergence.py, fits log2 convergence
rates per setup × level, writes summary CSVs, and generates per-setup plots.

Outputs (written to --output-dir):
    euler_conv_demo_errors.csv        — raw errors (one row per setup × dyadic × level)
    euler_conv_demo_rates.csv         — fitted convergence rates (one row per setup × level)
    euler_conv_demo_{slug}.{png,pdf}  — per-setup log-scale error plots

Usage
-----
    python analyse_euler_convergence.py
    python analyse_euler_convergence.py --input path/to/euler_conv_sweep.pkl
    python analyse_euler_convergence.py --no-plots
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

import _validation_util.plot_config as plot_config  # noqa: F401 — applies rcParams; must come after matplotlib.use()
from _validation_util.plot_config import new_fig, savefig_fig, MARKERS
from _validation_util.analysis_utils import fit_slope


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse Euler convergence sweep results")
    p.add_argument("--input", type=Path, default=None,
                   help="Pickle from sweep_euler_convergence.py "
                        "(default: <output-dir>/euler_conv_sweep.pkl)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_to_slug(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    pkl_path = args.input or (args.output_dir / "euler_conv_sweep.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Input not found: {pkl_path}\n"
            "Run sweep_euler_convergence.py first, or pass --input explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(pkl_path)

    rate_rows: list[dict] = []

    for label, g_label in df.groupby("label", sort=False):
        q = int(g_label["q"].iloc[0])
        R = int(g_label["R"].iloc[0])
        dyadic_orders = sorted(g_label["dyadic_order"].unique())
        levels        = sorted(g_label["level"].unique())
        x_dyadic      = np.array(dyadic_orders, dtype=float)
        n_levels      = len(levels)

        # Reconstruct per_level_errors array
        per_level_errors = np.full((len(dyadic_orders), n_levels), np.nan)
        for di, dyadic in enumerate(dyadic_orders):
            for ki, level in enumerate(levels):
                mask = (g_label["dyadic_order"] == dyadic) & (g_label["level"] == level)
                vals = g_label.loc[mask, "scaled_max_error"]
                if len(vals):
                    per_level_errors[di, ki] = float(vals.iloc[0])

        # Fit slopes
        for ki, level in enumerate(levels):
            slope, intercept = fit_slope(x_dyadic, per_level_errors[:, ki])
            print(f"  {label}  level {level} slope: {slope:.3f}")
            rate_rows.append(dict(
                label=label, q=q, R=R,
                level=level,
                slope=slope,
                intercept_log2=intercept,
            ))

        if args.no_plots:
            continue

        fig, ax = new_fig("half")
        cmap = plt.cm.viridis
        level_colors = [cmap(0.1 + 0.80 * ki / max(n_levels - 1, 1)) for ki in range(n_levels)]

        for ki, level in enumerate(levels):
            col = level_colors[ki]
            ax.semilogy(
                x_dyadic, per_level_errors[:, ki],
                marker=MARKERS[ki % len(MARKERS)],
                color=col,
                label=fr"$n={level}$",
            )
            r = next((r for r in rate_rows if r["label"] == label and r["level"] == level), None)
            if r and np.isfinite(r["slope"]):
                y_fit = 2.0 ** (r["intercept_log2"] + r["slope"] * x_dyadic)
                ax.semilogy(x_dyadic, y_fit, color=col, linestyle="--",
                            linewidth=0.8, alpha=0.55)

        ax.set_xlabel("dyadic order $p$")
        ax.set_ylabel(r"$\delta V_{n,p}$")
        ax.set_title(fr"FSSK - Euler convergence  $q={q},\,R={R}$")
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(ncol=min(n_levels, 4), loc="upper right")

        slug = _label_to_slug(label)
        stem = args.output_dir / f"euler_conv_demo_{slug}"
        savefig_fig(fig, stem, args.formats)
        plt.close(fig)

    df_rates = pd.DataFrame(rate_rows)
    path_errors = args.output_dir / "euler_conv_demo_errors.csv"
    path_rates  = args.output_dir / "euler_conv_demo_rates.csv"
    df.to_csv(path_errors, index=False)
    df_rates.to_csv(path_rates, index=False)
    print(f"\nSaved:\n  {path_errors}\n  {path_rates}")

    print("\n── Convergence rates (slope per setup × level) ──────────────────")
    print(df_rates[["label", "level", "slope"]].to_string(index=False))


if __name__ == "__main__":
    main()