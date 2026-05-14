"""
analyse_timing.py — Plot wall-clock and CPU timing results vs. predicted work W_q.

Reads the pickle produced by sweep_timing.py (via fssk_setup.main()), computes
the theoretical work proxy W_q per row, generates a log-log scatter plot, and
writes a per-n summary CSV.

Outputs (written to --output-dir):
    fssk_wall_vs_work_{regime}.{png,pdf}   — log-log scatter: W_q vs wall time
    fssk_timing_summary_{regime}.csv       — per-n Spearman/Pearson + range stats

Usage
-----
    python analyse_timing.py --regime MEDIUM
    python analyse_timing.py --input path/to/fssk_exact_scaling_timings_medium.pkl
    python analyse_timing.py --regime MEDIUM --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

import _validation_util.plot_config as plot_config  # noqa: F401 — applies rcParams; must come after matplotlib.use()
from _validation_util.plot_config import new_fig, savefig_fig, COLORS, MARKERS, SCATTER_SIZE

try:
    from scipy.stats import pearsonr, spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse FSSK timing sweep results")
    p.add_argument("--regime", default="MEDIUM",
                   help="Regime tag used to locate the input pickle (e.g. MEDIUM).")
    p.add_argument("--input", type=Path, default=None,
                   help="Explicit input pickle path (overrides --regime lookup).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_prepare(pkl_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    for col in ["q", "J", "R", "N", "m", "wall_hot_median_s", "cpu_hot_median_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["q", "J", "R", "N", "m", "wall_hot_median_s"]).copy()
    df["q"] = df["q"].astype(int)
    df["W_q"] = np.where(
        df["q"] == 1,
        (df["J"] - 1) * df["R"] ** 2 * df["m"] ** df["N"],
        (df["J"] - 1) * df["R"] ** 2 * df["N"] * df["m"] ** df["N"],
    )
    return df


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["wall_hot_mean_s", "W_q"])
    rows = []
    for q, g in sub.groupby("q"):
        log_w = np.log(g["W_q"])
        log_t = np.log(g["wall_hot_mean_s"])
        pearson = spearman = float("nan")
        if _SCIPY and len(g) >= 4:
            pearson  = float(pearsonr(log_w,  log_t)[0])
            spearman = float(spearmanr(log_w, log_t)[0])
        rows.append(dict(
            q=int(q),
            n_points=len(g),
            W_min=float(g["W_q"].min()),
            W_max=float(g["W_q"].max()),
            wall_min=float(g["wall_hot_mean_s"].min()),
            wall_max=float(g["wall_hot_mean_s"].max()),
            spearman_log=spearman,
            pearson_log=pearson,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_wall_vs_work(
    df: pd.DataFrame,
    out_dir: Path,
    tag: str,
    formats: list[str],
) -> None:
    sub = df.dropna(subset=["cpu_hot_mean_s"])
    qs = sorted(sub["q"].unique())

    fig, ax = new_fig("half")
    for i, q in enumerate(qs):
        g = sub[sub["q"] == q]
        ax.scatter(
            g["W_q"], g["wall_hot_mean_s"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            s=SCATTER_SIZE, edgecolors="none",
            label=fr"$q={q}$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Predicted work $W_q$")
    ax.set_ylabel(r"Mean elapsed time per path [s]")
    ax.set_title(r"FSSK $\mathrm{VSig}$ -- Computation Times")
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(ncol=len(qs), loc="upper left", frameon=True)

    stem = out_dir / f"fssk_wall_vs_work_{tag}"
    savefig_fig(fig, stem, formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tag = args.regime.lower()
    pkl_path = args.input or (args.output_dir / f"fssk_exact_scaling_timings_{tag}.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Input not found: {pkl_path}\n"
            "Run sweep_timing.py first, or pass --input explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(pkl_path)
    print(f"Loaded {len(df)} rows from {pkl_path}")
    print("q distribution:")
    print(df["q"].value_counts().sort_index().rename("count").to_frame().to_string())
    print()

    summary = build_summary(df)
    csv_path = args.output_dir / f"fssk_timing_summary_{tag}.csv"
    summary.to_csv(csv_path, index=False)
    print("Timing summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    sub = df.dropna(subset=["wall_hot_mean_s", "cpu_hot_mean_s"])
    if len(sub):
        ratio = sub["wall_hot_mean_s"] / sub["cpu_hot_mean_s"]
        print(
            f"\nWall/CPU ratio — median: {ratio.median():.3f}  "
            f"min: {ratio.min():.3f}  max: {ratio.max():.3f}"
        )

    if not args.no_plots:
        plot_wall_vs_work(df, args.output_dir, tag, args.formats)


if __name__ == "__main__":
    main()