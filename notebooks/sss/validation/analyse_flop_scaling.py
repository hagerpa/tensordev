"""
analyse_flop_scaling.py — XLA FLOPs vs predicted work W_q.

For each configuration the predicted work is

    W_q = (J - 1) R^2 m^N,        n = 1
    W_q = (J - 1) R^2 N m^N,      n > 1

Main output
-----------
One log-log scatter plot of XLA FLOPs vs W_q, coloured by n, with no
fitted regression lines.

    fssk_flop_scaling_<regime>_xla_flops_vs_predicted_work.{png,pdf}

Compact summary CSV
-------------------
One row per n:

    fssk_flop_scaling_<regime>_summary.csv

  columns: n, n_points, W_min, W_max, flops_min, flops_max,
           pearson_log, spearman_log

Usage
-----
    python validation/analyse_flop_scaling.py --regime MEDIUM
    python validation/analyse_flop_scaling.py --regime MEDIUM --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

# plot_config applies rcParams (LaTeX fonts, half-arXiv sizes) on import.
import _validation_util.plot_config as plot_config  # noqa: F401
from _validation_util.plot_config import new_fig, savefig_fig, COLORS, MARKERS, SCATTER_SIZE

try:
    from scipy.stats import pearsonr, spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze FSSK XLA FLOP scaling")
    p.add_argument("--regime", default="MEDIUM",
                   help="Regime tag, e.g. SMALL, MEDIUM, LARGE.")
    p.add_argument("--input", type=Path, default=None,
                   help="Explicit input pickle path (overrides --regime lookup).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs",
                   help="Directory for output files.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"],
                   help="Figure formats to write (default: png pdf).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_results(args: argparse.Namespace) -> pd.DataFrame:
    path = args.input or (args.output_dir / f"fssk_flop_scaling_{args.regime.lower()}.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}\n"
            "Run sweep_flop_scaling.py first, or pass --input explicitly."
        )
    return pd.read_pickle(path)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ["q", "J", "R", "N", "m", "xla_flops"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = (
        out["xla_flops"].notna()
        & np.isfinite(out["xla_flops"])
        & (out["xla_flops"] > 0)
        & (out["J"] > 1)
        & (out["R"] > 0)
        & (out["N"] > 0)
        & (out["m"] > 0)
    )
    out = out.loc[valid].copy()
    out["q"] = out["q"].astype(int)

    intervals = out["J"] - 1
    out["expected_work"] = np.where(
        out["q"] == 1,
        intervals * out["R"] ** 2 * out["m"] ** out["N"],
        intervals * out["R"] ** 2 * out["N"] * out["m"] ** out["N"],
    )
    return out


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, g in df.groupby("q"):
        log_w = np.log(g["expected_work"])
        log_f = np.log(g["xla_flops"])

        pearson = spearman = float("nan")
        if _SCIPY and len(g) >= 4:
            pearson  = float(pearsonr(log_w,  log_f)[0])
            spearman = float(spearmanr(log_w, log_f)[0])

        rows.append(dict(
            q=int(q),
            n_points=len(g),
            W_min=float(g["expected_work"].min()),
            W_max=float(g["expected_work"].max()),
            flops_min=float(g["xla_flops"].min()),
            flops_max=float(g["xla_flops"].max()),
            pearson_log=pearson,
            spearman_log=spearman,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_COLORS  = COLORS
_MARKERS = MARKERS


def plot_flops_vs_work(
    df: pd.DataFrame,
    out_dir: Path,
    tag: str,
    formats: list[str],
) -> None:
    fig, ax = new_fig("half")

    qs = sorted(df["q"].unique())
    for q, color, marker in zip(qs, _COLORS, _MARKERS):
        g = df[df["q"] == q]
        ax.scatter(
            g["expected_work"],
            g["xla_flops"],
            s=SCATTER_SIZE,
            alpha=0.80,
            color=color,
            marker=marker,
            edgecolors="none",
            label=fr"$q={q}$",
        )

    # Unit-slope fit per q: intercept from top-40 by expected_work, excluded from legend.
    import matplotlib.colors as mcolors
    def _brighten(c, f=0.45):
        r, g, b = mcolors.to_rgb(c)
        return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

    x_all = df["expected_work"].values.astype(float)
    y_all = df["xla_flops"].values.astype(float)
    finite = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
    x_line = np.array([x_all[finite].min(), x_all[finite].max()])
    for q, color in zip(qs, _COLORS):
        mask = (df["q"].values == q) & finite
        xq, yq = x_all[mask], y_all[mask]
        if len(xq) < 2:
            continue
        top = np.argsort(xq)[-40:]
        log_c = np.mean(np.log(yq[top]) - np.log(xq[top]))
        ax.plot(x_line, np.exp(log_c) * x_line,
                color=_brighten(color), linestyle="--", linewidth=0.9, label="_nolegend_")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Predicted work $W_q$")
    ax.set_ylabel(r"XLA FLOPs")
    ax.set_title(r"FSSK $\mathrm{VSig}$ -- FLOP Count")
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(ncol=len(qs), loc="upper left")

    stem = out_dir / f"fssk_flop_scaling_{tag}_xla_flops_vs_predicted_work"
    savefig_fig(fig, stem, formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    tag = args.regime.lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_results(args)
    df = prepare_dataframe(df_raw)

    print(f"Loaded rows     : {len(df_raw)}")
    print(f"Valid FLOP rows : {len(df)}")
    print(f"q values        : {sorted(df['q'].unique())}")
    print(f"J values        : {sorted(df['J'].unique())}")
    print(f"R values        : {sorted(df['R'].unique())}")
    print(f"N values        : {sorted(df['N'].unique())}")
    print(f"m values        : {sorted(df['m'].unique())}")
    print()

    summary = build_summary(df)
    csv_path = args.output_dir / f"fssk_flop_scaling_{tag}_summary.csv"
    summary.to_csv(csv_path, index=False)
    print("Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    if not args.no_plots:
        print()
        plot_flops_vs_work(df, args.output_dir, tag, args.formats)


if __name__ == "__main__":
    main()
