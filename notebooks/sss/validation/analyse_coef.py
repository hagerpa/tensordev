"""
analyse_coef.py — XLA FLOPs vs predicted work for FSSK coefficient computation.

Reads the pickle produced by sweep_coef.py, computes the theoretical work
proxy per row, generates log-log scatter plots (one per Lambda form), and
writes a summary CSV.

Predicted work (Prop. cost_fssk_weights in the paper):
    Dense  Lambda:  W = R³ + R²·N^q
    Jordan Lambda:  W = R²·N^q

Outputs (written to --output-dir):
    fssk_coef_scaling_{regime}_dense_flops_vs_work.{png,pdf}
    fssk_coef_scaling_{regime}_jordan_flops_vs_work.{png,pdf}
    fssk_coef_scaling_{regime}_summary.csv

Usage
-----
    python analyse_coef.py --regime MEDIUM
    python analyse_coef.py --input path/to/fssk_coef_scaling_medium.pkl
    python analyse_coef.py --regime MEDIUM --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

import _validation_util.plot_config as plot_config  # noqa: F401 — applies rcParams
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
    p = argparse.ArgumentParser(description="Analyse FSSK coefficient FLOP scaling")
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
    path = args.input or (args.output_dir / f"fssk_coef_scaling_{args.regime.lower()}.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}\n"
            "Run sweep_coef.py first, or pass --input explicitly."
        )
    return pd.read_pickle(path)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ["form", "q", "R", "N", "xla_flops"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in ["q", "R", "N", "xla_flops"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = (
        out["xla_flops"].notna()
        & np.isfinite(out["xla_flops"])
        & (out["xla_flops"] > 0)
        & (out["R"] > 0)
        & (out["N"] > 0)
        & (out["q"] > 0)
    )
    out = out.loc[valid].copy()
    out["q"] = out["q"].astype(int)

    # Dense:  W = R³ + R²·N^q
    # Jordan: W = R²·N^q
    Nq = out["N"] ** out["q"]
    out["expected_work_dense"]  = out["R"] ** 3 + out["R"] ** 2 * Nq
    out["expected_work_jordan"] = out["R"] ** 2 * Nq
    return out


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (form, q), g in df.groupby(["form", "q"]):
        work_col = "expected_work_dense" if form == "dense" else "expected_work_jordan"
        log_w = np.log(g[work_col])
        log_f = np.log(g["xla_flops"])

        pearson = spearman = float("nan")
        if _SCIPY and len(g) >= 4:
            pearson  = float(pearsonr(log_w,  log_f)[0])
            spearman = float(spearmanr(log_w, log_f)[0])

        rows.append(dict(
            form=form,
            q=int(q),
            n_points=len(g),
            W_min=float(g[work_col].min()),
            W_max=float(g[work_col].max()),
            flops_min=float(g["xla_flops"].min()),
            flops_max=float(g["xla_flops"].max()),
            pearson_log=pearson,
            spearman_log=spearman,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_form(
    df: pd.DataFrame,
    form: str,
    work_col: str,
    work_label: str,
    out_dir: Path,
    tag: str,
    formats: list[str],
) -> None:
    sub = df[df["form"] == form]
    if sub.empty:
        return

    qs = sorted(sub["q"].unique())
    fig, ax = new_fig("half")

    for i, q in enumerate(qs):
        g = sub[sub["q"] == q]
        ax.scatter(
            g[work_col],
            g["xla_flops"],
            s=SCATTER_SIZE,
            alpha=0.80,
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            edgecolors="none",
            label=fr"$q={q}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(fr"Predicted work ${work_label}$")
    ax.set_ylabel(r"XLA FLOPs")
    form_str = "Dense" if form == "dense" else "Jordan"
    ax.set_title(fr"FSSK coef -- {form_str} $\Lambda$ -- FLOP Count")
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(ncol=len(qs), loc="upper left")

    stem = out_dir / f"fssk_coef_scaling_{tag}_{form}_flops_vs_work"
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
    print(f"R values        : {sorted(df['R'].unique())}")
    print(f"N values        : {sorted(df['N'].unique())}")
    print(f"Forms           : {sorted(df['form'].unique())}")
    print()

    summary = build_summary(df)
    csv_path = args.output_dir / f"fssk_coef_scaling_{tag}_summary.csv"
    summary.to_csv(csv_path, index=False)
    print("Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    if not args.no_plots:
        print()
        plot_form(df, "dense",  "expected_work_dense",
                  r"R^3 + R^2 N^q", args.output_dir, tag, args.formats)
        plot_form(df, "jordan", "expected_work_jordan",
                  r"R^2 N^q",       args.output_dir, tag, args.formats)


if __name__ == "__main__":
    main()