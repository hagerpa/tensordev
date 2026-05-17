"""
analyse_fssk_approx.py — Plot FSSK approximation error vs R.

Reads the pickle produced by sweep_fssk_approx.py, plots per-level error
vs R for each beta, overlays the vsig (order=2) baseline, and saves CSVs.

Outputs (--output-dir):
    fssk_approx_fssk.csv
    fssk_approx_vsig.csv
    fssk_approx_{beta_tag}.{pdf,png}           per-level
    fssk_approx_{beta_tag}_aggregate.{pdf,png} aggregate (max over levels>0)

Usage
-----
    python analyse_fssk_approx.py
    python analyse_fssk_approx.py --input path/to/fssk_approx.pkl
    python analyse_fssk_approx.py --no-plots
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # notebooks/

import _validation_util.plot_config as _pc_module  # noqa: F401 — applies rcParams
from _validation_util.plot_config import new_fig, savefig_fig, COLORS, MARKERS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse FSSK approximation sweep")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    pkl_path = args.input or (args.output_dir / "fssk_approx.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Input not found: {pkl_path}\n"
            "Run sweep_fssk_approx.py first, or pass --input explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df_fssk: pd.DataFrame = data["fssk"]
    df_vsig: pd.DataFrame = data["vsig"]
    params: dict          = data["params"]

    print(f"Loaded: {pkl_path}")
    print(f"  betas = {params.get('betas')}")
    print(f"  R values = {params.get('R_values')}")

    betas = sorted(df_fssk["beta"].unique())

    for beta in betas:
        beta_tag = f"b{beta:.2f}".replace(".", "p")
        sub_fssk = df_fssk[df_fssk["beta"] == beta]
        sub_vsig = df_vsig[df_vsig["beta"] == beta]
        R_values = sorted(sub_fssk["R"].unique())
        levels   = sorted(sub_fssk["level"].unique())
        levels_pos = [lv for lv in levels if lv > 0]

        # vsig baseline: max error over levels > 0
        vsig_baseline = float(
            sub_vsig[sub_vsig["level"] > 0]["max_abs_entry"].max()
        )

        if args.no_plots:
            continue

        # ── Per-level plot ───────────────────────────────────────────────────
        fig, ax = new_fig("full")
        for ki, level in enumerate(levels_pos):
            sub = sub_fssk[sub_fssk["level"] == level].sort_values("R")
            ax.semilogy(sub["R"], sub["max_abs_entry"],
                        marker=MARKERS[ki % len(MARKERS)],
                        color=COLORS[ki % len(COLORS)],
                        label=f"level {level}")
        ax.set_xlabel("FSSK state dimension $R$")
        ax.set_ylabel(r"max abs entry ($n!$-scaled)")
        ax.set_title(fr"FSSK approximation vs PC reference — $\beta$={beta}")
        ax.legend()
        savefig_fig(fig, args.output_dir / f"fssk_approx_{beta_tag}", args.formats)
        plt.close(fig)

        # ── Aggregate plot ───────────────────────────────────────────────────
        fig, ax = new_fig("half")
        compact = (sub_fssk[sub_fssk["level"] > 0]
                   .groupby("R", as_index=False)
                   .agg(max_abs_entry=("max_abs_entry", "max"))
                   .sort_values("R"))
        ax.semilogy(compact["R"], compact["max_abs_entry"],
                    marker="o", color=COLORS[0], label="FSSK")
        ax.axhline(vsig_baseline, color=COLORS[1], linestyle="--", alpha=0.8,
                   label="vsig order=2 baseline")
        ax.set_xlabel("FSSK state dimension $R$")
        ax.set_ylabel(r"max abs entry ($n!$-scaled, all levels)")
        ax.set_title(fr"Aggregate FSSK error — $\beta$={beta}")
        ax.legend()
        savefig_fig(fig, args.output_dir / f"fssk_approx_{beta_tag}_aggregate", args.formats)
        plt.close(fig)

        print(f"  β={beta}: vsig baseline = {vsig_baseline:.3e}")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    path_fssk = args.output_dir / "fssk_approx_fssk.csv"
    path_vsig = args.output_dir / "fssk_approx_vsig.csv"
    df_fssk.to_csv(path_fssk, index=False)
    df_vsig.to_csv(path_vsig, index=False)
    print(f"\nSaved:\n  {path_fssk}\n  {path_vsig}")

    print("\n── FSSK aggregate error per beta × R (levels > 0) ───────────────")
    agg = (df_fssk[df_fssk["level"] > 0]
           .groupby(["beta", "R"], as_index=False)
           .agg(max_abs_entry=("max_abs_entry", "max")))
    print(agg.to_string(index=False))

    print("\n── vsig (order=2) baseline per beta ─────────────────────────────")
    baseline = (df_vsig[df_vsig["level"] > 0]
                .groupby("beta", as_index=False)
                .agg(max_abs_entry=("max_abs_entry", "max")))
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()
