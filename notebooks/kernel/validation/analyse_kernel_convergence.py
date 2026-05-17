"""
analyse_kernel_convergence.py — Analyse and plot FSSK kernel convergence results.

Reads the pickle produced by sweep_kernel_convergence.py and generates:

  Figure 1 (vsig_conv)
    Semi-log plot of max error vs VSig truncation level N.

  Figure 2 (pde_conv)
    Semi-log plot of max error vs dyadic order p for naive_euler / etd1 / heun,
    with least-squares O(2^{-slope·p}) reference lines.

  Figure 3 (timing)
    Wall-clock hot time vs parameter value for all methods.

  Figure 4 (jit_overhead)
    First-call (JIT) timing vs parameter value for all methods.

Also writes:
    kernel_conv_errors.csv   — full error table
    kernel_conv_rates.csv    — fitted convergence rates (PDE methods)

Usage
-----
    python analyse_kernel_convergence.py
    python analyse_kernel_convergence.py --input path/to/kernel_conv_sweep.pkl
    python analyse_kernel_convergence.py --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

import _validation_util.plot_config as _pc          # noqa: F401 — applies rcParams
from _validation_util.plot_config import new_fig, savefig_fig, COLORS, MARKERS
from _validation_util.analysis_utils import fit_slope


# ---------------------------------------------------------------------------
# Method display config
# ---------------------------------------------------------------------------

_METHOD_STYLE: dict[str, dict] = {
    "naive_euler": dict(label="naive",  color=COLORS[0], marker=MARKERS[0]),
    "etd1":        dict(label="exp.int.",         color=COLORS[1], marker=MARKERS[1]),
    "heun":        dict(label="pre.cor.",         color=COLORS[2], marker=MARKERS[2]),
    "vsig_trunc":  dict(label="trunc",   color=COLORS[3], marker=MARKERS[3]),
}

_PDE_METHODS = ["naive_euler", "etd1", "heun"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse FSSK kernel convergence sweep")
    p.add_argument("--input", type=Path, default=None,
                   help="Pickle from sweep_kernel_convergence.py "
                        "(default: <output-dir>/kernel_conv_sweep.pkl)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--formats",    nargs="+", default=["pdf", "png"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    # Accept partial checkpoint if final pkl not yet present
    pkl_path = args.input or (args.output_dir / "kernel_conv_sweep.pkl")
    if not pkl_path.exists():
        partial = args.output_dir / "kernel_conv_sweep_partial.pkl"
        if partial.exists():
            print(f"Final pkl not found; loading partial checkpoint: {partial}")
            pkl_path = partial
        else:
            raise FileNotFoundError(
                f"Input not found: {pkl_path}\n"
                "Run sweep_kernel_convergence.py first, or pass --input explicitly."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(pkl_path)

    # ── Experiment metadata ───────────────────────────────────────────────────
    row0 = df.iloc[0]
    print("── Experiment metadata ─────────────────────────────────────────────────")
    for key in ("regime", "n_paths", "J", "d", "q", "R", "m",
                "eig_min", "eig_max", "jordan_alpha", "dt",
                "ref_method", "ref_trunc", "seed_kernel", "seed_x", "seed_y"):
        if key in df.columns:
            print(f"  {key} = {row0[key]}")
    regime_name = str(row0.get("regime", ""))

    # ── Error summary ─────────────────────────────────────────────────────────
    print("\n── Error summary (ordered by workload level) ───────────────────────────")
    cols = ["workload_level", "method", "param_type", "param_value",
            "max_abs_error", "mean_abs_error", "wall_hot_s"]
    cols_present = [c for c in cols if c in df.columns]
    sort_keys = [c for c in ["workload_level", "method", "param_value"] if c in df.columns]
    print(df.sort_values(sort_keys)[cols_present].to_string(index=False))

    # ── Convergence rates for PDE methods ─────────────────────────────────────
    rate_rows: list[dict] = []
    print("\n── Fitted convergence rates (PDE methods, slope per log2-dyadic) ───────")
    for method in _PDE_METHODS:
        sub = df[df["method"] == method].sort_values("param_value")
        if sub.empty:
            continue
        x = sub["param_value"].to_numpy(dtype=float)
        y = sub["max_abs_error"].to_numpy(dtype=float)
        slope, intercept = fit_slope(x, y)
        print(f"  {method:<12s}  slope = {slope:+.3f}  intercept_log2 = {intercept:.3f}")
        rate_rows.append(dict(method=method, slope=slope, intercept_log2=intercept))

    df_rates = pd.DataFrame(rate_rows)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    path_errors = args.output_dir / "kernel_conv_errors.csv"
    path_rates  = args.output_dir / "kernel_conv_rates.csv"
    df.to_csv(path_errors, index=False)
    df_rates.to_csv(path_rates, index=False)
    print(f"\nSaved:\n  {path_errors}\n  {path_rates}")

    if args.no_plots:
        return

    title_suffix = f"  [{regime_name}]" if regime_name else ""

    # =========================================================================
    # Figure 1 — VSig truncation convergence
    # =========================================================================
    vsig_df = df[df["method"] == "vsig_trunc"].sort_values("param_value")
    if not vsig_df.empty:
        fig, ax = new_fig("half")
        style = _METHOD_STYLE["vsig_trunc"]
        ax.semilogy(
            vsig_df["param_value"],
            vsig_df["max_abs_error"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
        )
        ax.set_xlabel("Truncation level $N$")
        ax.set_ylabel(
            r"$\max_i\,\bigl|\kappa^{\mathrm{ref},N}(x^{(i)},w^{(i)})"
            r" - \kappa^{\mathrm{ref},N_{\max}}(x^{(i)},w^{(i)})\bigr|$"
        )
        ax.set_title("VSig inner-product convergence" + title_suffix)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend()
        stem = args.output_dir / "kernel_conv_vsig"
        savefig_fig(fig, stem, args.formats)
        plt.close(fig)

    # =========================================================================
    # Figure 2 — PDE scheme convergence (dyadic order)
    # =========================================================================
    pde_df = df[df["method"].isin(_PDE_METHODS)]
    if not pde_df.empty:
        fig, ax = new_fig("half")
        for method in _PDE_METHODS:
            sub = pde_df[pde_df["method"] == method].sort_values("param_value")
            if sub.empty:
                continue
            style = _METHOD_STYLE[method]
            x = sub["param_value"].to_numpy(dtype=float)
            y = sub["max_abs_error"].to_numpy(dtype=float)
            ax.semilogy(x, y, marker=style["marker"], color=style["color"],
                        label=style["label"])

        ax.set_xlabel(r"dyadic refinement level $\lambda$")
        ax.set_ylabel(r"$\delta\kappa_\lambda$")
        ax.set_title("PDE scheme convergence")
        # Force integer ticks on x-axis for dyadic order
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend()
        stem = args.output_dir / "kernel_conv_pde"
        savefig_fig(fig, stem, args.formats)
        plt.close(fig)

    # =========================================================================
    # Figure 3 — Hot wall time + wall/CPU ratio (2 panels)
    # =========================================================================
    fig, (ax_t, ax_r) = new_fig("full", ncols=2)

    all_methods = ["vsig_trunc"] + _PDE_METHODS

    # Left panel: wall time vs parameter
    for method in all_methods:
        sub = df[df["method"] == method].sort_values("param_value")
        if sub.empty:
            continue
        style = _METHOD_STYLE[method]
        ax_t.plot(sub["param_value"], sub["wall_hot_s"],
                  marker=style["marker"], color=style["color"], label=style["label"])

    ax_t.set_xlabel("Parameter value  (trunc $N$ / dyadic order $\lambda$)")
    ax_t.set_ylabel("Hot-call wall time (s)")
    ax_t.set_title("Hot-call timing" + title_suffix)
    ax_t.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_t.legend(fontsize=6)

    # Right panel: per-method min/median/max of wall/CPU ratio
    ratio_methods, ratio_mins, ratio_meds, ratio_maxs = [], [], [], []
    for method in all_methods:
        sub = df[df["method"] == method].copy()
        if sub.empty or sub["cpu_hot_s"].isna().all():
            continue
        valid = sub[(sub["cpu_hot_s"] > 0) & sub["wall_hot_s"].notna()]
        if valid.empty:
            continue
        ratios = valid["wall_hot_s"] / valid["cpu_hot_s"]
        ratio_methods.append(_METHOD_STYLE[method]["label"])
        ratio_mins.append(float(ratios.min()))
        ratio_meds.append(float(ratios.median()))
        ratio_maxs.append(float(ratios.max()))

    if ratio_methods:
        xs = np.arange(len(ratio_methods))
        colors = [COLORS[i % len(COLORS)] for i in range(len(ratio_methods))]
        ax_r.bar(xs, ratio_meds, color=colors, alpha=0.7, zorder=3, label="median")
        ax_r.errorbar(xs, ratio_meds,
                      yerr=[np.array(ratio_meds) - np.array(ratio_mins),
                             np.array(ratio_maxs) - np.array(ratio_meds)],
                      fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        ax_r.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_r.set_xticks(xs)
        ax_r.set_xticklabels(ratio_methods, rotation=20, ha="right", fontsize=6)
        ax_r.set_ylabel("Wall time / CPU time")
        ax_r.set_title("Wall/CPU ratio (hot calls)" + title_suffix)
        ax_r.grid(True, axis="y", alpha=0.2)

        print("\n── Wall/CPU ratio summary (< 1 = parallel, = 1 = single-threaded) ─────")
        for m, mn, med, mx in zip(ratio_methods, ratio_mins, ratio_meds, ratio_maxs):
            print(f"  {m:<20s}  min={mn:.2f}  median={med:.2f}  max={mx:.2f}")

    fig.tight_layout()
    stem = args.output_dir / "kernel_conv_timing"
    savefig_fig(fig, stem, args.formats)
    plt.close(fig)

    # =========================================================================
    # Figure 4 — First-call (JIT) timing
    # =========================================================================
    fig, ax = new_fig("half")
    for method in ["vsig_trunc"] + _PDE_METHODS:
        sub = df[df["method"] == method].sort_values("param_value")
        if sub.empty:
            continue
        style = _METHOD_STYLE[method]
        ax.plot(sub["param_value"], sub["wall_first_s"],
                marker=style["marker"], color=style["color"],
                linestyle=":", linewidth=1.2, alpha=0.8,
                label=f"{style['label']} (first)")

    ax.set_xlabel("Parameter value  (trunc $N$ / dyadic order $p$)")
    ax.set_ylabel("First-call wall time (s, includes JIT)")
    ax.set_title("JIT + first-execution overhead" + title_suffix)
    # Force integer ticks on x-axis (both N and p are integers)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=6)
    stem = args.output_dir / "kernel_conv_jit_overhead"
    savefig_fig(fig, stem, args.formats)
    plt.close(fig)

    # =========================================================================
    # Figure 5 — Error vs computational cost  (all methods on one axes)
    #
    # x-axis: hot-call wall time (seconds) — proxy for computational cost
    # y-axis: max absolute error vs VSig reference
    #
    # Each method sweeps a curve from (cheap, high-error) at low param value
    # to (expensive, low-error) at high param value.  Methods that are "lower
    # and to the left" give better accuracy per unit of compute.
    # =========================================================================
    fig, ax = new_fig("full", aspect=1.6)    # wider for the legend

    all_methods = ["vsig_trunc"] + _PDE_METHODS
    for method in all_methods:
        sub = df[df["method"] == method].sort_values("wall_hot_s")
        if sub.empty:
            continue
        # Drop reference row (error = 0 → undefined on log scale)
        sub = sub[sub["max_abs_error"] > 0]
        if sub.empty:
            continue
        style = _METHOD_STYLE[method]
        x = sub["wall_hot_s"].to_numpy(dtype=float)
        y = sub["max_abs_error"].to_numpy(dtype=float)
        pv = sub["param_value"].to_numpy()

        line, = ax.loglog(x, y,
                          marker=style["marker"], color=style["color"],
                          label=style["label"])

        # Annotate each point with its parameter value
        for xi, yi, pvi in zip(x, y, pv):
            ax.annotate(
                str(pvi),
                xy=(xi, yi),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5,
                color=style["color"],
            )

    ax.set_xlabel("Hot-call wall time (s)  [proxy for cost]")
    ax.set_ylabel(r"$\max_i\,|k(X_i,Y_i) - k_{\mathrm{ref}}(X_i,Y_i)|$")
    ax.set_title("Accuracy vs. computational cost — all methods" + title_suffix)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(loc="upper right")
    stem = args.output_dir / "kernel_conv_cost_accuracy"
    savefig_fig(fig, stem, args.formats)
    plt.close(fig)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()



