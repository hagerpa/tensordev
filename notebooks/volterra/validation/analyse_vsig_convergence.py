"""
analyse_vsig_convergence.py — Fit convergence rates and plot vsig_conv results.

Reads the pickle produced by sweep_vsig_convergence.py and produces two figures
per beta:

  1. convergence  — aggregate terminal error (max n!-scaled over levels > 0)
                    vs x_val, x range = vsig dyadic orders only.
                    Lines: vsig orders 0/1/2 + PC (limited to vsig x range).

  2. tradeoff     — aggregate terminal error vs wall-clock time (loglog).
                    vsig uses its x range; PC uses its full range.

Outputs (--output-dir):
    vsig_conv_{mode}_rates.csv
    vsig_conv_{mode}_{beta_tag}_convergence.{pdf,png}
    vsig_conv_{mode}_{beta_tag}_tradeoff.{pdf,png}

Usage
-----
    python analyse_vsig_convergence.py
    python analyse_vsig_convergence.py --input path/to/vsig_conv_dyadic.pkl
    python analyse_vsig_convergence.py --no-plots
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
from _validation_util.analysis_utils import fit_slope


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse vsig convergence sweep")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--mode", choices=["steps", "dyadic"], default=None)
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mode-specific plot/fit helpers
# ---------------------------------------------------------------------------

def _make_mode_helpers(mode: str):
    if mode == "steps":
        def _plot(ax, xs, ys, **kw):
            return ax.loglog(np.asarray(xs, float), ys, **kw)

        def _fit_p(xs, ys):
            log_xs = np.log10(np.asarray(xs, float))
            log_ys = np.log10(np.where(np.asarray(ys) > 0, ys, np.nan))
            ok = np.isfinite(log_ys)
            if ok.sum() < 2:
                return float("nan")
            return -float(np.polyfit(log_xs[ok], log_ys[ok], 1)[0])

        def _fit_line(xs, ys):
            log_xs = np.log10(np.asarray(xs, float))
            log_ys = np.log10(np.where(np.asarray(ys) > 0, ys, np.nan))
            ok = np.isfinite(log_ys)
            if ok.sum() < 2:
                return ys * np.nan
            return 10 ** np.polyval(np.polyfit(log_xs[ok], log_ys[ok], 1), log_xs)

        x_label = "steps $S$"
    else:
        def _plot(ax, xs, ys, **kw):
            return ax.semilogy(np.asarray(xs, float), ys, **kw)

        def _fit_p(xs, ys):
            slope, _ = fit_slope(np.asarray(xs, float), np.asarray(ys))
            return -slope

        def _fit_line(xs, ys):
            xs_f = np.asarray(xs, float)
            slope, intercept = fit_slope(xs_f, np.asarray(ys))
            return 2.0 ** (intercept + slope * xs_f)

        x_label = r"$\lambda$"

    return _plot, _fit_p, _fit_line, x_label


# ---------------------------------------------------------------------------
# Per-beta plotting
# ---------------------------------------------------------------------------

def _plot_beta(df_term, params, beta, mode, args, rate_rows):
    _plot, _fit_p, _fit_line, x_label = _make_mode_helpers(mode)
    vsig_orders = params.get("vsig_orders", [0, 1, 2])
    batch       = params.get("batch", 1)
    out_dir = args.output_dir
    beta_tag = f"b{beta:.2f}".replace(".", "p")
    formats = args.formats

    sub = df_term[df_term["beta"] == beta]
    sub_vsig = sub[sub["method"] == "vsig"]
    sub_pc   = sub[sub["method"] == "pc"]

    x_vals_vsig = sorted(sub_vsig["x_val"].unique())

    order_styles = {0: "-", 1: "--", 2: ":"}

    def _agg_max(df_m):
        return (df_m[df_m["level"] > 0]
                .groupby("x_val", as_index=False)
                .agg(max_abs_entry=("max_abs_entry", "max"))
                .sort_values("x_val"))

    # ── Figure 1: convergence (vsig x range only) ────────────────────────────
    fig, ax = new_fig("half")

    for oi, order in enumerate(vsig_orders):
        sub_o = _agg_max(sub_vsig[sub_vsig["order"] == order])
        xs = sub_o["x_val"].values
        ys = sub_o["max_abs_entry"].values
        p = _fit_p(xs, ys)
        _plot(ax, xs, ys,
              marker=MARKERS[oi], linestyle=order_styles.get(order, "-"),
              color=COLORS[oi], label=f"ord{order}")
        _plot(ax, xs, _fit_line(xs, ys),
              color=COLORS[oi], linestyle="--", linewidth=0.9, alpha=0.55)
        rate_rows.append(dict(beta=beta, method="vsig", order=order, conv_rate=p))

    # PC limited to vsig x range
    sub_pc_lim = _agg_max(sub_pc[sub_pc["x_val"].isin(x_vals_vsig)])
    if not sub_pc_lim.empty:
        xs_pc = sub_pc_lim["x_val"].values
        ys_pc = sub_pc_lim["max_abs_entry"].values
        p_pc = _fit_p(xs_pc, ys_pc)
        _plot(ax, xs_pc, ys_pc,
              marker="s", color=COLORS[-1], label="PC")
        _plot(ax, xs_pc, _fit_line(xs_pc, ys_pc),
              color=COLORS[-1], linestyle="--", linewidth=0.9, alpha=0.55)
        rate_rows.append(dict(beta=beta, method="pc", order=None, conv_rate=p_pc))

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\delta V_\lambda$")
    ax.set_title(r"$\mathrm{VSig}$ scheme convergence" + f" — $\\beta = {beta}$")
    ax.legend(fontsize=6)
    if mode == "dyadic":
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    savefig_fig(fig, out_dir / f"vsig_conv_{mode}_{beta_tag}_convergence", formats)
    plt.close(fig)

    # ── Figure 2: error-runtime tradeoff ─────────────────────────────────────
    fig, ax = new_fig("half")

    for oi, order in enumerate(vsig_orders):
        sub_o = (sub_vsig[sub_vsig["order"] == order]
                 .pipe(lambda d: d[d["level"] > 0])
                 .groupby("x_val", as_index=False)
                 .agg(max_abs_entry=("max_abs_entry", "max"),
                      wall_median_s=("wall_median_s", "first"))
                 .sort_values("x_val"))
        if sub_o.empty:
            continue
        xs_t = sub_o["wall_median_s"].values * 1e3 / batch
        ys_t = sub_o["max_abs_entry"].values
        ax.loglog(xs_t, ys_t,
                  marker=MARKERS[oi], linestyle=order_styles.get(order, "-"),
                  color=COLORS[oi], label=f"ord{order}")
        for x_val, xt, yt in zip(sub_o["x_val"].values, xs_t, ys_t):
            ax.annotate(str(x_val), xy=(xt, yt), fontsize=6,
                        xytext=(3, 0), textcoords="offset points")

    # PC: full range
    sub_pc_rt = (sub_pc[sub_pc["level"] > 0]
                 .groupby("x_val", as_index=False)
                 .agg(max_abs_entry=("max_abs_entry", "max"),
                      wall_median_s=("wall_median_s", "first"))
                 .sort_values("x_val"))
    if not sub_pc_rt.empty:
        xs_rt = sub_pc_rt["wall_median_s"].values * 1e3 / batch
        ys_rt = sub_pc_rt["max_abs_entry"].values
        ax.loglog(xs_rt, ys_rt, marker="s", color=COLORS[-1], label="PC")
        for x_val, xt, yt in zip(sub_pc_rt["x_val"].values, xs_rt, ys_rt):
            ax.annotate(str(x_val), xy=(xt, yt), fontsize=6,
                        color=COLORS[-1], xytext=(3, 0), textcoords="offset points")

    ax.set_xlabel("elapsed time per path (ms)")
    ax.set_ylabel(r"$\delta V_\lambda$")
    ax.set_title(r"$\mathrm{VSig}$ error vs.\ time" + f" — $\\beta = {beta}$")
    ax.legend(fontsize=6)

    savefig_fig(fig, out_dir / f"vsig_conv_{mode}_{beta_tag}_tradeoff", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.input is not None:
        pkl_path = args.input
        if args.mode is None:
            args.mode = "dyadic" if "dyadic" in pkl_path.stem else "steps"
    else:
        mode_guess = args.mode or "dyadic"
        pkl_path = args.output_dir / f"vsig_conv_{mode_guess}.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Input not found: {pkl_path}\n"
            "Run sweep_vsig_convergence.py first, or pass --input explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df_term: pd.DataFrame = data["terminal"]
    params: dict          = data["params"]
    mode = params.get("convergence_mode", args.mode or "dyadic")

    print(f"Loaded: {pkl_path}  ({len(df_term)} terminal rows)")
    print(f"  convergence_mode = {mode}")
    print(f"  betas = {params.get('betas')}")

    betas = sorted(df_term["beta"].unique())
    rate_rows: list[dict] = []

    for beta in betas:
        print(f"\nβ = {beta}")
        if not args.no_plots:
            _plot_beta(df_term, params, beta, mode, args, rate_rows)

    df_rates = pd.DataFrame(rate_rows)
    path_rates = args.output_dir / f"vsig_conv_{mode}_rates.csv"
    df_rates.to_csv(path_rates, index=False)
    print(f"\nSaved: {path_rates}")

    print("\n── Convergence rates ─────────────────────────────────────────────")
    print(df_rates.to_string(index=False))


if __name__ == "__main__":
    main()
