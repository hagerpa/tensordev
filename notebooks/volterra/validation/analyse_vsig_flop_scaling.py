"""
analyse_vsig_flop_scaling.py — XLA FLOPs vs predicted work for Volterra vsig.

Reads vsig_flop_scaling.pkl and produces:

  fft_total_flops = fft_pre_flops + fft_hot_flops  (derived column)

Predicted work formulas (m = path dimension = d column):
  W_quad(J, N, m)  = J^2  * N * m^N
  W_fft(J, N, m)   = J * log2(J) * N * m^N

Outputs
-------
  vsig_flop_scaling_summary.csv          — compact per-(d,N) summary
  vsig_flop_scaling_fft.{png,pdf}        — XLA FLOPs vs W_fft
  vsig_flop_scaling_quad.{png,pdf}       — XLA FLOPs vs W_quad

Usage
-----
    python validation/analyse_vsig_flop_scaling.py
    python validation/analyse_vsig_flop_scaling.py --no-plots
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # notebooks/

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
    p = argparse.ArgumentParser(description="Analyse Volterra vsig XLA FLOP scaling")
    p.add_argument("--regime", default="MEDIUM",
                   help="Regime tag (e.g. SMALL, MEDIUM, LARGE).")
    p.add_argument("--input", type=Path, default=None,
                   help="Explicit input pickle path (overrides --regime lookup).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "validation_outputs",
                   help="Directory for output files.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"],
                   help="Figure formats (default: png pdf).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_results(args: argparse.Namespace) -> pd.DataFrame:
    tag  = args.regime.lower()
    path = args.input or (args.output_dir / f"vsig_flop_scaling_{tag}.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}\n"
            "Run sweep_vsig_flop_scaling.py first, or pass --input explicitly."
        )
    return pd.read_pickle(path)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ["d", "m", "q", "J", "N", "fft_pre_flops", "fft_hot_flops", "quad_flops"]
    out = df.copy()
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["fft_total_flops"] = out["fft_pre_flops"] + out["fft_hot_flops"]
    valid = (
        out["fft_total_flops"].notna()
        & out["quad_flops"].notna()
        & np.isfinite(out["fft_total_flops"])
        & np.isfinite(out["quad_flops"])
        & (out["fft_total_flops"] > 0)
        & (out["quad_flops"] > 0)
        & (out["J"] > 1)
        & (out["N"] > 0)
        & (out["m"] > 0)
    )
    out = out.loc[valid].copy()
    out["d"] = out["d"].astype(int)   # input path dim = kernel.path_dim
    out["m"] = out["m"].astype(int)   # Volterra path dim = kernel.m = A.shape[1]
    out["q"] = out["q"].astype(int)   # kernel components = kernel.q = A.shape[0]
    out["N"] = out["N"].astype(int)

    # m = Volterra path dimension (kernel.m); S = J - 1.
    m = out["m"]
    S = out["J"] - 1

    # nfft = next power of 2 above 2S-1 (the FFT padding used in _causal_conv_fft_batched).
    # Many J values share the same nfft; the FFT cost scales with nfft*log2(nfft), not J*log2(J).
    def _next_pow2(n):
        return 1 if n <= 1 else int(1) << int(n - 1).bit_length()

    out["nfft"] = (S * 2 - 1).apply(_next_pow2)
    log2_nfft = np.log2(out["nfft"].clip(lower=1))

    Nq = out["N"] ** out["q"]

    out["W_quad"] = out["J"] ** 2 * out["N"] * m ** out["N"]
    out["W_fft"]  = out["J"] * np.log2(out["J"].clip(lower=1)) * Nq * m ** out["N"]

    return out


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise FLOP data grouped by (d, N, q).

    Key diagnostics:
    - pearson_log_fft / pearson_log_quad: correlation with the simple formula
      (poor correlation at high N is expected due to the workload filter narrowing
      the J range, not because the formula is wrong).
    - J_slope_fft / J_slope_quad: OLS slope of log(flops) on log(J) within each
      (N, q) group.  Expected value 1.0 for fft_total and 2.0 for quad.  A value
      near 1.0 for quad would indicate that XLA cost_analysis counts the lax.scan
      body only once rather than S times.
    """
    rows = []
    for (d, N, q), g in df.groupby(["d", "N", "q"]):
        def _corr(x, y):
            if not _SCIPY or len(g) < 4:
                return float("nan"), float("nan")
            return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])

        def _slope(log_x: np.ndarray, log_y: np.ndarray) -> float:
            """OLS slope of log_y on log_x (scalar, NaN when underdetermined)."""
            if len(log_x) < 2:
                return float("nan")
            try:
                from scipy.stats import linregress
                return float(linregress(log_x, log_y).slope)
            except Exception:
                return float("nan")

        p_fft,  s_fft  = _corr(np.log(g["W_fft"]),  np.log(g["fft_total_flops"]))
        p_quad, s_quad = _corr(np.log(g["W_quad"]), np.log(g["quad_flops"]))

        logJ = np.log(g["J"].values.astype(float))
        slope_fft  = _slope(logJ, np.log(g["fft_total_flops"].values.astype(float)))
        slope_quad = _slope(logJ, np.log(g["quad_flops"].values.astype(float)))

        rows.append(dict(
            d=int(d), N=int(N), q=int(q),
            n_points=len(g),
            fft_total_flops_min=float(g["fft_total_flops"].min()),
            fft_total_flops_max=float(g["fft_total_flops"].max()),
            quad_flops_min=float(g["quad_flops"].min()),
            quad_flops_max=float(g["quad_flops"].max()),
            pearson_log_fft=p_fft,
            spearman_log_fft=s_fft,
            pearson_log_quad=p_quad,
            spearman_log_quad=s_quad,
            J_slope_fft=slope_fft,
            J_slope_quad=slope_quad,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    stem: Path,
    formats: list[str],
    *,
    group_col: str = "N",
    group_label_fmt: str = r"$N={v}$",
    legend: bool = True,
    fit_n: int = 40,
) -> None:
    fig, ax = new_fig("half")

    vals = sorted(df[group_col].unique())
    color_cycle  = itertools.cycle(COLORS)
    marker_cycle = itertools.cycle(MARKERS)
    for v in vals:
        g = df[df[group_col] == v]
        ax.scatter(
            g[x_col],
            g[y_col],
            s=SCATTER_SIZE,
            alpha=0.80,
            color=next(color_cycle),
            marker=next(marker_cycle),
            edgecolors="none",
            label=group_label_fmt.format(v=v),
        )

    # Unit-slope fit per q: log(y) = log(c) + log(x), intercept from top-fit_n by x.
    import matplotlib.colors as mcolors
    def _brighten(c, f=0.45):
        r, g, b = mcolors.to_rgb(c)
        return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

    q_vals = sorted(df["q"].unique())
    q_color = {v: COLORS[i % len(COLORS)] for i, v in enumerate(q_vals)}
    x_all = df[x_col].values.astype(float)
    y_all = df[y_col].values.astype(float)
    finite = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
    x_line = np.array([x_all[finite].min(), x_all[finite].max()])
    for q_v in q_vals:
        mask = (df["q"].values == q_v) & finite
        xq, yq = x_all[mask], y_all[mask]
        if len(xq) < 2:
            continue
        top = np.argsort(xq)[-fit_n:]
        log_c = np.mean(np.log(yq[top]) - np.log(xq[top]))
        ax.plot(
            x_line, np.exp(log_c) * x_line,
            color=_brighten(q_color[q_v]), linestyle="--", linewidth=0.9,
            label="_nolegend_",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)
    if legend:
        ax.legend(ncol=len(vals), loc="upper left")

    savefig_fig(fig, stem, formats)
    plt.close(fig)


_FFT_X_LABEL  = r"Predicted Work $W_\mathrm{FFT}$"
_FFT_Y_LABEL  = r"XLA FLOPs"
_QUAD_X_LABEL = r"Predicted Work $W_\mathrm{quad}$"
_QUAD_Y_LABEL = r"XLA FLOPs"


def plot_all(df: pd.DataFrame, out_dir: Path, tag: str, formats: list[str]) -> None:
    for group_col, group_fmt, show_legend, suffix in [
        ("q", r"$q={v}$", True,  "byq"),
        ("N", r"$N={v}$", False, "byN"),
        ("nfft", r"$n_\mathrm{{fft}}={v}$", False, "bynfft"),
    ]:
        _scatter_plot(
            df,
            x_col="W_fft",
            y_col="fft_total_flops",
            x_label=_FFT_X_LABEL,
            y_label=_FFT_Y_LABEL,
            title=r"VSig -- FFT Alg. -- FLOP Count",
            stem=out_dir / f"vsig_flop_scaling_{tag}_fft_{suffix}",
            formats=formats,
            group_col=group_col,
            group_label_fmt=group_fmt,
            legend=show_legend,
        )
        _scatter_plot(
            df,
            x_col="W_quad",
            y_col="quad_flops",
            x_label=_QUAD_X_LABEL,
            y_label=_QUAD_Y_LABEL,
            title=r"VSig -- Quad. Alg. -- FLOP Count",
            stem=out_dir / f"vsig_flop_scaling_{tag}_quad_{suffix}",
            formats=formats,
            group_col=group_col,
            group_label_fmt=group_fmt,
            legend=show_legend,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    tag  = args.regime.lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_results(args)
    df = prepare_dataframe(df_raw)

    print(f"Regime          : {args.regime}")
    print(f"Loaded rows     : {len(df_raw)}")
    print(f"Valid rows      : {len(df)}")
    print(f"m values        : {sorted(df['m'].unique())}  (kernel.m = Volterra path dim)")
    print(f"q values        : {sorted(df['q'].unique())}  (kernel.q = #components, always 1)")
    print(f"d values        : {sorted(df['d'].unique())}  (kernel.path_dim = input path dim)")
    print(f"J values        : {sorted(df['J'].unique())}")
    print(f"N values        : {sorted(df['N'].unique())}")
    print()

    summary = build_summary(df)
    csv_path = args.output_dir / f"vsig_flop_scaling_{tag}_summary.csv"
    summary.to_csv(csv_path, index=False)
    print("Summary (J-slopes, grouped by N and q):")
    cols = ["d", "N", "q", "n_points", "J_slope_fft", "J_slope_quad"]
    print(summary[cols].to_string(index=False))
    print()

    # N-slope: OLS slope of log(FLOPs) on N within each (J, q) group.
    # quad: W=J²·N·m^N → log W ≈ N·log(m) + log(N) ≈ N·log(m)  (log(m)≈1.099 for m=3)
    # fft:  W=J·log(nfft)·N^q·m^N → N-slope ≈ log(m) + q/N → log(m) asymptotically.
    print(f"N-slope analysis  (expected log(m)=log(3)≈{np.log(3):.3f} for pure m^N):")
    from scipy.stats import linregress
    n_rows = []
    for (d, J, q_grp), g in df.groupby(["d", "J", "q"]):
        Nv = g["N"].values.astype(float)
        if len(Nv) < 2:
            continue
        def _ns(y):
            try:
                return float(linregress(Nv, np.log(y)).slope)
            except Exception:
                return float("nan")
        n_rows.append(dict(
            d=int(d), J=int(J), q=int(q_grp),
            N_slope_fft_hot=_ns(g["fft_hot_flops"].values.astype(float)),
            N_slope_quad=_ns(g["quad_flops"].values.astype(float)),
        ))
    n_df = pd.DataFrame(n_rows)
    print(n_df.groupby("q")[["N_slope_fft_hot", "N_slope_quad"]].mean().to_string())
    print()

    print(f"\nSaved: {csv_path}")

    if not args.no_plots:
        print()
        plot_all(df, args.output_dir, tag, args.formats)
        print(
            f"Figures saved to {args.output_dir}/vsig_flop_scaling_{tag}_"
            f"{{fft,quad,fft_byq,quad_byq}}.{{png,pdf}}"
        )


if __name__ == "__main__":
    main()
