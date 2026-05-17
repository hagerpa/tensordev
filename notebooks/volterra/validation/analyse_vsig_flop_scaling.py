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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    required = ["d", "q", "J", "N", "fft_pre_flops", "fft_hot_flops", "quad_flops"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["fft_total_flops"] = out["fft_pre_flops"] + out["fft_hot_flops"]

    # XLA cost_analysis() for lax.scan reports body cost only, not body × S.
    # New sweep pkl already has quad_flops = body × S (column quad_flops_body
    # stores the raw body cost).  Old pkl only has the body cost in quad_flops.
    # Detect which format we have and apply the ×S correction if needed.
    if "quad_flops_body" not in out.columns:
        # Old format: quad_flops is body cost only → scale up by S = J - 1.
        out["quad_flops_body"] = out["quad_flops"]
        out["quad_flops"] = out["quad_flops"] * (out["J"] - 1)

    valid = (
        out["fft_hot_flops"].notna()
        & out["quad_flops"].notna()
        & np.isfinite(out["fft_hot_flops"])
        & np.isfinite(out["quad_flops"])
        & (out["fft_hot_flops"] > 0)
        & (out["quad_flops"] > 0)
        & (out["J"] > 1)
        & (out["N"] > 0)
        & (out["d"] > 0)
    )
    out = out.loc[valid].copy()
    out["d"] = out["d"].astype(int)
    out["q"] = out["q"].astype(int)
    out["N"] = out["N"].astype(int)

    # m = path dimension (d column); formulas use J directly (not S = J-1).
    # N^q m^N: the multi-index channel sum has leading term ~ N^q m^N / (q-1)!
    # so q appears in the *exponent* of N, not as a linear prefactor.
    m = out["d"]
    log2J = np.log2(out["J"].clip(lower=1))
    Nq = out["N"] ** out["q"]

    out["W_quad"] = out["J"] ** 2 * Nq * m ** out["N"]
    out["W_fft"]  = out["J"] * log2J * Nq * m ** out["N"]

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
            fft_pre_flops_min=float(g["fft_pre_flops"].min()),
            fft_pre_flops_max=float(g["fft_pre_flops"].max()),
            fft_hot_flops_min=float(g["fft_hot_flops"].min()),
            fft_hot_flops_max=float(g["fft_hot_flops"].max()),
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
) -> None:
    fig, ax = new_fig("half")

    vals = sorted(df[group_col].unique())
    for v, color, marker in zip(vals, COLORS, MARKERS):
        g = df[df[group_col] == v]
        ax.scatter(
            g[x_col],
            g[y_col],
            s=SCATTER_SIZE,
            alpha=0.80,
            color=color,
            marker=marker,
            edgecolors="none",
            label=group_label_fmt.format(v=v),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(ncol=len(vals), loc="upper left")

    savefig_fig(fig, stem, formats)
    plt.close(fig)


def plot_all(df: pd.DataFrame, out_dir: Path, tag: str, formats: list[str]) -> None:
    # Colored by N — checks that the formula tracks N·m^N scaling.
    _scatter_plot(
        df,
        x_col="W_fft",
        y_col="fft_total_flops",
        x_label=r"Predicted work $W_\mathrm{fft} = J \log_2(J)\, N\, m^N$",
        y_label=r"XLA FLOPs",
        title=r"$\mathrm{VSig}$ FFT — FLOP count",
        stem=out_dir / f"vsig_flop_scaling_{tag}_fft",
        formats=formats,
    )
    _scatter_plot(
        df,
        x_col="W_quad",
        y_col="quad_flops",
        x_label=r"Predicted work $W_\mathrm{quad} = J^2 N\, m^N$",
        y_label=r"XLA FLOPs",
        title=r"$\mathrm{VSig}$ quadratic — FLOP count",
        stem=out_dir / f"vsig_flop_scaling_{tag}_quad",
        formats=formats,
    )
    # Colored by q — q is a proportionality constant so should only shift
    # points vertically (parallel lines in log-log), not change the slope.
    # If slopes differ by q, the formula is missing a q-dependent factor.
    _scatter_plot(
        df,
        x_col="W_fft",
        y_col="fft_total_flops",
        x_label=r"Predicted work $W_\mathrm{fft} = J \log_2(J)\, N\, m^N$",
        y_label=r"XLA FLOPs",
        title=r"$\mathrm{VSig}$ FFT — grouped by $q$",
        stem=out_dir / f"vsig_flop_scaling_{tag}_fft_byq",
        formats=formats,
        group_col="q",
        group_label_fmt=r"$q={v}$",
    )
    _scatter_plot(
        df,
        x_col="W_quad",
        y_col="quad_flops",
        x_label=r"Predicted work $W_\mathrm{quad} = J^2 N\, m^N$",
        y_label=r"XLA FLOPs",
        title=r"$\mathrm{VSig}$ quadratic — grouped by $q$",
        stem=out_dir / f"vsig_flop_scaling_{tag}_quad_byq",
        formats=formats,
        group_col="q",
        group_label_fmt=r"$q={v}$",
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
    print(f"d values (m)    : {sorted(df['d'].unique())}")
    print(f"q values        : {sorted(df['q'].unique())}")
    print(f"J values        : {sorted(df['J'].unique())}")
    print(f"N values        : {sorted(df['N'].unique())}")
    print()

    summary = build_summary(df)
    csv_path = args.output_dir / f"vsig_flop_scaling_{tag}_summary.csv"
    summary.to_csv(csv_path, index=False)
    print("Summary:")
    print(summary.to_string(index=False))
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
