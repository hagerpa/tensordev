"""
analyse_timings.py — Load FSSK timing results, evaluate fixed-slope scaling
model, run individual per-axis fits, and produce all diagnostic plots.

Expects the timing pickle produced by run_timings.py to already exist in
``validation_outputs/``.  Run that script first if needed:

    python run_timings.py --regime MEDIUM

Then analyse:

    python analyse_timings.py --regime MEDIUM

Fixed-slope models
------------------
q > 1:   T ≈ C_q  J  R²  N  m^N
q = 1:   T ≈ C    J  R²  m^N

We fix the theoretical exponents and fit only the multiplicative constants,
then check that the log-space R² is close to 1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from plot_config import finish_plot  # sets rcParams on import

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse FSSK exact-scaling timings")
    p.add_argument(
        "--regime",
        choices=["MEDIUM", "LARGE"],
        default="MEDIUM",
        help="Which timing pickle to load (default: MEDIUM)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "validation_outputs",
        help="Directory containing the timing pickle",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_r2(y, yhat):
    y    = np.asarray(y,    dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    rss  = float(np.sum((y - yhat) ** 2))
    tss  = float(np.sum((y - y.mean()) ** 2))
    return np.nan if tss == 0 else 1.0 - rss / tss


def fit_curve_group(df, *, x_col, y_col):
    """OLS log-space linear fit.

    Returns ``(df_annotated, fit_info)`` where ``fit_info`` is a dict with
    keys ``slope``, ``intercept``, ``r2``, ``n``, ``mean_abs_resid``,
    ``max_abs_resid``.
    """
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)

    slope, intercept, r_value, _, _ = scipy_stats.linregress(x, y)
    yhat  = intercept + slope * x
    resid = y - yhat

    fit_info = dict(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r_value ** 2),
        n=int(len(x)),
        mean_abs_resid=float(np.mean(np.abs(resid))),
        max_abs_resid=float(np.max(np.abs(resid))),
    )

    df_out = df.copy()
    df_out["_fitted"]  = yhat
    df_out["_resid"]   = resid
    return df_out, fit_info


def fixed_model_summary(df, *, group_col="fixed_model"):
    rows = []
    for name, g in df.groupby(group_col):
        rows.append(dict(
            fixed_model=name,
            n=len(g),
            r2_log=log_r2(g["log_hot_time"], g["pred_log_hot_time_fixed"]),
            mean_abs_log_resid=float(np.mean(np.abs(g["fixed_resid_log"]))),
            max_abs_log_resid=float(np.max(np.abs(g["fixed_resid_log"]))),
            mean_abs_rel_error=float(np.mean(np.abs(g["fixed_rel_error"]))),
            max_abs_rel_error=float(np.max(np.abs(g["fixed_rel_error"]))),
        ))
    return pd.DataFrame(rows)


def matched_axis_data(df, *, family, q, direction):
    """Select axial-extension rows plus the cube rows with matching fixed params."""
    if direction == "J":
        ext_design, fixed_cols = "J_ext", ["R", "N"]
    elif direction == "R2":
        ext_design, fixed_cols = "R_ext", ["J", "N"]
    elif direction == "truncation":
        ext_design, fixed_cols = "N_ext", ["J", "R"]
    else:
        raise ValueError(direction)

    base = df[(df["family"] == family) & (df["q"] == q)] if family == "qgt1" \
           else df[df["family"] == family]

    ext = base[base["design"] == ext_design].copy()
    if len(ext) == 0:
        return ext

    fixed_values = {col: ext[col].iloc[0] for col in fixed_cols}
    cube = base[base["design"] == "cube"].copy()
    for col, val in fixed_values.items():
        cube = cube[cube[col] == val]

    out = pd.concat([cube, ext], ignore_index=True)
    out = out.drop_duplicates(["family", "q", "J", "R", "N"], keep="first")
    return out

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_timing_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    TIME_COL = "wall_hot_median_s" if "wall_hot_median_s" in df_raw.columns \
               else "wall_hot_mean_s"

    df = df_raw[np.isfinite(df_raw[TIME_COL]) & (df_raw[TIME_COL] > 0)].copy()
    df["runtime"]      = df[TIME_COL]
    df["log_hot_time"] = np.log(df["runtime"])
    df["log_runtime"]  = df["log_hot_time"]   # alias used by direction_specs

    df["log_J"]     = np.log(df["J"])
    df["two_log_R"] = 2.0 * np.log(df["R"])
    df["log_mN"]    = df["N"] * np.log(df["m"])
    df["log_NmN"]   = np.log(df["N"]) + df["log_mN"]

    df["log_cost_q1"]   = df["log_J"] + df["two_log_R"] + df["log_mN"]
    df["log_cost_qgt1"] = df["log_J"] + df["two_log_R"] + df["log_NmN"]
    df["cost_q1"]       = np.exp(df["log_cost_q1"])
    df["cost_qgt1"]     = np.exp(df["log_cost_qgt1"])

    print("Using runtime column:", TIME_COL)
    return df, TIME_COL

# ---------------------------------------------------------------------------
# Fixed-slope model
# ---------------------------------------------------------------------------

def fit_fixed_slope(df):
    df_qgt1 = df[df["family"] == "qgt1"].copy()
    df_q1   = df[df["family"] == "q1"  ].copy()

    # q = 1: one global constant
    df_q1["log_constant_estimate"]    = df_q1["log_hot_time"] - df_q1["log_cost_q1"]
    q1_log_C = float(df_q1["log_constant_estimate"].mean())
    q1_C     = float(np.exp(q1_log_C))
    df_q1["log_C"]                    = q1_log_C
    df_q1["C"]                        = q1_C
    df_q1["pred_log_hot_time_fixed"]  = df_q1["log_C"] + df_q1["log_cost_q1"]
    df_q1["pred_hot_time_fixed"]      = np.exp(df_q1["pred_log_hot_time_fixed"])
    df_q1["fixed_resid_log"]          = df_q1["log_hot_time"] - df_q1["pred_log_hot_time_fixed"]
    df_q1["fixed_rel_error"]          = df_q1["pred_hot_time_fixed"] / df_q1["runtime"] - 1.0
    df_q1["fixed_model"]              = "q1_fixed_slopes"

    # q > 1: one constant per q value
    df_qgt1["log_constant_estimate"] = df_qgt1["log_hot_time"] - df_qgt1["log_cost_qgt1"]

    df_q_constants = (
        df_qgt1
        .groupby("q", as_index=False)
        .agg(
            log_C=("log_constant_estimate", "mean"),
            log_C_std=("log_constant_estimate", "std"),
            n=("log_constant_estimate", "count"),
        )
    )
    df_q_constants["C"]             = np.exp(df_q_constants["log_C"])
    df_q_constants["C_std_factor"]  = np.exp(df_q_constants["log_C_std"])
    df_q_constants["relative_C"]    = df_q_constants["C"] / df_q_constants["C"].iloc[0]

    df_qgt1 = df_qgt1.merge(df_q_constants[["q", "log_C", "C"]], on="q", how="left")
    df_qgt1["pred_log_hot_time_fixed"] = df_qgt1["log_C"] + df_qgt1["log_cost_qgt1"]
    df_qgt1["pred_hot_time_fixed"]     = np.exp(df_qgt1["pred_log_hot_time_fixed"])
    df_qgt1["fixed_resid_log"]         = df_qgt1["log_hot_time"] - df_qgt1["pred_log_hot_time_fixed"]
    df_qgt1["fixed_rel_error"]         = df_qgt1["pred_hot_time_fixed"] / df_qgt1["runtime"] - 1.0
    df_qgt1["fixed_model"]             = "qgt1_fixed_slopes"

    df_fixed_all = pd.concat([df_qgt1, df_q1], ignore_index=True)

    return df_qgt1, df_q1, df_q_constants, df_fixed_all, q1_C

# ---------------------------------------------------------------------------
# Individual per-axis fits
# ---------------------------------------------------------------------------

DIRECTION_SPECS = [
    dict(
        direction="J",
        x_col="log_J",
        x_label=r"$\log J$",
        y_qgt1=lambda x: x["log_runtime"] - x["two_log_R"] - x["log_NmN"],
        y_q1=lambda x:   x["log_runtime"] - x["two_log_R"] - x["log_mN"],
        y_label_qgt1=r"$\log(T/(R^2 N m^N))$",
        y_label_q1=r"$\log(T/(R^2 m^N))$",
    ),
    dict(
        direction="R2",
        x_col="two_log_R",
        x_label=r"$2\log R$",
        y_qgt1=lambda x: x["log_runtime"] - x["log_J"] - x["log_NmN"],
        y_q1=lambda x:   x["log_runtime"] - x["log_J"] - x["log_mN"],
        y_label_qgt1=r"$\log(T/(J N m^N))$",
        y_label_q1=r"$\log(T/(J m^N))$",
    ),
    dict(
        direction="truncation",
        x_col_qgt1="log_NmN",
        x_col_q1="log_mN",
        x_label_qgt1=r"$\log(Nm^N)$",
        x_label_q1=r"$\log(m^N)$",
        y_qgt1=lambda x: x["log_runtime"] - x["log_J"] - x["two_log_R"],
        y_q1=lambda x:   x["log_runtime"] - x["log_J"] - x["two_log_R"],
        y_label_qgt1=r"$\log(T/(J R^2))$",
        y_label_q1=r"$\log(T/(J R^2))$",
    ),
]


def run_individual_fits(df):
    fit_rows, fit_parts = [], []

    for family in ["qgt1", "q1"]:
        q_values = (sorted(df[df["family"] == family]["q"].unique())
                    if family == "qgt1" else [1])

        for spec in DIRECTION_SPECS:
            direction = spec["direction"]
            for q in q_values:
                g0 = matched_axis_data(df, family=family, q=q, direction=direction).copy()
                if len(g0) < 2:
                    continue

                if direction == "truncation":
                    x_col   = spec["x_col_qgt1"]   if family == "qgt1" else spec["x_col_q1"]
                    x_label = spec["x_label_qgt1"]  if family == "qgt1" else spec["x_label_q1"]
                else:
                    x_col, x_label = spec["x_col"], spec["x_label"]

                if family == "qgt1":
                    g0["partial_log_runtime"] = spec["y_qgt1"](g0)
                    y_label = spec["y_label_qgt1"]
                else:
                    g0["partial_log_runtime"] = spec["y_q1"](g0)
                    y_label = spec["y_label_q1"]

                g_fit, fit_info = fit_curve_group(g0, x_col=x_col, y_col="partial_log_runtime")
                g_fit["fit_family"]    = family
                g_fit["fit_q"]         = q
                g_fit["fit_direction"] = direction
                g_fit["fit_x_col"]     = x_col
                g_fit["fit_x_label"]   = x_label
                g_fit["fit_y_label"]   = y_label

                fit_rows.append(dict(
                    family=family, q=q, direction=direction,
                    x_col=x_col, x_label=x_label, y_label=y_label,
                    expected_slope=1.0,
                    slope=fit_info["slope"],
                    slope_error=fit_info["slope"] - 1.0,
                    intercept=fit_info["intercept"],
                    r2=fit_info["r2"],
                    n=fit_info["n"],
                    n_cube=int((g_fit["design"] == "cube").sum()),
                    n_ext=int((g_fit["design"] != "cube").sum()),
                    mean_abs_resid=fit_info["mean_abs_resid"],
                    max_abs_resid=fit_info["max_abs_resid"],
                ))
                fit_parts.append(g_fit)

    df_individual_fits = pd.DataFrame(fit_rows)
    df_individual_pred = pd.concat(fit_parts, ignore_index=True)
    return df_individual_fits, df_individual_pred

# ---------------------------------------------------------------------------
# Plots — fixed-slope model (Plots 1-8)
# ---------------------------------------------------------------------------

def plot_fixed_slope(df, df_qgt1, df_q1, df_q_constants, df_fixed_all, q1_C):
    # 1. Predicted vs measured.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    for ax, g_all, title in [
        (axes[0], df_qgt1,
         rf"$q>1$: fixed slopes, q-specific constants, "
         rf"$R^2={log_r2(df_qgt1['log_hot_time'], df_qgt1['pred_log_hot_time_fixed']):.4f}$"),
        (axes[1], df_q1,
         rf"$q=1$: fixed slopes, single constant, "
         rf"$R^2={log_r2(df_q1['log_hot_time'], df_q1['pred_log_hot_time_fixed']):.4f}$"),
    ]:
        for design, marker in [("cube","o"),("J_ext","s"),("R_ext","^"),("N_ext","D")]:
            g = g_all[g_all["design"] == design]
            if len(g):
                ax.scatter(g["runtime"], g["pred_hot_time_fixed"],
                           marker=marker, alpha=0.75, label=design)
        lo = min(g_all["runtime"].min(), g_all["pred_hot_time_fixed"].min())
        hi = max(g_all["runtime"].max(), g_all["pred_hot_time_fixed"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("measured hot runtime [s]")
        ax.set_ylabel("predicted hot runtime [s]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-slope scaling model: predicted vs measured")
    plt.tight_layout(); plt.show()

    # 2. q-dependent constants for q > 1.
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(df_q_constants["q"], df_q_constants["relative_C"], marker="o")
    finish_plot(title=r"Fitted constants in fixed-slope model for $q>1$",
                xlabel="q", ylabel="constant relative to q=2",
                yscale="log", legend=False)

    # 3. Log-constant estimates by q and design.
    plt.figure(figsize=(8.5, 5))
    for design, marker in [("cube","o"),("J_ext","s"),("R_ext","^"),("N_ext","D")]:
        g = df_qgt1[df_qgt1["design"] == design]
        if len(g):
            plt.scatter(g["q"], g["log_constant_estimate"],
                        marker=marker, alpha=0.65, label=design)
    for _, row in df_q_constants.iterrows():
        plt.hlines(row["log_C"], xmin=row["q"]-0.35, xmax=row["q"]+0.35,
                   color="black", linewidth=1.2)
    finish_plot(title=r"$q>1$: log-constant estimates after theoretical normalization",
                xlabel="q", ylabel=r"$\log T - \log(JR^2Nm^N)$",
                legend=True, legend_ncol=2)

    # 4. Residuals by family and design.
    plt.figure(figsize=(8.5, 5))
    for (family, design), g in df_fixed_all.groupby(["family", "design"]):
        x = np.full(len(g), f"{family}\n{design}")
        plt.scatter(x, g["fixed_resid_log"], alpha=0.7, label=f"{family} {design}")
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    finish_plot(title="Fixed-slope model residuals",
                xlabel=None, ylabel="log-runtime residual", legend=False)

    # 5. Axial extension raw curves — q > 1.
    for design in ["J_ext", "R_ext", "N_ext"]:
        varied = {"J_ext": "J", "R_ext": "R", "N_ext": "N"}[design]
        df_d = df_qgt1[df_qgt1["design"] == design]
        if not len(df_d):
            continue
        plt.figure(figsize=(8.5, 5))
        for q, g in df_d.groupby("q"):
            g = g.sort_values(varied)
            plt.plot(g[varied], g["runtime"], marker="o", label=f"q={q}")
        finish_plot(title=rf"$q>1$ axial extension: runtime vs {varied}",
                    xlabel=varied, ylabel="hot runtime [s]",
                    xscale="log" if varied in ("J","R") else None,
                    yscale="log", legend=True, legend_ncol=2)

    # 6. Axial extension raw curves — q = 1.
    for design in ["J_ext", "R_ext", "N_ext"]:
        varied = {"J_ext": "J", "R_ext": "R", "N_ext": "N"}[design]
        df_d = df_q1[df_q1["design"] == design]
        if not len(df_d):
            continue
        plt.figure(figsize=(7.5, 4.8))
        g = df_d.sort_values(varied)
        plt.plot(g[varied], g["runtime"], marker="o", label="q=1")
        finish_plot(title=rf"$q=1$ axial extension: runtime vs {varied}",
                    xlabel=varied, ylabel="hot runtime [s]",
                    xscale="log" if varied in ("J","R") else None,
                    yscale="log", legend=True)

    # 7. Normalized runtime diagnostics.
    df_norm = df_fixed_all.copy()
    df_norm["norm_q1"]   = df_norm["runtime"] / df_norm["cost_q1"]
    df_norm["norm_qgt1"] = df_norm["runtime"] / df_norm["cost_qgt1"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    g1 = df_norm[df_norm["family"] == "q1"]
    axes[0].scatter(g1["N"], g1["norm_q1"], alpha=0.7)
    axes[0].axhline(q1_C, linestyle="--", color="black", linewidth=1.2,
                    label="fitted constant")
    axes[0].set_title(r"$q=1$: $T / (J R^2 m^N)$")
    axes[0].set_xlabel("N"); axes[0].set_ylabel("normalized runtime")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25); axes[0].legend(fontsize=8)

    g2 = df_norm[df_norm["family"] == "qgt1"]
    for q, g in g2.groupby("q"):
        axes[1].scatter(g["N"], g["norm_qgt1"], alpha=0.7, label=f"q={q}")
    for _, row in df_q_constants.iterrows():
        axes[1].axhline(row["C"], linestyle="--", linewidth=0.9, alpha=0.4)
    axes[1].set_title(r"$q>1$: $T / (J R^2 N m^N)$")
    axes[1].set_xlabel("N"); axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Normalized hot-runtime diagnostics")
    plt.tight_layout(); plt.show()

    # 8. Construction / first-call / hot-time decomposition.
    df_time_bar = (
        df
        .groupby("family", as_index=False)
        .agg(
            wall_construct_s=("wall_construct_s",  "median"),
            wall_first_call_s=("wall_first_call_s", "median"),
            hot_time=("runtime", "median"),
        )
    )
    x = np.arange(len(df_time_bar))
    width = 0.25
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(x - width, df_time_bar["wall_construct_s"],  width, label="construction")
    plt.bar(x,         df_time_bar["wall_first_call_s"], width, label="first call")
    plt.bar(x + width, df_time_bar["hot_time"],          width, label="hot execution")
    plt.xticks(x, df_time_bar["family"])
    finish_plot(title="Median timing decomposition by family",
                xlabel=None, ylabel="time [s]", yscale="log", legend=True)


# ---------------------------------------------------------------------------
# Plots — individual curve fits (Plots B and C)
# ---------------------------------------------------------------------------

def plot_individual_fits(df_individual_fits, df_individual_pred):
    # Plot B: q > 1
    for direction in ["J", "R2", "truncation"]:
        df_d = df_individual_pred[
            (df_individual_pred["fit_family"]    == "qgt1") &
            (df_individual_pred["fit_direction"] == direction)
        ]
        if not len(df_d):
            continue
        x_col   = df_d["fit_x_col"].iloc[0]
        x_label = df_d["fit_x_label"].iloc[0]
        y_label = df_d["fit_y_label"].iloc[0]
        plt.figure(figsize=(9, 5.5))
        for q, g in df_d.groupby("fit_q"):
            g = g.sort_values(x_col)
            fit_row = df_individual_fits[
                (df_individual_fits["family"]    == "qgt1") &
                (df_individual_fits["q"]         == q) &
                (df_individual_fits["direction"] == direction)
            ].iloc[0]
            g_cube = g[g["design"] == "cube"]
            g_ext  = g[g["design"] != "cube"]
            if len(g_cube):
                plt.plot(g_cube[x_col], g_cube["partial_log_runtime"],
                         marker="o", linestyle="", alpha=0.85, label=rf"$q={q}$ cube")
            if len(g_ext):
                plt.plot(g_ext[x_col], g_ext["partial_log_runtime"],
                         marker="s", linestyle="", alpha=0.85, label=rf"$q={q}$ ext")
            x_line = np.linspace(g[x_col].min(), g[x_col].max(), 100)
            plt.plot(x_line, fit_row["intercept"] + fit_row["slope"] * x_line,
                     linestyle="--",
                     label=rf"$q={q}$ fit, slope={fit_row['slope']:.2f}")
        finish_plot(title=rf"$q>1$: matched individual {direction} scaling",
                    xlabel=x_label, ylabel=y_label, legend=True, legend_ncol=2)

    # Plot C: q = 1
    for direction in ["J", "R2", "truncation"]:
        df_d = df_individual_pred[
            (df_individual_pred["fit_family"]    == "q1") &
            (df_individual_pred["fit_direction"] == direction)
        ]
        if not len(df_d):
            continue
        x_col   = df_d["fit_x_col"].iloc[0]
        x_label = df_d["fit_x_label"].iloc[0]
        y_label = df_d["fit_y_label"].iloc[0]
        g = df_d.sort_values(x_col)
        fit_row = df_individual_fits[
            (df_individual_fits["family"]    == "q1") &
            (df_individual_fits["direction"] == direction)
        ].iloc[0]
        plt.figure(figsize=(7.8, 5.0))
        g_cube = g[g["design"] == "cube"]
        g_ext  = g[g["design"] != "cube"]
        if len(g_cube):
            plt.plot(g_cube[x_col], g_cube["partial_log_runtime"],
                     marker="o", linestyle="", alpha=0.85, label="cube")
        if len(g_ext):
            plt.plot(g_ext[x_col], g_ext["partial_log_runtime"],
                     marker="s", linestyle="", alpha=0.85, label="extension")
        x_line = np.linspace(g[x_col].min(), g[x_col].max(), 100)
        plt.plot(x_line, fit_row["intercept"] + fit_row["slope"] * x_line,
                 linestyle="--",
                 label=rf"fit, slope={fit_row['slope']:.2f}, $R^2={fit_row['r2']:.3f}$")
        finish_plot(title=rf"$q=1$: matched individual {direction} scaling",
                    xlabel=x_label, ylabel=y_label, legend=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    in_pkl = args.output_dir / f"fssk_exact_scaling_timings_{args.regime.lower()}.pkl"
    print(f"Loading {in_pkl} …")
    df_raw = pd.read_pickle(in_pkl)
    print(f"  {len(df_raw)} rows  |  columns: {list(df_raw.columns)}\n")

    # ── Prepare ──────────────────────────────────────────────────────────
    df, TIME_COL = prepare_timing_data(df_raw)

    m_values = sorted(df["m"].unique())
    assert len(m_values) == 1, f"Expected single m, got {m_values}"
    m = m_values[0]

    print(f"\nFixed q > 1 model:  T ≈ C_q  J  R²  N  m^N")
    print(f"Fixed q = 1 model:  T ≈ C    J  R²  m^N\n")

    # ── Fixed-slope model ─────────────────────────────────────────────────
    df_qgt1, df_q1, df_q_constants, df_fixed_all, q1_C = fit_fixed_slope(df)

    print(f"q = 1 fitted constant C: {q1_C:.6g}")
    print("q > 1 fitted constants:")
    print(df_q_constants.to_string(index=False))

    df_fixed_summary = fixed_model_summary(df_fixed_all)
    print("\nFixed-model summary:")
    print(df_fixed_summary.to_string(index=False))

    df_resid_summary = (
        df_fixed_all
        .assign(abs_log_resid=lambda x: np.abs(x["fixed_resid_log"]))
        .groupby(["family","design"], as_index=False)
        .agg(
            n=("run_id","count"),
            mean_abs_log_resid=("abs_log_resid","mean"),
            max_abs_log_resid=("abs_log_resid","max"),
            mean_abs_rel_error=("fixed_rel_error", lambda x: float(np.mean(np.abs(x)))),
            max_abs_rel_error= ("fixed_rel_error", lambda x: float(np.max(np.abs(x)))),
        )
    )
    print("\nResidual summary by design:")
    print(df_resid_summary.to_string(index=False))

    df_time_summary = (
        df
        .groupby(["family","design"], as_index=False)
        .agg(
            n=("run_id","count"),
            mean_construct_time=("wall_construct_s",  "mean"),
            mean_first_call_time=("wall_first_call_s", "mean"),
            mean_hot_time=("runtime","mean"),
            median_hot_time=("runtime","median"),
            max_hot_time=("runtime","max"),
        )
    )
    print("\nTiming summary:")
    print(df_time_summary.to_string(index=False))

    df_constant_by_design = (
        df_fixed_all
        .groupby(["family","design","q"], as_index=False)
        .agg(
            mean_log_constant=("log_constant_estimate","mean"),
            std_log_constant= ("log_constant_estimate","std"),
            n=("log_constant_estimate","count"),
        )
    )
    df_constant_by_design["constant"] = np.exp(df_constant_by_design["mean_log_constant"])
    print("\nLog-constant by design:")
    print(df_constant_by_design.to_string(index=False))

    # ── Fixed-slope plots ─────────────────────────────────────────────────
    plot_fixed_slope(df, df_qgt1, df_q1, df_q_constants, df_fixed_all, q1_C)

    # ── Individual per-axis fits ──────────────────────────────────────────
    df_individual_fits, df_individual_pred = run_individual_fits(df)

    print("\nIndividual scaling fits:")
    print(df_individual_fits.to_string(index=False))
    print()
    print(df_individual_fits.pivot_table(
        index=["family","q"], columns="direction", values="slope"
    ).reset_index().to_string(index=False))

    # ── Individual-fit plots ──────────────────────────────────────────────
    plot_individual_fits(df_individual_fits, df_individual_pred)

    # ── Final compact printout ────────────────────────────────────────────
    print(f"\nRegime: {args.regime}  |  runtime column: {TIME_COL}")
    print("\nFixed-slope q > 1:  T ≈ C_q  J  R²  N  m^N")
    print(f"  log-space R²         : {log_r2(df_qgt1['log_hot_time'], df_qgt1['pred_log_hot_time_fixed']):.6f}")
    print(f"  mean |log residual|  : {float(np.mean(np.abs(df_qgt1['fixed_resid_log']))):.6f}")
    print(f"  max  |log residual|  : {float(np.max(np.abs(df_qgt1['fixed_resid_log']))):.6f}")
    print("\nFixed-slope q = 1:  T ≈ C  J  R²  m^N")
    print(f"  fitted C             : {q1_C:.6g}")
    print(f"  log-space R²         : {log_r2(df_q1['log_hot_time'], df_q1['pred_log_hot_time_fixed']):.6f}")
    print(f"  mean |log residual|  : {float(np.mean(np.abs(df_q1['fixed_resid_log']))):.6f}")
    print(f"  max  |log residual|  : {float(np.max(np.abs(df_q1['fixed_resid_log']))):.6f}")


if __name__ == "__main__":
    main()

