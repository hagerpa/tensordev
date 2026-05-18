"""
analyse_kernel_flop.py — Analyse and plot FSSK kernel FLOP scaling results.

Reads the pickle produced by sweep_kernel_flop.py and generates:

  Figure 1 (flop_scaling)
    Log-log scatter: XLA FLOPs vs. expected cost J^2 × R × f(R)
    with least-squares fit to determine the empirical exponent on R.

  Figure 2 (compile_time)
    Compile time vs. problem size for ETD1 and Heun.

  Figure 3 (precomp_vs_pde)
    Log-log scatter comparing precomp FLOPs, PDE-by-subtraction FLOPs, and
    total kernel FLOPs, to visualise which stage dominates.

Also writes:
    kernel_flop_summary.csv   — full results table
    kernel_flop_fit.csv       — fitted cost-model parameters

Usage
-----
    python analyse_kernel_flop.py                      # auto-detects most recent
    python analyse_kernel_flop.py --regime SMALL       # analyses small regime
    python analyse_kernel_flop.py --input path/to/kernel_flop_scaling_medium.pkl
    python analyse_kernel_flop.py --no-plots
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

import _validation_util.plot_config as _pc          # noqa: F401
from _validation_util.plot_config import new_fig, savefig_fig, COLORS, MARKERS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse kernel FLOP scaling sweep")
    p.add_argument("--input", type=Path, default=None,
                   help="Pickle from sweep_kernel_flop.py "
                        "(default: auto-detect most recent kernel_flop_scaling_*.pkl)")
    p.add_argument("--regime", type=str, default=None,
                   help="Regime name to analyse (SMALL/MEDIUM/LARGE). "
                        "If not specified, auto-detects the most recent file.")
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

    # Determine input file
    if args.input:
        pkl_path = args.input
    elif args.regime:
        # Use specified regime
        regime_tag = args.regime.lower()
        pkl_path = args.output_dir / f"kernel_flop_scaling_{regime_tag}.pkl"
    else:
        # Auto-detect: find most recent kernel_flop_scaling_*.pkl
        pattern = args.output_dir / "kernel_flop_scaling_*.pkl"
        candidates = list(args.output_dir.glob("kernel_flop_scaling_*.pkl"))
        if candidates:
            # Sort by modification time, most recent first
            pkl_path = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"Auto-detected input: {pkl_path.name}")
        else:
            # Fallback to medium (old behavior) for error message
            pkl_path = args.output_dir / "kernel_flop_scaling_medium.pkl"

    if not pkl_path.exists():
        available = list(args.output_dir.glob("kernel_flop_scaling_*.pkl"))
        avail_msg = f"\nAvailable files: {[p.name for p in available]}" if available else ""
        raise FileNotFoundError(
            f"Input not found: {pkl_path}\n"
            f"Run sweep_kernel_flop.py first, or pass --input or --regime explicitly.{avail_msg}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(pkl_path)

    # ── Experiment metadata ───────────────────────────────────────────────────
    row0 = df.iloc[0]
    regime_name = str(row0.get("regime", ""))
    print("── Experiment metadata ─────────────────────────────────────────────────")
    for key in ("regime", "n_paths", "d", "q", "m", "jordan_form"):
        if key in df.columns:
            print(f"  {key} = {row0[key]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────────")
    print(f"  Total configs: {len(df)}")
    print(f"  Schemes: {sorted(df['scheme'].unique())}")
    print(f"  J range: {df['J'].min()}..{df['J'].max()}")
    print(f"  R range: {df['R'].min()}..{df['R'].max()}")
    print(f"  dyadic range: {df['dyadic'].min()}..{df['dyadic'].max()}")

    # ── Cost model fit ────────────────────────────────────────────────────────
    # Expected: FLOPs ≈ c × J^2 × R × (sum block_sizes^2)
    # For simplicity, fit:  log(FLOPs) = a + 2·log(J_eff) + b·log(R)
    # where J_eff = J × 2^dyadic, and b measures the R-dependency exponent.

    df = df.copy()
    df["J_eff"] = df["J"] * (2.0 ** df["dyadic"])
    df = df[df["xla_flops"].notna() & (df["xla_flops"] > 0)].copy()

    fit_rows = []
    print("\n── Fitted cost model ───────────────────────────────────────────────────")
    for scheme in df["scheme"].unique():
        sub = df[df["scheme"] == scheme].copy()
        if sub.empty:
            continue

        # Prefer PDE-only FLOPs for the fit; fall back to total
        flop_col = "pde_xla_flops" if "pde_xla_flops" in sub.columns else "xla_flops"
        sub[flop_col] = pd.to_numeric(sub[flop_col], errors="coerce")
        sub = sub[sub[flop_col].notna() & (sub[flop_col] > 0)].copy()
        if sub.empty:
            continue

        # lax.scan correction: multiply one-body cost by (J-1)^2 scan steps
        sub["pde_flops_corrected"] = pd.to_numeric(sub[flop_col], errors="coerce") * (sub["J_eff"] - 1) ** 2
        sub = sub[sub["pde_flops_corrected"].notna() & (sub["pde_flops_corrected"] > 0)].copy()
        if sub.empty:
            continue

        log_flops = np.log(sub["pde_flops_corrected"].to_numpy(dtype=float))
        log_J_eff = np.log(sub["J_eff"].to_numpy(dtype=float))
        log_R     = np.log(sub["R"].to_numpy(dtype=float))
        log_q     = np.log(sub["q"].to_numpy(dtype=float))

        # OLS fit: log(FLOPs) = intercept + c_J·log(J_eff) + c_R·log(R) + c_q·log(q)
        X = np.column_stack([np.ones_like(log_J_eff), log_J_eff, log_R, log_q])
        coeffs, residuals, rank, s = np.linalg.lstsq(X, log_flops, rcond=None)
        intercept, c_J, c_R, c_q = coeffs

        ss_res = residuals[0] if residuals.size > 0 else np.sum((log_flops - X @ coeffs)**2)
        ss_tot = np.sum((log_flops - log_flops.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"  {scheme}:  log(FLOPs) = {intercept:.2f} + {c_J:.2f}·log(J_eff) + {c_R:.2f}·log(R) + {c_q:.2f}·log(q)")
        print(f"           R² = {r_squared:.4f}  (expect c_q ≈ 2, c_J ≈ 2, c_R ≈ 2)")

        fit_rows.append(dict(
            scheme=scheme,
            intercept=intercept,
            coef_log_J_eff=c_J,
            coef_log_R=c_R,
            coef_log_q=c_q,
            r_squared=r_squared,
            n_samples=len(sub),
        ))

    df_fit = pd.DataFrame(fit_rows)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    path_summary = args.output_dir / "kernel_flop_summary.csv"
    path_fit     = args.output_dir / "kernel_flop_fit.csv"
    df.to_csv(path_summary, index=False)
    df_fit.to_csv(path_fit, index=False)
    print(f"\nSaved:\n  {path_summary}\n  {path_fit}")

    # ── Precomp vs PDE breakdown ──────────────────────────────────────────────
    if "precomp_xla_flops" in df.columns and "pde_by_subtraction_xla_flops" in df.columns:
        sub_h = df[df["scheme"] == "heun"].copy()
        sub_h["precomp_xla_flops"] = pd.to_numeric(sub_h["precomp_xla_flops"], errors="coerce")
        sub_h["pde_by_subtraction_xla_flops"] = pd.to_numeric(
            sub_h["pde_by_subtraction_xla_flops"], errors="coerce"
        )
        valid_mask = (
            sub_h["xla_flops"].notna() & (sub_h["xla_flops"] > 0)
            & sub_h["precomp_xla_flops"].notna() & (sub_h["precomp_xla_flops"] > 0)
            & sub_h["pde_by_subtraction_xla_flops"].notna() & (sub_h["pde_by_subtraction_xla_flops"] > 0)
        )
        sub_h = sub_h[valid_mask]
        if not sub_h.empty:
            sub_h["precomp_fraction"] = sub_h["precomp_xla_flops"] / sub_h["xla_flops"]
            sub_h["pde_fraction"] = sub_h["pde_by_subtraction_xla_flops"] / sub_h["xla_flops"]
            print("\n── Precomp vs PDE-solver cost breakdown (Heun) ─────────────────────────")
            print(f"  rows with valid precomp+pde data : {len(sub_h)}")
            print(f"  precomp fraction  mean={sub_h['precomp_fraction'].mean():.3f}  "
                  f"median={sub_h['precomp_fraction'].median():.3f}  "
                  f"min={sub_h['precomp_fraction'].min():.3f}  "
                  f"max={sub_h['precomp_fraction'].max():.3f}")
            print(f"  pde fraction      mean={sub_h['pde_fraction'].mean():.3f}  "
                  f"median={sub_h['pde_fraction'].median():.3f}  "
                  f"min={sub_h['pde_fraction'].min():.3f}  "
                  f"max={sub_h['pde_fraction'].max():.3f}")

    if args.no_plots:
        return

    # =========================================================================
    # Figure 1 — FLOP scaling: observed vs. expected
    # =========================================================================
    fig, ax = new_fig("half")

    sub = df[df["scheme"] == "heun"].copy()
    pde_col = "pde_xla_flops" if "pde_xla_flops" in sub.columns else "xla_flops"
    sub[pde_col] = pd.to_numeric(sub[pde_col], errors="coerce")
    sub["precomp_xla_flops"] = pd.to_numeric(sub.get("precomp_xla_flops", np.nan), errors="coerce")
    sub = sub[sub[pde_col].notna() & (sub[pde_col] > 0)].copy()

    if not sub.empty:
        # lax.scan correction for PDE part; precomp already fully counted
        sub["pde_flops_corrected"] = sub[pde_col] * (sub["J_eff"] - 1) ** 2
        sub["total_flops"] = sub["precomp_xla_flops"].fillna(0) + sub["pde_flops_corrected"]
        sub["expected_cost"] = (sub["J_eff"] ** 2) * (sub["R"] ** 2)
        q_vals = sorted(sub["q"].unique())
        x_line = np.geomspace(sub["expected_cost"].min(), sub["expected_cost"].max(), 200)

        print(f"\n── Global fit  total FLOPs = C·J²R² ───────────────────────────────────")
        log_C = np.mean(
            np.log(sub["total_flops"].to_numpy(float))
            - np.log(sub["expected_cost"].to_numpy(float))
        )
        C = np.exp(log_C)
        print(f"  C = {C:.4g}")

        for i, q_val in enumerate(q_vals):
            q_sub = sub[sub["q"] == q_val]
            ax.loglog(q_sub["expected_cost"], q_sub["total_flops"],
                      marker=MARKERS[0], linestyle="",
                      color=COLORS[0], alpha=0.4, markersize=3)

        ax.loglog(x_line, C * x_line,
                  linestyle="--", color="lightsteelblue", linewidth=1.0, alpha=0.85,
                  label=r"$\hat{C}\cdot J^2 R^2$")

    ax.set_xlabel(r"$J^2 R^2$")
    ax.set_ylabel("XLA FLOPs")
    ax.set_title(r"FSSK $\mathrm{VSig}$-kernel — FLOP Count")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.15)
    stem = args.output_dir / "kernel_flop_scaling"
    savefig_fig(fig, stem, args.formats)
    plt.close(fig)

    # =========================================================================
    # Figure 2 — Compile time
    # =========================================================================
    fig, ax = new_fig("half")

    sub = df[df["scheme"] == "heun"].copy()
    if not sub.empty:
        sub = sub.sort_values("xla_compile_time_s")
        style = dict(color=COLORS[0], marker=MARKERS[0])
        ax.plot(range(len(sub)), sub["xla_compile_time_s"],
                marker=style["marker"], linestyle="", color=style["color"],
                alpha=0.6, markersize=3, label="Heun")

    ax.set_xlabel("Configuration index (sorted by compile time)")
    ax.set_ylabel("XLA compile time (s)")
    ax.set_title("Kernel XLA compile time (Heun, dyadic=0)")
    ax.legend()
    stem = args.output_dir / "kernel_flop_compile_time"
    savefig_fig(fig, stem, args.formats)
    plt.close(fig)

    # =========================================================================
    # Figure 3 — Precomp vs PDE-solver FLOP breakdown
    # =========================================================================
    if "precomp_xla_flops" in df.columns and "pde_by_subtraction_xla_flops" in df.columns:
        sub_h = df[df["scheme"] == "heun"].copy()
        sub_h["precomp_xla_flops"] = pd.to_numeric(sub_h["precomp_xla_flops"], errors="coerce")
        sub_h["pde_by_subtraction_xla_flops"] = pd.to_numeric(
            sub_h["pde_by_subtraction_xla_flops"], errors="coerce"
        )
        valid_mask = (
            sub_h["xla_flops"].notna() & (sub_h["xla_flops"] > 0)
            & sub_h["precomp_xla_flops"].notna() & (sub_h["precomp_xla_flops"] > 0)
            & sub_h["pde_by_subtraction_xla_flops"].notna()
            & (sub_h["pde_by_subtraction_xla_flops"] > 0)
        )
        sub_h = sub_h[valid_mask]

        if not sub_h.empty:
            sub_h["expected_cost"] = (sub_h["J_eff"] ** 2) * (sub_h["R"] ** 2)
            q_vals = sorted(sub_h["q"].unique())

            fig, ax = new_fig("half")
            for i, q_val in enumerate(q_vals):
                color = COLORS[i % len(COLORS)]
                qs = sub_h[sub_h["q"] == q_val]
                ax.loglog(
                    qs["expected_cost"], qs["xla_flops"],
                    marker=MARKERS[i % len(MARKERS)], linestyle="",
                    color=color, alpha=0.7, markersize=3,
                    label=fr"total $q={q_val}$",
                )
                ax.loglog(
                    qs["expected_cost"], qs["precomp_xla_flops"],
                    marker=MARKERS[i % len(MARKERS)], linestyle="",
                    color=color, alpha=0.4, markersize=3,
                    label=fr"precomp $q={q_val}$",
                )
                ax.loglog(
                    qs["expected_cost"], qs["pde_by_subtraction_xla_flops"],
                    marker="x", linestyle="",
                    color=color, alpha=0.55, markersize=3,
                    label=fr"PDE $q={q_val}$",
                )

            ax.set_xlabel(r"$J^2 R^2$")
            ax.set_ylabel("XLA FLOPs")
            ax.set_title(r"FSSK — Precomp vs PDE-solver FLOP split")
            ax.legend(ncol=len(q_vals), loc="upper left", fontsize="xx-small")
            ax.grid(True, which="both", alpha=0.15)
            stem = args.output_dir / "kernel_flop_precomp_vs_pde"
            savefig_fig(fig, stem, args.formats)
            plt.close(fig)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()

