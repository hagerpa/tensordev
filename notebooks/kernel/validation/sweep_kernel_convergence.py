"""
sweep_kernel_convergence.py — FSSK signature-kernel convergence sweep.

Paths
-----
Two independent batches of n_paths 3-D unit-speed paths (X and Y) are drawn.
All kernel values are batchwise:  K[i] = k(X_i, Y_i),  shape (n_paths,).
This is far cheaper than the full n_paths×n_paths Gram matrix.

Methods swept
-------------
  1. vsig_trunc  — VSig inner product,       trunc ∈ {1, …, max_trunc}
  2. naive_euler — Naive forward-Euler PDE,  dyadic ∈ {0, …, max_dyadic}
  3. etd1        — ETD1 scheme,              dyadic ∈ {0, …, max_dyadic}
  4. heun        — Heun scheme,              dyadic ∈ {0, …, max_dyadic}

Reference: Heun at max_dyadic.

Iteration order (cheapest first)
---------------------------------
Configurations are grouped by cost level l = 0, 1, …, max_dyadic:

  level l  →  PDE methods at dyadic=l
           +  vsig at trunc = 2l+1 and 2l+2  (if ≤ max_trunc)
    so light configs of all methods are always collected first.

Regimes
-------
  SMALL   max_dyadic=3, max_trunc=8,  n_paths=100, J=33
  MEDIUM  max_dyadic=5, max_trunc=12, n_paths=100, J=33
  LARGE   max_dyadic=6, max_trunc=14, n_paths=100, J=33

Output (written to --output-dir)
----------------------------------
  kernel_conv_sweep.pkl          — final result DataFrame
  kernel_conv_sweep_partial.pkl  — live checkpoint (overwritten after every row)

Usage
-----
    python sweep_kernel_convergence.py
    python sweep_kernel_convergence.py --regime SMALL
    python sweep_kernel_convergence.py --regime LARGE --n-paths 50
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))              # notebooks/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sss" / "validation"))  # fssk_setup
sys.path.insert(0, str(Path(__file__).resolve().parent))                  # validation/

from tensordev.sss.kernel import FSSK
from tensordev.kernel.fssk import fssk_sigkernel
from tensordev.util.random_paths import unit_speed_paths

from fssk_setup import random_fssk          # shared FSSK parameter generator
from vsig_kernel_ref import vsig_kernel
from naive_euler import naive_euler_fssk_kernel

from _validation_util.timing_utils import time_call, time_warmup


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelRegime:
    name:        str
    max_dyadic:  int
    max_trunc:   int
    n_paths:     int = 100
    J:           int = 65    # nodes; intervals = J-1; dt = 1/(J-1)
    desc:        str = ""


REGIMES: dict[str, KernelRegime] = {
    "SMALL":  KernelRegime("SMALL",  max_dyadic=4, max_trunc=10,
                            desc="fast dev run — p=0..3, N=1..10"),
    "MEDIUM": KernelRegime("MEDIUM", max_dyadic=5, max_trunc=12,
                            desc="standard validation — p=0..5, N=1..12"),
    "LARGE":  KernelRegime("LARGE",  max_dyadic=6, max_trunc=14,
                            desc="full production sweep — p=0..6, N=1..14"),
}


# ---------------------------------------------------------------------------
# Workload-ordered plan
# ---------------------------------------------------------------------------

_TRUNC_PER_LEVEL = 2    # vsig truncation levels paired with each dyadic level
_PDE_METHODS     = ["naive_euler", "etd1", "heun"]


def build_plan(regime: KernelRegime) -> list[dict]:
    """Return configs ordered cheapest-first.

    Level l  →  PDE methods at dyadic=l
             +  vsig at trunc = 2l+1,  2l+2  (if ≤ max_trunc)
    """
    plan: list[dict] = []
    for l in range(regime.max_dyadic + 1):
        # PDE methods (ordered by cost within the level)
        for method in _PDE_METHODS:
            plan.append(dict(
                workload_level = l,
                method         = method,
                param_type     = "dyadic_order",
                param_value    = l,
            ))
        # Paired VSig truncation levels
        for k in range(_TRUNC_PER_LEVEL):
            trunc = l * _TRUNC_PER_LEVEL + k + 1
            if 1 <= trunc <= regime.max_trunc:
                plan.append(dict(
                    workload_level = l,
                    method         = "vsig_trunc",
                    param_type     = "trunc",
                    param_value    = trunc,
                ))
    return plan


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FSSK kernel convergence sweep — vsig / naive-euler / etd1 / heun"
    )
    p.add_argument("--regime",        choices=list(REGIMES), default="MEDIUM",
                   help="Preset regime (SMALL / MEDIUM / LARGE). Default: MEDIUM.")
    # Sweep overrides
    p.add_argument("--n-paths",       type=int,   default=None,
                   help="Override number of paths per batch.")
    p.add_argument("--J",             type=int,   default=None,
                   help="Override number of path nodes.")
    p.add_argument("--max-trunc",     type=int,   default=None,
                   help="Override max VSig truncation level.")
    p.add_argument("--max-dyadic",    type=int,   default=None,
                   help="Override max dyadic order.")
    # Kernel parameters (forwarded to random_fssk from fssk_setup)
    p.add_argument("--R",             type=int,   default=4,
                   help="State-space dimension R. Default: 4.")
    p.add_argument("--q",             type=int,   default=1,
                   help="Number of FSSK components q. Default: 1.")
    p.add_argument("--m",             type=int,   default=3,
                   help="Latent-path dimension m (A has shape q×m×d). Default: 3.")
    p.add_argument("--eig-min",       type=float, default=0.0,
                   help="Minimum eigenvalue of Lambda. Default: 0.0.")
    p.add_argument("--eig-max",       type=float, default=1.0,
                   help="Maximum eigenvalue of Lambda. Default: 1.0.")
    p.add_argument("--jordan-alpha",  type=float, default=0.25,
                   help="Super-diagonal Jordan coupling. Default: 0.25.")
    # Seeds
    p.add_argument("--seed-x",        type=int,   default=20260514,
                   help="RNG seed for X paths.")
    p.add_argument("--seed-y",        type=int,   default=20260515,
                   help="RNG seed for Y paths (independent of X).")
    p.add_argument("--seed-kernel",   type=int,   default=42,
                   help="RNG seed for kernel parameters.")
    p.add_argument("--output-dir",    type=Path,
                   default=Path(__file__).parent / "validation_outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    regime = REGIMES[args.regime]

    # Apply overrides
    n_paths    = args.n_paths    or regime.n_paths
    J          = args.J          or regime.J
    max_trunc  = args.max_trunc  or regime.max_trunc
    max_dyadic = args.max_dyadic or regime.max_dyadic
    R, q, m    = args.R, args.q, args.m
    d          = 3
    dt         = 1.0 / (J - 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_pkl         = args.output_dir / "kernel_conv_sweep.pkl"
    out_pkl_partial = args.output_dir / "kernel_conv_sweep_partial.pkl"

    # Build an effective regime for printing / plan
    effective = KernelRegime(
        name        = args.regime,
        max_dyadic  = max_dyadic,
        max_trunc   = max_trunc,
        n_paths     = n_paths,
        J           = J,
        desc        = regime.desc,
    )

    _log("═" * 70)
    _log(f"FSSK kernel convergence sweep  —  regime {effective.name}  ({regime.desc})")
    _log(f"  n_paths={n_paths}  J={J}  d={d}  dt={dt:.6f}")
    _log(f"  max_trunc={max_trunc}  max_dyadic={max_dyadic}  (reference: vsig_trunc N={max_trunc})")
    _log(f"  R={R}  q={q}  m={m}  eig=[{args.eig_min}, {args.eig_max}]  seed_kernel={args.seed_kernel}")
    _log(f"  seed_x={args.seed_x}  seed_y={args.seed_y}")
    _log(f"  output_dir={args.output_dir}")
    _log("═" * 70)

    # ── Generate two independent path batches ─────────────────────────────────
    _log("Generating path batches …")
    X = unit_speed_paths(dt=dt, dt_fine=dt / 8, n_paths=n_paths, dim=d, seed=args.seed_x)
    Y = unit_speed_paths(dt=dt, dt_fine=dt / 8, n_paths=n_paths, dim=d, seed=args.seed_y)
    _log(f"  X.shape={X.shape}  Y.shape={Y.shape}   (batchwise: k(X_i, Y_i))")

    # ── Kernel (via shared random_fssk from fssk_setup) ───────────────────────
    kernel = random_fssk(
        q=q, R=R, m=m, d=d,
        seed=args.seed_kernel,
        eig_min=args.eig_min,
        eig_max=args.eig_max,
    )
    _log(f"Kernel: R={R}  q={q}  m={m}  seed={args.seed_kernel}  "
         f"A.shape={kernel.A.shape}  b.shape={kernel.b.shape}")

    # ── Build workload-ordered plan ───────────────────────────────────────────
    plan       = build_plan(effective)
    total_rows = len(plan)
    _log(f"Plan: {total_rows} configurations  (ordered cheapest-first)")
    _log("")

    # Print plan summary
    for cfg in plan:
        _log(f"  level={cfg['workload_level']}  {cfg['method']:<12s}  "
             f"{cfg['param_type']}={cfg['param_value']}")
    _log("")

    # ── Reference: VSig inner product at max_trunc ────────────────────────────
    # VSig at max_trunc is the most accurate reference: it is the exact
    # signature series truncated at the highest level we sweep, and is fast to
    # compute (no PDE grid required).
    ref_trunc = max_trunc
    _log(f"Computing reference (VSig inner product, trunc={ref_trunc}) …")

    def _vsig(trunc):
        return vsig_kernel(
            X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
            trunc=trunc, pairwise=False,
        )

    def _heun(dyadic):
        return fssk_sigkernel(
            X, Y,
            kernel=kernel, dt_x=dt, dt_y=dt,
            evaluate="terminal", pairwise=False,
            backend="scan", scheme="heun", dyadic_order=dyadic,
        )

    K_ref_jax, t_ref = time_call(_vsig, ref_trunc)
    K_ref = np.asarray(K_ref_jax)
    _log(f"  Reference done.  K_ref.shape={K_ref.shape}  wall={t_ref.wall_s:.2f}s")
    _log("")

    # ── Sweep ────────────────────────────────────────────────────────────────
    rows:     list[dict] = []
    run_id    = 0
    sweep_t0  = time.perf_counter()

    def _eta() -> str:
        done = len(rows)
        if done == 0:
            return "ETA ??"
        elapsed   = time.perf_counter() - sweep_t0
        remaining = elapsed / done * (total_rows - done)
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        return f"{done}/{total_rows}  ETA {h:02d}h{m:02d}m{s:02d}s"

    def _checkpoint() -> None:
        pd.DataFrame(rows).to_pickle(out_pkl_partial)

    def run_config(cfg: dict) -> None:
        nonlocal run_id
        method      = cfg["method"]
        param_type  = cfg["param_type"]
        param_value = cfg["param_value"]

        # ── Build callable ────────────────────────────────────────────────────
        if method == "vsig_trunc":
            fn = lambda: _vsig(param_value)
        elif method == "naive_euler":
            fn = lambda: naive_euler_fssk_kernel(
                X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
                pairwise=False, dyadic_order=param_value,
            )
        elif method == "etd1":
            fn = lambda: fssk_sigkernel(
                X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
                evaluate="terminal", pairwise=False,
                backend="scan", scheme="etd1", dyadic_order=param_value,
            )
        elif method == "heun":
            fn = lambda: _heun(param_value)
        else:
            raise ValueError(f"Unknown method: {method}")

        # ── Time it ───────────────────────────────────────────────────────────
        result_first, t_first = time_call(fn)
        time_warmup(fn, n_warmup=1)
        result_hot,   t_hot   = time_call(fn)

        K        = np.asarray(result_first)
        diff     = np.abs(K - K_ref)
        max_err  = float(diff.max())
        mean_err = float(diff.mean())

        row = dict(
            run_id            = run_id,
            regime            = effective.name,
            workload_level    = cfg["workload_level"],
            method            = method,
            param_type        = param_type,
            param_value       = param_value,
            # meta
            n_paths           = n_paths,
            J                 = J,
            d                 = d,
            q                 = q,
            R                 = R,
            m                 = m,
            eig_min           = args.eig_min,
            eig_max           = args.eig_max,
            jordan_alpha      = args.jordan_alpha,
            dt                = dt,
            seed_kernel       = args.seed_kernel,
            seed_x            = args.seed_x,
            seed_y            = args.seed_y,
            ref_method        = "vsig_trunc",
            ref_trunc         = ref_trunc,
            # error
            max_abs_error     = max_err,
            mean_abs_error    = mean_err,
            # timing — first call (JIT + execution)
            wall_first_s      = t_first.wall_s,
            cpu_first_s       = t_first.cpu_s,
            ru_first_utime_s  = t_first.ru_utime_s,
            ru_first_stime_s  = t_first.ru_stime_s,
            # timing — hot call
            wall_hot_s        = t_hot.wall_s,
            cpu_hot_s         = t_hot.cpu_s,
            ru_hot_utime_s    = t_hot.ru_utime_s,
            ru_hot_stime_s    = t_hot.ru_stime_s,
            ru_hot_nvcsw      = t_hot.ru_nvcsw,
            ru_hot_nivcsw     = t_hot.ru_nivcsw,
            ru_hot_minflt     = t_hot.ru_minflt,
            ru_hot_majflt     = t_hot.ru_majflt,
        )
        rows.append(row)
        run_id += 1
        _checkpoint()

        ref_note = "  ← ref" if (method == "vsig_trunc" and param_value == ref_trunc) else ""
        _log(
            f"  lvl={cfg['workload_level']}  {method:<12s}  "
            f"{param_type}={param_value:<3d}  "
            f"max_err={max_err:.3e}  "
            f"wall_first={t_first.wall_s:.2f}s  "
            f"wall_hot={t_hot.wall_s:.3f}s  "
            f"[{_eta()}]"
            + ref_note
        )

    # ── Iterate by workload level ─────────────────────────────────────────────
    prev_level = -1
    for cfg in plan:
        if cfg["workload_level"] != prev_level:
            _log(f"── Workload level {cfg['workload_level']} " + "─" * 55)
            prev_level = cfg["workload_level"]
        run_config(cfg)

    # ── Final save ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_pickle(out_pkl)
    out_pkl_partial.unlink(missing_ok=True)

    elapsed = time.perf_counter() - sweep_t0
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)

    _log("")
    _log("═" * 70)
    _log(f"Sweep complete in {h:02d}h{m:02d}m{s:02d}s.  {len(df)} rows → {out_pkl}")
    _log("")
    print(
        df[["regime", "workload_level", "method", "param_value",
            "max_abs_error", "wall_hot_s"]]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()













