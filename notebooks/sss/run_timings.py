"""
run_timings.py — FSS-kernel exact scaling benchmark
====================================================

Timing uses MEDIAN over n_repeats hot calls, which is robust to scheduling noise.

Usage
-----
    python run_timings.py                  # defaults to HOME regime
    python run_timings.py --regime HOME
    python run_timings.py --regime SERVER
    python run_timings.py --regime HOME --repeats 10
    python run_timings.py --dry-run        # print design, exit without timing
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensordev.sss import StateSpaceSignature

# ---------------------------------------------------------------------------
# 2. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSS-kernel exact scaling benchmark")
    p.add_argument(
        "--regime",
        choices=["HOME", "SERVER"],
        default="HOME",
        help="Regime key controlling grid sizes (default: HOME)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override n_repeats from the regime config",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "validation_outputs",
        help="Directory for .pkl / .csv output",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print design and estimated counts, then exit without running",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# 4. Regime configs
# ---------------------------------------------------------------------------

REGIMES = {
    "HOME": dict(
        n_repeats=2,
        # q > 1
        qgt1_cube_q=[2, 3, 4, 5],
        qgt1_cube_J=[64, 128, 256],
        qgt1_cube_R=[1, 2, 3],
        qgt1_cube_N=[4, 5, 6],
        qgt1_ext_q=[2, 3, 4, 5, 6, 7],
        qgt1_ext_J=[384, 512, 768, 1024],
        qgt1_ext_R=[5, 6, 8, 10],
        qgt1_ext_N=[7, 8, 9],
        # q = 1
        q1_cube_J=[64, 128, 256],
        q1_cube_R=[1, 2, 3],
        q1_cube_N=[4, 5, 6, 7],
        q1_ext_J=[384, 512, 768, 1024],
        q1_ext_R=[5, 6, 8, 10],
        q1_ext_N=[8, 9, 10],
        # shared midpoints for axial sweeps
        J_mid=128,
        R_mid=2,
        N_mid=5,
    ),
    "SERVER": dict(
    n_repeats=3,

    # q > 1
    qgt1_cube_q=[2, 3, 4, 5, 6, 7],
    qgt1_cube_J=[64, 128, 256, 384],
    qgt1_cube_R=[1, 2, 3, 4],
    qgt1_cube_N=[4, 5, 6, 7, 8, 9],

    qgt1_ext_q=[2, 3, 4, 5, 6, 7],
    qgt1_ext_J=[512, 768, 1024, 1536, 2048],
    qgt1_ext_R=[5, 6, 8, 10],
    qgt1_ext_N=[10, 12, 14, 16],

    # q = 1
    q1_cube_J=[64, 128, 256, 384],
    q1_cube_R=[1, 2, 3, 4],
    q1_cube_N=[4, 5, 6, 7, 8, 9, 10],

    q1_ext_J=[512, 768, 1024, 1536, 2048],
    q1_ext_R=[5, 6, 8, 10],
    q1_ext_N=[11, 12, 14, 16, 18],

    J_mid=128,
    R_mid=2,
    N_mid=5,
)
}

# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------


def brownian_paths(*, n_paths: int, J: int, d: int, seed: int):
    key = jax.random.PRNGKey(seed)
    dt = 1.0 / (J - 1)
    dW = jnp.sqrt(dt) * jax.random.normal(key, shape=(n_paths, J - 1, d), dtype=jnp.float64)
    X = jnp.concatenate(
        [jnp.zeros((n_paths, 1, d), dtype=jnp.float64), jnp.cumsum(dW, axis=1)],
        axis=1,
    )
    return X, dt


def random_fssk_params(
    *, q: int, R: int, m: int, d: int, seed: int,
    eig_min: float = 0.1, eig_max: float = 1.5, jordan_alpha: float = 0.25,
    dtype=np.float64,
):
    rng = np.random.default_rng(seed)
    eigs = rng.uniform(eig_min, eig_max, size=R)
    Jmat = np.diag(eigs)
    for i in range(R - 1):
        Jmat[i, i + 1] = jordan_alpha
    G = rng.normal(size=(R, R))
    Q, _ = np.linalg.qr(G)
    Lambda = Q @ Jmat @ Q.T
    A = np.empty((q, m, d), dtype=dtype)
    for p in range(q):
        G = rng.normal(size=(d, m) if m <= d else (m, d))
        Qp, _ = np.linalg.qr(G)
        A[p] = Qp.T if m <= d else Qp
    b = rng.normal(size=(q, R))
    b /= np.sum(np.abs(b), axis=0, keepdims=True)
    return (
        jnp.asarray(Lambda.astype(dtype)),
        jnp.asarray(A.astype(dtype)),
        jnp.asarray(b.astype(dtype)),
    )


def block_until_ready_tree(tree):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), tree)


def time_exact_components(*, X, dt, Lambda, A, b, N, n_repeats):
    """
    Returns (construct_time, first_call_time, hot_median, hot_mean, hot_std).

    - construct_time  : StateSpaceSignature.from_matrix(...)
    - first_call_time : first vsig call including JAX compilation
    - hot_median      : median of n_repeats hot calls  ← primary statistic;
                        robust to multi-thread scheduling noise
    - hot_mean/std    : also recorded for completeness
    """
    t0 = time.perf_counter()
    sss = StateSpaceSignature.from_matrix(Lambda=Lambda, A=A, b=b, trunc=N)
    construct_time = time.perf_counter() - t0

    def call():
        return sss.vsig(X, dt=dt, axis=-2)

    # First call: includes JAX JIT compilation
    t0 = time.perf_counter()
    block_until_ready_tree(call())
    first_call_time = time.perf_counter() - t0

    # One additional warmup (evicts cold-cache effects) — not recorded
    block_until_ready_tree(call())

    hot = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        block_until_ready_tree(call())
        hot.append(time.perf_counter() - t0)

    return (
        float(construct_time),
        float(first_call_time),
        float(np.median(hot)),
        float(np.mean(hot)),
        float(np.std(hot)),
    )

# ---------------------------------------------------------------------------
# 6. Design matrix
# ---------------------------------------------------------------------------

def build_design(cfg: dict, m: int, d: int, n_paths: int) -> pd.DataFrame:
    configs = []

    def add(family, design, q, J, R, N):
        configs.append(dict(family=family, design=design, q=q, J=J, R=R, N=N,
                            m=m, d=d, n_paths=n_paths))

    # q > 1 cube
    for q, J, R, N in itertools.product(
            cfg["qgt1_cube_q"], cfg["qgt1_cube_J"],
            cfg["qgt1_cube_R"], cfg["qgt1_cube_N"]):
        add("qgt1", "cube", q, J, R, N)

    # q > 1 axial extensions
    for q, J in itertools.product(cfg["qgt1_ext_q"], cfg["qgt1_ext_J"]):
        add("qgt1", "J_ext", q, J, cfg["R_mid"], cfg["N_mid"])
    for q, R in itertools.product(cfg["qgt1_ext_q"], cfg["qgt1_ext_R"]):
        add("qgt1", "R_ext", q, cfg["J_mid"], R, cfg["N_mid"])
    for q, N in itertools.product(cfg["qgt1_ext_q"], cfg["qgt1_ext_N"]):
        add("qgt1", "N_ext", q, cfg["J_mid"], cfg["R_mid"], N)

    # q = 1 cube
    for J, R, N in itertools.product(
            cfg["q1_cube_J"], cfg["q1_cube_R"], cfg["q1_cube_N"]):
        add("q1", "cube", 1, J, R, N)

    # q = 1 axial extensions
    for J in cfg["q1_ext_J"]:
        add("q1", "J_ext", 1, J, cfg["R_mid"], cfg["N_mid"])
    for R in cfg["q1_ext_R"]:
        add("q1", "R_ext", 1, cfg["J_mid"], R, cfg["N_mid"])
    for N in cfg["q1_ext_N"]:
        add("q1", "N_ext", 1, cfg["J_mid"], cfg["R_mid"], N)

    df = (
        pd.DataFrame(configs)
        .drop_duplicates(["family", "q", "J", "R", "N", "m", "d", "n_paths"], keep="first")
        .reset_index(drop=True)
    )
    return df

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = REGIMES[args.regime]
    n_repeats = args.repeats if args.repeats is not None else cfg["n_repeats"]

    m, d, n_paths = 2, 2, 2
    seed0 = 20260428

    df_design = build_design(cfg, m=m, d=d, n_paths=n_paths)

    print(f"Regime          : {args.regime}")
    print(f"n_repeats       : {n_repeats}")
    print(f"Total runs      : {len(df_design)}")
    print()
    print(df_design.groupby(["family", "design"]).size().rename("n_runs").to_string())
    print()

    if args.dry_run:
        print("Dry-run mode — exiting without running timings.")
        return


    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime_tag = args.regime.lower()
    out_pkl = args.output_dir / f"fssk_exact_scaling_timings_{regime_tag}.pkl"
    out_csv = args.output_dir / f"fssk_exact_scaling_timings_{regime_tag}.csv"

    rows = []

    for run_id, row in df_design.iterrows():
        cfg_row = row.to_dict()
        q, J, R, N = int(cfg_row["q"]), int(cfg_row["J"]), int(cfg_row["R"]), int(cfg_row["N"])

        print(
            f"[{run_id + 1:03d}/{len(df_design)}]  "
            f"{cfg_row['family']:>4} | {cfg_row['design']:>5} | "
            f"q={q}  J={J}  R={R}  N={N}",
            flush=True,
        )

        X, dt = brownian_paths(n_paths=n_paths, J=J, d=d, seed=seed0 + 10_000 + run_id)
        Lambda, A, b = random_fssk_params(q=q, R=R, m=m, d=d, seed=seed0 + run_id)

        construct_t, first_t, hot_median, hot_mean, hot_std = time_exact_components(
            X=X, dt=dt, Lambda=Lambda, A=A, b=b, N=N, n_repeats=n_repeats,
        )

        intervals = J - 1
        q1_proxy   = intervals * (R ** 2) * (m ** N)
        qgt1_proxy = q1_proxy * N

        rows.append(dict(
            run_id=run_id, regime=args.regime,
            family=cfg_row["family"], design=cfg_row["design"],
            q=q, J=J, intervals=intervals, R=R, m=m, d=d, N=N,
            n_paths=n_paths, n_repeats=n_repeats,
            construct_time=construct_t,
            first_call_time=first_t,
            hot_time_median=hot_median,
            hot_time_mean=hot_mean,
            hot_time_std=hot_std,
            q1_proxy=q1_proxy,
            qgt1_proxy=qgt1_proxy,
        ))

        # checkpoint every 25 runs
        if (run_id + 1) % 25 == 0:
            pd.DataFrame(rows).to_pickle(out_pkl)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  checkpoint → {out_pkl}", flush=True)

    df_timings = pd.DataFrame(rows)
    df_timings.to_pickle(out_pkl)
    df_timings.to_csv(out_csv, index=False)

    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")
    print(df_timings[["family", "design", "q", "J", "R", "N",
                       "hot_time_median", "hot_time_mean", "hot_time_std"]].to_string())


if __name__ == "__main__":
    main()










