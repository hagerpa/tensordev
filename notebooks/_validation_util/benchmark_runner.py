"""
benchmark_runner.py — Generic benchmark loop and shared result schema.

Public API
----------
IDENTITY_FIELDS : OrderedDict
    run_id, regime, method, family, design — common to every benchmark.

TIMING_FIELDS : OrderedDict
    wall_*/cpu_*/ru_* timing measurements produced by timing_utils.

XLA_FIELDS : OrderedDict
    xla_* fields produced by xla_utils.compile_and_profile.

ERROR_FIELDS : OrderedDict
    error_metric_name / error_value / error_reference_tag placeholders.

TIMING_SCHEMA : OrderedDict
    Concatenation of all four groups above.  Validation-set scripts extend
    this with their own hyperparameter and cost-proxy fields to build a
    full RESULT_SCHEMA.

make_empty_row(schema) -> dict
    Return a dict with every schema key initialised to None.

run_benchmark(design_df, row_fn, ...) -> pd.DataFrame
    Iterate a design DataFrame, call row_fn per row, checkpoint
    periodically, and write .pkl / .csv output.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Schema field groups
# ---------------------------------------------------------------------------

IDENTITY_FIELDS: OrderedDict = OrderedDict([
    ("run_id",  ("int", "sequential index within the current run")),
    ("regime",  ("str", "'SMALL', 'MEDIUM' or 'LARGE'")),
    ("method",  ("str", "algorithm tag, e.g. 'exact_vsig'")),
    ("family",  ("str", "experiment-specific grouping label")),
    ("design",  ("str", "'random'")),
])

TIMING_FIELDS: OrderedDict = OrderedDict([
    # ── wall-clock (seconds) ──────────────────────────────────────────────
    ("wall_construct_s",      ("float",      "wall time for object construction")),
    ("wall_first_call_s",     ("float",      "wall time for first call (includes JAX JIT)")),
    ("wall_hot_median_s",     ("float",      "median wall time over hot calls")),
    ("wall_hot_mean_s",       ("float",      "mean wall time over hot calls")),
    ("wall_hot_std_s",        ("float",      "std dev of wall time over hot calls")),
    ("wall_hot_iqr_s",        ("float",      "IQR of wall time over hot calls")),
    ("wall_hot_min_s",        ("float",      "minimum wall time over hot calls")),
    ("wall_hot_max_s",        ("float",      "maximum wall time over hot calls")),
    # ── process CPU (seconds) ─────────────────────────────────────────────
    ("cpu_construct_s",       ("float|None", "process CPU time for construction")),
    ("cpu_first_call_s",      ("float|None", "process CPU time for first call")),
    ("cpu_hot_median_s",      ("float|None", "median process CPU time over hot calls")),
    ("cpu_hot_mean_s",        ("float|None", "mean process CPU time over hot calls")),
    ("cpu_hot_std_s",         ("float|None", "std dev of process CPU time over hot calls")),
    ("cpu_hot_iqr_s",         ("float|None", "IQR of process CPU time over hot calls")),
    # ── OS resource counters (None on Windows) ────────────────────────────
    ("ru_construct_utime_s",  ("float|None", "user CPU time for construction (getrusage)")),
    ("ru_construct_stime_s",  ("float|None", "sys CPU time for construction (getrusage)")),
    ("ru_first_call_utime_s", ("float|None", "user CPU time for first call (getrusage)")),
    ("ru_first_call_stime_s", ("float|None", "sys CPU time for first call (getrusage)")),
    ("ru_hot_utime_median_s", ("float|None", "median user CPU over hot calls (getrusage)")),
    ("ru_hot_utime_mean_s",   ("float|None", "mean user CPU over hot calls (getrusage)")),
    ("ru_hot_utime_std_s",    ("float|None", "std dev user CPU over hot calls (getrusage)")),
    ("ru_hot_utime_iqr_s",    ("float|None", "IQR user CPU over hot calls (getrusage)")),
    ("ru_hot_stime_median_s", ("float|None", "median sys CPU over hot calls (getrusage)")),
    ("ru_hot_stime_mean_s",   ("float|None", "mean sys CPU over hot calls (getrusage)")),
    ("ru_hot_stime_std_s",    ("float|None", "std dev sys CPU over hot calls (getrusage)")),
    ("ru_hot_stime_iqr_s",    ("float|None", "IQR sys CPU over hot calls (getrusage)")),
    ("ru_hot_nvcsw_total",    ("int|None",   "total voluntary context switches over hot calls")),
    ("ru_hot_nivcsw_total",   ("int|None",   "total involuntary context switches over hot calls")),
    ("ru_hot_minflt_total",   ("int|None",   "total minor page faults over hot calls")),
    ("ru_hot_majflt_total",   ("int|None",   "total major page faults over hot calls")),
])

XLA_FIELDS: OrderedDict = OrderedDict([
    # ── XLA timing ───────────────────────────────────────────────────────
    ("xla_lower_time_s",          ("float|None", "wall seconds for .lower() — HLO emission")),
    ("xla_compile_time_s",        ("float|None", "wall seconds for .compile() — XLA compilation")),
    # ── XLA cost-model scalars (compiled.cost_analysis()) ────────────────
    ("xla_flops",                 ("float|None", "total FLOPs")),
    ("xla_bytes_accessed",        ("float|None", "total bytes read + written")),
    ("xla_transcendentals",       ("float|None", "transcendental-op count")),
    # ── XLA memory scalars (compiled.memory_analysis()) ───────────────────
    ("xla_argument_bytes",        ("int|None",   "HBM for input buffers")),
    ("xla_output_bytes",          ("int|None",   "HBM for output buffers")),
    ("xla_temp_bytes",            ("int|None",   "HBM for temporaries / scratch")),
    ("xla_alias_bytes",           ("int|None",   "HBM for aliased buffers")),
    ("xla_peak_memory_bytes",     ("int|None",   "peak HBM across the computation")),
    ("xla_generated_code_bytes",  ("int|None",   "compiled kernel binary size")),
    # ── raw metadata ─────────────────────────────────────────────────────
    ("xla_hlo_path",              ("str|None",   "path to serialised HLO proto")),
    ("xla_profile_path",          ("str|None",   "path to XLA execution profile")),
    ("cost_analysis_raw",         ("str|None",   "JSON of compiled.cost_analysis() output")),
    ("memory_analysis_raw",       ("str|None",   "JSON of compiled.memory_analysis() fields")),
])

ERROR_FIELDS: OrderedDict = OrderedDict([
    ("error_metric_name",   ("str|None",   "name of the error metric, e.g. 'max_abs'")),
    ("error_value",         ("float|None", "scalar error against a reference")),
    ("error_reference_tag", ("str|None",   "tag identifying the reference, e.g. 'euler_dyadic6'")),
])

TIMING_SCHEMA: OrderedDict = OrderedDict([
    *IDENTITY_FIELDS.items(),
    *TIMING_FIELDS.items(),
    *XLA_FIELDS.items(),
    *ERROR_FIELDS.items(),
])


def make_empty_row(schema: OrderedDict) -> dict:
    """Return a dict with every schema key initialised to None."""
    return dict.fromkeys(schema, None)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    design_df: pd.DataFrame,
    row_fn: Callable[[int, dict], dict],
    *,
    output_dir: Path,
    output_stem: str,
    dry_run: bool = False,
    checkpoint_every: int = 25,
    column_order: "list[str] | None" = None,
) -> pd.DataFrame:
    """Iterate a design DataFrame, call row_fn per row, checkpoint, write output.

    Parameters
    ----------
    design_df : pd.DataFrame
        One row per parameter configuration.
    row_fn : Callable[[int, dict], dict]
        ``row_fn(run_id, design_row_dict)`` — benchmark one configuration and
        return a complete result dict.  run_id is the integer DataFrame index.
    output_dir : Path
        Output directory; created if absent.
    output_stem : str
        Base filename without extension, e.g. ``'fssk_timings_medium'``.
    dry_run : bool
        Print the design DataFrame and return an empty DataFrame.
    checkpoint_every : int
        Write intermediate results every N completed rows.
    column_order : list[str] | None
        If given, reorder / select columns in the final DataFrame.
        Columns not present in the DataFrame are silently ignored.

    Returns
    -------
    pd.DataFrame  (empty when dry_run=True)
    """
    if dry_run:
        print(f"Design ({len(design_df)} rows):")
        print(design_df.to_string())
        return pd.DataFrame()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = output_dir / f"{output_stem}.pkl"
    out_csv = output_dir / f"{output_stem}.csv"

    rows: list[dict] = []

    for run_id, design_row in design_df.iterrows():
        row = row_fn(int(run_id), design_row.to_dict())
        rows.append(row)

        if len(rows) % checkpoint_every == 0:
            _write(pd.DataFrame(rows), out_pkl, out_csv)
            print(f"  checkpoint → {out_pkl}", flush=True)

    df = pd.DataFrame(rows)
    if column_order is not None:
        present = [c for c in column_order if c in df.columns]
        df = df[present]
    _write(df, out_pkl, out_csv)
    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")
    return df


def _write(df: pd.DataFrame, pkl: Path, csv: Path) -> None:
    df.to_pickle(pkl)
    df.to_csv(csv, index=False)
