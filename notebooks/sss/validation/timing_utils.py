"""
timing_utils.py — Wall-clock, process-CPU, and OS-resource timing helpers.

Every timed call blocks on JAX asynchronous dispatch before the clocks stop,
so timings reflect true end-to-end latency including device-side execution.

Public API
----------
block_until_ready(output)
    Traverse any pytree and call .block_until_ready() on every JAX array leaf.
    Returns the output unchanged so it can be used inline.

TimingRecord  (dataclass)
    One function call's worth of measurements across all three timer sources.
    Fields:
        wall_s       float   — perf_counter delta (seconds)
        cpu_ns       int     — process_time_ns delta (nanoseconds, cross-platform)
        ru_utime_s   float   — getrusage user CPU time (None on non-Unix)
        ru_stime_s   float   — getrusage system CPU time
        ru_nvcsw     int     — voluntary context switches
        ru_nivcsw    int     — involuntary context switches
        ru_minflt    int     — minor (soft) page faults
        ru_majflt    int     — major (hard) page faults
    Derived property:
        cpu_s        float   — cpu_ns / 1e9

time_call(fn, *args, **kwargs) -> tuple[result, TimingRecord]
    Time one call; block_until_ready on the result before stopping clocks.

time_warmup(fn, *args, n_warmup=1, **kwargs) -> result
    Run fn n_warmup times with block_until_ready, return last result.
    Discards timing records — use this before time_hot_calls().

time_hot_calls(fn, *args, n_calls: int, **kwargs) -> list[TimingRecord]
    Time n_calls back-to-back calls; return list of TimingRecords.

aggregate_timing(records) -> dict
    Compute summary statistics over a list[TimingRecord].
    Returned keys match hot-call fields in RESULT_SCHEMA (run_timings.py):
        wall_hot_{median,mean,std,min,max}_s
        cpu_hot_{median,mean,std}_s
        ru_hot_utime_median_s, ru_hot_stime_median_s
        ru_hot_{nvcsw,nivcsw,minflt,majflt}_total

Notes
-----
- block_until_ready uses `hasattr` so it is safe for non-JAX outputs
  (plain Python objects, StateSpaceSignature instances, etc.).
- resource.getrusage is Unix-only; on Windows all ru_* fields are None.
- process_time_ns() measures the sum of user+kernel CPU time consumed by
  the Python process; on macOS this includes time on all threads. It differs
  from wall time when JAX dispatches work to background threads.
"""

import dataclasses
import time
from typing import Any

import numpy as np
import jax

try:
    import resource as _resource
    _HAS_RESOURCE = True
except ImportError:                      # Windows
    _HAS_RESOURCE = False


# ---------------------------------------------------------------------------
# block_until_ready
# ---------------------------------------------------------------------------

def block_until_ready(output: Any) -> Any:
    """Call .block_until_ready() on every JAX array leaf in *output*.

    Works with:
    - Single JAX arrays
    - Tuples / lists of arrays (e.g. DenseElem signatures)
    - Registered JAX pytrees
    - Plain Python objects (no-op — ``hasattr`` guard prevents AttributeError)

    Returns *output* unchanged so the function can be used inline::

        result = block_until_ready(sss.vsig(X, dt=dt, axis=-2))
    """
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        output,
    )
    return output


# ---------------------------------------------------------------------------
# TimingRecord
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TimingRecord:
    """Timing measurements for one function call across three timer sources.

    All fields are set at construction time from deltas captured immediately
    before and after the call (with block_until_ready in between).

    ``ru_*`` fields are ``None`` on platforms without ``resource.getrusage``
    (i.e. Windows).
    """

    # ── wall clock ──────────────────────────────────────────────────────
    wall_s: float
    """Wall time in seconds (time.perf_counter delta)."""

    # ── process CPU ──────────────────────────────────────────────────────
    cpu_ns: int
    """Process CPU time in nanoseconds (time.process_time_ns delta).
    Counts user + kernel time on all threads of this process."""

    # ── getrusage ─────────────────────────────────────────────────────────
    ru_utime_s: "float | None" = None
    """User CPU time in seconds (getrusage ru_utime delta)."""

    ru_stime_s: "float | None" = None
    """System (kernel) CPU time in seconds (getrusage ru_stime delta)."""

    ru_nvcsw: "int | None" = None
    """Voluntary context switches (getrusage ru_nvcsw delta)."""

    ru_nivcsw: "int | None" = None
    """Involuntary context switches (getrusage ru_nivcsw delta)."""

    ru_minflt: "int | None" = None
    """Minor (soft) page faults — page reclaimed without disk I/O
    (getrusage ru_minflt delta)."""

    ru_majflt: "int | None" = None
    """Major (hard) page faults — page required disk I/O
    (getrusage ru_majflt delta)."""

    @property
    def cpu_s(self) -> float:
        """Process CPU time in seconds (cpu_ns / 1e9)."""
        return self.cpu_ns / 1_000_000_000.0


# ---------------------------------------------------------------------------
# Internal: getrusage snapshot helpers
# ---------------------------------------------------------------------------

def _ru_snapshot():
    """Return a getrusage snapshot or None if unavailable."""
    if _HAS_RESOURCE:
        return _resource.getrusage(_resource.RUSAGE_SELF)
    return None


def _ru_delta(before, after) -> dict:
    """Compute per-field deltas between two getrusage snapshots."""
    keys = ("ru_utime_s", "ru_stime_s", "ru_nvcsw", "ru_nivcsw", "ru_minflt", "ru_majflt")
    if before is None or after is None:
        return dict.fromkeys(keys, None)
    return {
        "ru_utime_s": float(after.ru_utime  - before.ru_utime),
        "ru_stime_s": float(after.ru_stime  - before.ru_stime),
        "ru_nvcsw":   int(after.ru_nvcsw    - before.ru_nvcsw),
        "ru_nivcsw":  int(after.ru_nivcsw   - before.ru_nivcsw),
        "ru_minflt":  int(after.ru_minflt   - before.ru_minflt),
        "ru_majflt":  int(after.ru_majflt   - before.ru_majflt),
    }


# ---------------------------------------------------------------------------
# time_call
# ---------------------------------------------------------------------------

def time_call(fn, *args, **kwargs) -> tuple[Any, TimingRecord]:
    """Time a single call to ``fn(*args, **kwargs)``.

    Clocks start immediately before the Python call and stop only after
    ``block_until_ready`` on the result, ensuring JAX async dispatch is
    included.

    Returns
    -------
    (result, record) : tuple
        *result* is the unchanged return value of ``fn``; *record* is a
        :class:`TimingRecord` with wall, CPU-process, and getrusage timing.
    """
    # ── snapshot pre-call ────────────────────────────────────────────────
    ru_before     = _ru_snapshot()
    cpu_ns_before = time.process_time_ns()
    wall_before   = time.perf_counter()

    # ── execute ──────────────────────────────────────────────────────────
    result = fn(*args, **kwargs)
    block_until_ready(result)

    # ── snapshot post-call ───────────────────────────────────────────────
    wall_after   = time.perf_counter()
    cpu_ns_after = time.process_time_ns()
    ru_after     = _ru_snapshot()

    record = TimingRecord(
        wall_s=float(wall_after   - wall_before),
        cpu_ns=int(cpu_ns_after   - cpu_ns_before),
        **_ru_delta(ru_before, ru_after),
    )
    return result, record


# ---------------------------------------------------------------------------
# time_warmup
# ---------------------------------------------------------------------------

def time_warmup(fn, *args, n_warmup: int = 1, **kwargs) -> Any:
    """Run *fn* ``n_warmup`` times, blocking on each result; return last result.

    Timing records are discarded.  Call this between the JIT-compilation
    first call and the hot-timing loop to evict cold-cache effects.

    Example
    -------
    >>> _, first_record = time_call(vsig_fn, X)     # first call: JIT compile
    >>> time_warmup(vsig_fn, X)                      # evict cold-cache effects
    >>> hot_records = time_hot_calls(vsig_fn, X, n_calls=5)
    """
    result = None
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        block_until_ready(result)
    return result


# ---------------------------------------------------------------------------
# time_hot_calls
# ---------------------------------------------------------------------------

def time_hot_calls(fn, *args, n_calls: int, **kwargs) -> list[TimingRecord]:
    """Time ``n_calls`` back-to-back calls to ``fn(*args, **kwargs)``.

    Each call is individually timed with :func:`time_call` (wall +
    process-CPU + getrusage).  The returned list has exactly ``n_calls``
    entries.

    Parameters
    ----------
    fn : callable
    *args, **kwargs : forwarded to fn
    n_calls : int   Number of timed calls.

    Returns
    -------
    list[TimingRecord]
    """
    records: list[TimingRecord] = []
    for _ in range(n_calls):
        _, record = time_call(fn, *args, **kwargs)
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# aggregate_timing
# ---------------------------------------------------------------------------

def aggregate_timing(records: list[TimingRecord]) -> dict:
    """Compute summary statistics over a list of :class:`TimingRecord` objects.

    Returned keys align with the hot-call fields in ``RESULT_SCHEMA`` in
    ``run_timings.py`` so the dict can be splat-merged directly into a result
    row.

    Wall-clock and CPU statistics
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``wall_hot_median_s``, ``wall_hot_mean_s``, ``wall_hot_std_s``,
    ``wall_hot_iqr_s``, ``wall_hot_min_s``, ``wall_hot_max_s``

    ``cpu_hot_median_s``, ``cpu_hot_mean_s``, ``cpu_hot_std_s``,
    ``cpu_hot_iqr_s``

    getrusage statistics (None when resource is unavailable)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``ru_hot_utime_median_s``, ``ru_hot_utime_mean_s``, ``ru_hot_utime_std_s``,
    ``ru_hot_utime_iqr_s``,
    ``ru_hot_stime_median_s``, ``ru_hot_stime_mean_s``, ``ru_hot_stime_std_s``,
    ``ru_hot_stime_iqr_s``
    — median/mean/std/IQR of user/system CPU times across hot calls.

    ``ru_hot_nvcsw_total``, ``ru_hot_nivcsw_total``
    — total voluntary / involuntary context switches across all hot calls.

    ``ru_hot_minflt_total``, ``ru_hot_majflt_total``
    — total minor / major page faults across all hot calls.

    Parameters
    ----------
    records : list[TimingRecord]   Non-empty list of hot-call records.

    Returns
    -------
    dict
    """
    if not records:
        raise ValueError("records must be non-empty.")

    walls = np.asarray([r.wall_s       for r in records], dtype=np.float64)
    cpus  = np.asarray([r.cpu_ns / 1e9 for r in records], dtype=np.float64)

    out: dict = {
        # ── wall ──────────────────────────────────────────────────────────
        "wall_hot_median_s": float(np.median(walls)),
        "wall_hot_mean_s":   float(np.mean(walls)),
        "wall_hot_std_s":    float(np.std(walls)),
        "wall_hot_iqr_s":    float(np.percentile(walls, 75) - np.percentile(walls, 25)),
        "wall_hot_min_s":    float(np.min(walls)),
        "wall_hot_max_s":    float(np.max(walls)),
        # ── process CPU ───────────────────────────────────────────────────
        "cpu_hot_median_s":  float(np.median(cpus)),
        "cpu_hot_mean_s":    float(np.mean(cpus)),
        "cpu_hot_std_s":     float(np.std(cpus)),
        "cpu_hot_iqr_s":     float(np.percentile(cpus, 75) - np.percentile(cpus, 25)),
    }

    # ── getrusage ─────────────────────────────────────────────────────────
    if records[0].ru_utime_s is not None:
        utimes = np.asarray([r.ru_utime_s for r in records], dtype=np.float64)
        stimes = np.asarray([r.ru_stime_s for r in records], dtype=np.float64)
        out.update(
            ru_hot_utime_median_s=float(np.median(utimes)),
            ru_hot_utime_mean_s=float(np.mean(utimes)),
            ru_hot_utime_std_s=float(np.std(utimes)),
            ru_hot_utime_iqr_s=float(np.percentile(utimes, 75) - np.percentile(utimes, 25)),
            ru_hot_stime_median_s=float(np.median(stimes)),
            ru_hot_stime_mean_s=float(np.mean(stimes)),
            ru_hot_stime_std_s=float(np.std(stimes)),
            ru_hot_stime_iqr_s=float(np.percentile(stimes, 75) - np.percentile(stimes, 25)),
            ru_hot_nvcsw_total=int(sum(r.ru_nvcsw  for r in records)),
            ru_hot_nivcsw_total=int(sum(r.ru_nivcsw for r in records)),
            ru_hot_minflt_total=int(sum(r.ru_minflt  for r in records)),
            ru_hot_majflt_total=int(sum(r.ru_majflt  for r in records)),
        )
    else:
        out.update(
            ru_hot_utime_median_s=None,
            ru_hot_utime_mean_s=None,
            ru_hot_utime_std_s=None,
            ru_hot_utime_iqr_s=None,
            ru_hot_stime_median_s=None,
            ru_hot_stime_mean_s=None,
            ru_hot_stime_std_s=None,
            ru_hot_stime_iqr_s=None,
            ru_hot_nvcsw_total=None,
            ru_hot_nivcsw_total=None,
            ru_hot_minflt_total=None,
            ru_hot_majflt_total=None,
        )

    return out


