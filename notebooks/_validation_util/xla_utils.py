"""
xla_utils.py — JAX XLA cost-analysis and memory-analysis helpers.

Public API
----------
compile_and_profile(fn, *args, static_argnums=(), static_argnames=(), **kwargs)
    -> dict
    Lower and compile ``fn`` on concrete *or abstract* inputs, run
    ``cost_analysis()`` and ``memory_analysis()``, and return a flat dict of
    scalar fields whose names match RESULT_SCHEMA in run_timings.py.

    When all array inputs are ``jax.ShapeDtypeStruct`` objects no real arrays
    are allocated and no device execution occurs — only tracing, HLO emission,
    XLA compilation, and the cost/memory queries run.  This makes large-N or
    large-J sweeps cheap: a ShapeDtypeStruct for shape (2, 65536, 2) float64
    costs the same to lower/compile as one for shape (2, 64, 2).

flatten_cost(cost_dict) -> dict[str, float]
    Normalise the raw ``cost_analysis()`` dict to a clean flat mapping.

flatten_memory(mem_stats) -> dict[str, int | None]
    Extract scalar integer fields from a ``CompiledMemoryStats`` object.

_safe_json(obj) -> str
    Best-effort JSON serialisation; replaces un-serialisable leaves with repr.

Background — confirmed JAX 0.10 / CPU backend
----------------------------------------------
    compiled.cost_analysis() -> dict
        'flops'              : float   total FLOPs
        'transcendentals'    : float   transcendental-op count (absent when 0)
        'bytes accessed'     : float   total bytes read + written
        'bytes accessed0{}'  : float   bytes read from inputs (per-operand)
        'bytes accessed1{}'  : float   bytes written incl. temps
        'bytes accessedout{}': float   bytes written to output buffers
        'utilization0/1{}'   : float   cache-utilisation hints

    compiled.memory_analysis() -> CompiledMemoryStats
        argument_size_in_bytes, output_size_in_bytes, alias_size_in_bytes,
        temp_size_in_bytes, generated_code_size_in_bytes, peak_memory_in_bytes,
        host_{argument,output,alias,temp,generated_code}_size_in_bytes,
        serialized_buffer_assignment_proto  (bytes — excluded from JSON)

    lowered.as_text() -> str   StableHLO / MLIR text of the computation
"""

import json
import time
from typing import Any

import jax
import numpy as np

def _any_abstract(*args: Any, **kwargs: Any) -> bool:
    """Return True if any positional or keyword argument is a ShapeDtypeStruct."""
    for a in args:
        if isinstance(a, jax.ShapeDtypeStruct):
            return True
    for v in kwargs.values():
        if isinstance(v, jax.ShapeDtypeStruct):
            return True
    return False


# ---------------------------------------------------------------------------
# Utility: flatten the cost_analysis dict
# ---------------------------------------------------------------------------

#: Keys with per-operand decorations stripped by flatten_cost.
_COST_OPERAND_SUFFIXES = ("0{}", "1{}", "out{}")


def flatten_cost(cost: dict) -> dict[str, float]:
    """Normalise a raw ``cost_analysis()`` dict into clean scalar floats.

    Top-level plain keys (``'flops'``, ``'transcendentals'``,
    ``'bytes accessed'``) are kept verbatim after replacing spaces with
    underscores.  Per-operand decorated keys (ending in ``0{}``, ``1{}``,
    ``out{}``) are included with stripped decorators so callers can
    inspect them if needed.

    Returns
    -------
    dict mapping clean str keys to float values.
    """
    if not isinstance(cost, dict):
        return {}

    out: dict[str, float] = {}
    for k, v in cost.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        clean = k.replace(" ", "_").replace("{}", "").rstrip("_")
        out[clean] = fv

    return out


# ---------------------------------------------------------------------------
# Utility: extract scalar fields from CompiledMemoryStats
# ---------------------------------------------------------------------------

_MEMORY_INT_ATTRS: tuple[str, ...] = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "generated_code_size_in_bytes",
    "peak_memory_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
    "host_generated_code_size_in_bytes",
)


def flatten_memory(mem_stats: Any) -> dict[str, "int | None"]:
    """Extract scalar integer fields from a ``CompiledMemoryStats`` object.

    Unknown attributes are silently skipped so the function is safe against
    backend variation across JAX versions.
    """
    out: dict[str, "int | None"] = {}
    for attr in _MEMORY_INT_ATTRS:
        val = getattr(mem_stats, attr, None)
        out[attr] = int(val) if (val is not None and isinstance(val, (int, float))) else None
    return out


# ---------------------------------------------------------------------------
# Utility: safe JSON serialisation
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    """Best-effort JSON dump; non-serialisable leaves become repr() strings."""
    def _default(o: Any) -> Any:
        if isinstance(o, bytes):
            return f"<bytes len={len(o)}>"
        if hasattr(o, "_asdict"):
            return o._asdict()
        if hasattr(o, "__dict__"):
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        return repr(o)

    try:
        return json.dumps(obj, default=_default, indent=None)
    except Exception:
        return repr(obj)


def _memory_stats_to_serialisable(mem_stats: Any) -> dict:
    """Convert a CompiledMemoryStats object to a JSON-friendly plain dict."""
    d = {}
    for attr in _MEMORY_INT_ATTRS:
        val = getattr(mem_stats, attr, None)
        if val is not None:
            d[attr] = int(val)
    return d


# ---------------------------------------------------------------------------
# Core helper: compile_and_profile
# ---------------------------------------------------------------------------

def compile_and_profile(
    fn,
    *args,
    static_argnums: tuple[int, ...] = (),
    static_argnames: "tuple[str, ...] | frozenset[str]" = (),
    **kwargs,
) -> dict:
    """Lower, compile, and XLA-profile ``fn(*args, **kwargs)``.

    Accepts **concrete arrays or** ``jax.ShapeDtypeStruct`` **abstract specs**
    interchangeably.  Using abstract specs is strongly preferred for
    FLOP/memory-cost sweeps: no device memory is allocated and no kernel is
    executed — only tracing, HLO emission, and XLA compilation occur, so
    sweeping J from 64 to 65 536 has negligible overhead.

    Parameters
    ----------
    fn : callable
        Plain (non-jitted) Python function.
    *args, **kwargs :
        Concrete arrays *or* ``jax.ShapeDtypeStruct`` objects for traced
        (non-static) inputs.  Plain Python scalars (``int``, ``float``) are
        always concrete and are traced as such.
    static_argnums : tuple[int, ...]
        Positional argument indices marked static for ``jax.jit``.
    static_argnames : tuple[str, ...] | frozenset[str]
        Keyword argument names marked static for ``jax.jit``
        (e.g. ``static_argnames=("axis",)`` for ``sss.vsig(..., axis=-2)``).

    Returns
    -------
    dict with the following keys (all ``None`` when the backend does not
    support the query or a step fails):

    Metadata
        ``inputs_are_abstract``  True when at least one arg is ShapeDtypeStruct

    Timing
        ``xla_lower_time_s``     wall seconds for ``.lower()``
        ``xla_compile_time_s``   wall seconds for ``.compile()``

    XLA cost-model  (from ``compiled.cost_analysis()``)
        ``xla_flops``            total FLOPs
        ``xla_bytes_accessed``   total bytes read + written
        ``xla_transcendentals``  transcendental-op count

    XLA memory  (from ``compiled.memory_analysis()``)
        ``xla_argument_bytes``       input buffers
        ``xla_output_bytes``         output buffers
        ``xla_temp_bytes``           temporaries / scratch
        ``xla_alias_bytes``          aliased buffers
        ``xla_peak_memory_bytes``    peak HBM across the computation
        ``xla_generated_code_bytes`` compiled kernel binary size

    Raw serialised metadata
        ``cost_analysis_raw``    JSON of raw ``cost_analysis()`` output
        ``memory_analysis_raw``  JSON of raw ``memory_analysis()`` fields

    Diagnostic  (present only on failure)
        ``_lower_error``, ``_compile_error``, ``_cost_error``, ``_memory_error``

    Usage — abstract-input sweep
    ----------------------------
    >>> spec = jax.ShapeDtypeStruct((n_paths, J, d), jnp.float64)
    >>> row = compile_and_profile(
    ...     lambda x: sss.vsig(x, dt=dt, axis=-2),
    ...     spec,                          # abstract: no allocation
    ...     static_argnames=("axis",),
    ... )
    >>> row["xla_flops"], row["xla_peak_memory_bytes"]
    """
    result: dict[str, Any] = {
        "inputs_are_abstract":    _any_abstract(*args, **kwargs),
        "xla_lower_time_s":       None,
        "xla_compile_time_s":     None,
        "xla_flops":              None,
        "xla_bytes_accessed":     None,
        "xla_transcendentals":    None,
        "xla_argument_bytes":     None,
        "xla_output_bytes":       None,
        "xla_temp_bytes":         None,
        "xla_alias_bytes":        None,
        "xla_peak_memory_bytes":  None,
        "xla_generated_code_bytes": None,
        "cost_analysis_raw":      None,
        "memory_analysis_raw":    None,
    }

    # ── 1. Lower ─────────────────────────────────────────────────────────
    jitted = jax.jit(
        fn,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )
    t0 = time.perf_counter()
    try:
        lowered = jitted.lower(*args, **kwargs)
    except Exception as exc:
        result["_lower_error"] = str(exc)
        return result
    result["xla_lower_time_s"] = float(time.perf_counter() - t0)

    # ── 2. Compile ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        compiled = lowered.compile()
    except Exception as exc:
        result["_compile_error"] = str(exc)
        return result
    result["xla_compile_time_s"] = float(time.perf_counter() - t0)

    # ── 3. Cost analysis ─────────────────────────────────────────────────
    try:
        raw_cost = compiled.cost_analysis()
        result["cost_analysis_raw"] = _safe_json(raw_cost)

        flat = flatten_cost(raw_cost)
        result["xla_flops"]           = flat.get("flops")
        result["xla_bytes_accessed"]  = flat.get("bytes_accessed")
        result["xla_transcendentals"] = flat.get("transcendentals")
    except Exception as exc:
        result["_cost_error"] = str(exc)

    # ── 4. Memory analysis ───────────────────────────────────────────────
    try:
        raw_mem = compiled.memory_analysis()
        result["memory_analysis_raw"] = _safe_json(
            _memory_stats_to_serialisable(raw_mem)
        )

        flat_mem = flatten_memory(raw_mem)
        result["xla_argument_bytes"]       = flat_mem.get("argument_size_in_bytes")
        result["xla_output_bytes"]         = flat_mem.get("output_size_in_bytes")
        result["xla_temp_bytes"]           = flat_mem.get("temp_size_in_bytes")
        result["xla_alias_bytes"]          = flat_mem.get("alias_size_in_bytes")
        result["xla_peak_memory_bytes"]    = flat_mem.get("peak_memory_in_bytes")
        result["xla_generated_code_bytes"] = flat_mem.get("generated_code_size_in_bytes")
    except Exception as exc:
        result["_memory_error"] = str(exc)

    return result

