#!/usr/bin/env python3
"""Benchmark wordwise signatures and scalar-FSSK states on one NVIDIA GPU.

This is a synchronized diagnostic, not a pytest performance assertion.  The
scope covers every supported total-degree and bidegree layout in standard and
shear coordinates.

The safe smoke defaults retain the original small float32 ordinary-signature
case.  A broader ordinary validation run can be requested with, for example::

    python benchmarks/wordwise_gpu.py \
        --families total-standard total-shear \
                   bidegree-standard bidegree-shear \
                   bidegree-partial-standard bidegree-partial-shear \
        --dtypes float32 float64 \
        --block-modes terminal accumulated independent \
        --warmups 5 --repeats 30 \
        --output benchmarks/results/wordwise_gpu.json

A bounded scalar-FSSK smoke run is::

    python benchmarks/wordwise_gpu.py \
        --workloads fssk --fssk-outputs state readout \
        --state-dim 3 --coefficient-modes uniform

The existing ``--batch`` option controls both workloads.  For example,
``--workloads fssk --fssk-outputs state --batch 10000`` exercises large-batch
state recursion while retaining the same bounded truncations and resource
checks.

Automatic dispatch is measured as a direct public call: placing that call
inside an outer ``jax.jit`` intentionally selects the portable traced route.
Forced native and forced portable calls are separately lowered and compiled so
their compilation and first-execution costs remain visible.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import platform as platform_module
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jaxlib
import numpy as np

import tensordev as td
from tensordev._wordwise.dispatch import (
    _automatic_wordwise_release_eligible,
    _normalized_compute_capability,
    _supported_cuda_device,
    fssk_q1_wordwise_candidate_eligible,
    ordinary_wordwise_candidate_eligible,
)
from tensordev._wordwise.layout import (
    build_layout_plan,
    clear_layout_plan_cache,
)
from tensordev._wordwise.ordinary import run_ordinary_wordwise
from tensordev._wordwise.pallas_ordinary import clear_pallas_ordinary_cache
from tensordev._wordwise.pallas_quotient import clear_pallas_quotient_cache
from tensordev.development.free import (
    _execute_portable_free_development_call,
    _finalize_free_development_call,
    _prepare_free_development_call,
)
from tensordev.sss import FSSK
from tensordev.sss import state_update as fssk_state_update
from tensordev.sss.state_update import fssk_state_from_coef, fssk_vsig


FAMILIES = (
    "total-standard",
    "total-shear",
    "bidegree-standard",
    "bidegree-shear",
    "bidegree-partial-standard",
    "bidegree-partial-shear",
)
BLOCK_MODES = ("terminal", "accumulated", "independent")
WORKLOADS = ("ordinary", "fssk")
FSSK_OUTPUTS = ("state", "readout")
COEFFICIENT_MODES = ("uniform", "varying")
INITIAL_STATES = ("zero", "nonzero")
_MAX_STATE_DIM = 128
_MAX_CASE_ELEMENTS = 250_000_000
_BOOTSTRAP_SEED = 0x5EED
_BOOTSTRAP_RESAMPLES = 10_000
_MEDIAN_SPEEDUP_GATE = 0.20
_CONFIDENCE_SPEEDUP_GATE = 0.10


@dataclass(frozen=True, slots=True)
class _Runner:
    function: Callable[[Any], Any]
    eager: bool = False


def _package_version() -> str:
    try:
        return importlib.metadata.version("tensordev")
    except importlib.metadata.PackageNotFoundError:
        return "source checkout"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronized ordinary-signature and scalar-FSSK benchmark for "
            "the TensorDev NVIDIA wordwise path."
        )
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=["ordinary"],
        help="workloads to measure (default: ordinary)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=["total-standard"],
        help="core families to measure (default: total-standard)",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float64"),
        default=["float32"],
        help="floating-point dtypes to measure (default: float32)",
    )
    parser.add_argument(
        "--block-modes",
        nargs="+",
        choices=BLOCK_MODES,
        default=["terminal"],
        help="sequence outputs to measure (default: terminal)",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--tile-words", type=int, default=128)
    parser.add_argument("--tile-prime-words", type=int, default=16)
    parser.add_argument(
        "--fssk-outputs",
        nargs="+",
        choices=FSSK_OUTPUTS,
        default=list(FSSK_OUTPUTS),
        help="FSSK results to measure (default: state readout)",
    )
    parser.add_argument(
        "--coefficient-modes",
        nargs="+",
        choices=COEFFICIENT_MODES,
        default=["uniform"],
        help="FSSK coefficient time grids (default: uniform)",
    )
    parser.add_argument(
        "--initial-states",
        nargs="+",
        choices=INITIAL_STATES,
        default=["zero"],
        help="FSSK recursion seeds (default: zero)",
    )
    parser.add_argument("--state-dim", type=int, default=3)
    parser.add_argument("--quad-order", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; results are always printed to stdout",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch", "steps", "block_size", "repeats"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.state_dim <= 0 or args.state_dim > _MAX_STATE_DIM:
        raise SystemExit(
            f"--state-dim must be between 1 and {_MAX_STATE_DIM}"
        )
    if args.quad_order <= 0 or args.quad_order > 256:
        raise SystemExit("--quad-order must be between 1 and 256")
    if args.tile_words <= 0 or args.tile_words & (args.tile_words - 1):
        raise SystemExit("--tile-words must be a positive power of two")
    if (
        args.tile_prime_words <= 0
        or args.tile_prime_words & (args.tile_prime_words - 1)
    ):
        raise SystemExit("--tile-prime-words must be a positive power of two")
    if any(mode != "terminal" for mode in args.block_modes):
        if args.steps % args.block_size:
            raise SystemExit(
                "--block-size must divide --steps for blocked output modes"
            )


def _supported_devices() -> tuple[Any, ...]:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return ()
    return tuple(device for device in devices if _supported_cuda_device(device))


def _select_device(index: int) -> Any:
    devices = _supported_devices()
    if not devices:
        raise SystemExit(
            "No supported NVIDIA JAX device was found. This benchmark requires "
            "CUDA compute capability 8.0 or newer."
        )
    if index < 0 or index >= len(devices):
        raise SystemExit(
            f"--device-index={index} is outside the supported-device range "
            f"0..{len(devices) - 1}"
        )
    return devices[index]


def _case(family: str) -> tuple[Any, int | tuple[int, int]]:
    if family == "total-standard":
        return td.make_core(dims=2, max_trunc=3), 3
    if family == "total-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=3,
                coordinates="shear",
            ),
            3,
        )
    if family == "bidegree-standard":
        return td.make_core(dims=(1, 1), max_trunc=(2, 1)), (2, 1)
    if family == "bidegree-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                coordinates="shear",
            ),
            (2, 1),
        )
    if family == "bidegree-partial-standard":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                partially_symmetrized=True,
            ),
            (2, 1),
        )
    if family == "bidegree-partial-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            (2, 1),
        )
    raise AssertionError(f"unknown family {family!r}")


def _block_options(mode: str, block_size: int) -> tuple[int | None, bool]:
    if mode == "terminal":
        return None, False
    if mode == "accumulated":
        return block_size, True
    if mode == "independent":
        return block_size, False
    raise AssertionError(f"unknown block mode {mode!r}")


def _make_input(
    *,
    batch: int,
    steps: int,
    dtype: np.dtype,
    seed: int,
    device: Any,
) -> jax.Array:
    generator = np.random.default_rng(seed)
    increments = generator.normal(
        loc=0.0,
        scale=0.03,
        size=(batch, steps, 2),
    ).astype(dtype)
    return jax.device_put(increments, device)


def _prepare(
    increments: jax.Array,
    *,
    core: Any,
    truncation: int | tuple[int, int],
    block_size: int | None,
    accumulate: bool,
):
    return _prepare_free_development_call(
        (increments,),
        trunc=truncation,
        increment_input=True,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        core=core,
    )


def _ordinary_runners(
    *,
    core: Any,
    truncation: int | tuple[int, int],
    block_size: int | None,
    accumulate: bool,
    tile_words: int,
    tile_prime_words: int,
) -> dict[str, _Runner]:
    def automatic(increments):
        return td.path_signature(
            increments,
            trunc=truncation,
            increment_input=True,
            axis=-2,
            block_size=block_size,
            accumulate=accumulate,
            core=core,
        )

    def forced_pallas(increments):
        call = _prepare(
            increments,
            core=core,
            truncation=truncation,
            block_size=block_size,
            accumulate=accumulate,
        )
        result = run_ordinary_wordwise(
            call,
            tile_words=tile_words,
            tile_prime_words=tile_prime_words,
        )
        return _finalize_free_development_call(
            call,
            result,
            runner_applied_seed=False,
            runner_emitted_starting_point=False,
        )

    def forced_portable(increments):
        call = _prepare(
            increments,
            core=core,
            truncation=truncation,
            block_size=block_size,
            accumulate=accumulate,
        )
        return _execute_portable_free_development_call(call)

    return {
        "automatic-public-eager": _Runner(automatic, eager=True),
        "forced-pallas-jit": _Runner(forced_pallas),
        "forced-portable-jit": _Runner(forced_portable),
    }


def _make_fssk_kernel(
    *,
    state_dim: int,
    dtype: np.dtype,
    quad_order: int,
    device: Any,
) -> FSSK:
    diagonal = np.linspace(0.2, 0.75, state_dim, dtype=dtype)
    weights = np.linspace(0.8, -0.2, state_dim, dtype=dtype)[None]
    kernel = FSSK.from_matrix(
        Lambda=np.diag(diagonal),
        A=np.eye(2, dtype=dtype)[None],
        b=weights,
        quad_order=quad_order,
    )
    return jax.device_put(kernel, device)


def _make_fssk_time_and_coefficients(
    kernel: FSSK,
    *,
    mode: str,
    steps: int,
    maximum_order: int,
    dtype: np.dtype,
    device: Any,
):
    if mode == "uniform":
        dt = np.asarray(0.07, dtype=dtype)
    elif mode == "varying":
        dt = np.linspace(0.04, 0.1, steps, dtype=dtype)
    else:
        raise AssertionError(f"unknown coefficient mode {mode!r}")
    dt = jax.device_put(dt, device)
    coefficients = kernel.coef(
        dt,
        trunc=maximum_order,
        dtype=dtype,
    )
    return dt, _synchronize(coefficients)


def _make_fssk_initial_state(
    *,
    mode: str,
    core: Any,
    truncation: int | tuple[int, int],
    plan: Any,
    state_dim: int,
    dtype: np.dtype,
    device: Any,
) -> Any:
    if mode == "zero":
        return None
    if mode != "nonzero":
        raise AssertionError(f"unknown initial-state mode {mode!r}")

    blocks = []
    offset = 1
    for block in plan.blocks[1:]:
        size = state_dim * block.width
        values = np.arange(offset, offset + size, dtype=dtype)
        values = (0.001 * values).reshape(
            (1, 1, 1, state_dim, block.width)
        )
        blocks.append(jax.device_put(values, device))
        offset += size
    standard = plan.assemble_first_on(blocks)
    native = core.tensor_from_standard_coordinates(
        standard,
        trunc=truncation,
        first_on=True,
    )
    return _synchronize(native)


def _prepare_fssk(
    projected: jax.Array,
    *,
    coefficients: Any,
    core: Any,
    seq_core: Any,
    truncation: int | tuple[int, int],
    block_size: int | None,
    accumulate: bool,
    initial_state: Any,
):
    return fssk_state_update._prepare_fssk_state_from_coef_call(
        projected,
        coef=coefficients,
        trunc=truncation,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=False,
        core=core,
        seq_core=seq_core,
    )


def _prepare_fssk_path(
    increments: jax.Array,
    *,
    kernel: FSSK,
    dt: Any,
    core: Any,
    seq_core: Any,
    truncation: int | tuple[int, int],
    maximum_order: int,
    block_size: int | None,
    accumulate: bool,
    initial_state: Any,
):
    return fssk_state_update._prepare_fssk_state_call(
        increments,
        kernel=kernel,
        dt=dt,
        trunc=truncation,
        maximum_order=maximum_order,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=False,
        increment_input=True,
        core=core,
        seq_core=seq_core,
    )


def _fssk_runners(
    *,
    output_kind: str,
    coefficients: Any,
    dt: Any,
    kernel: FSSK,
    core: Any,
    seq_core: Any,
    truncation: int | tuple[int, int],
    maximum_order: int,
    block_size: int | None,
    accumulate: bool,
    initial_state: Any,
) -> dict[str, _Runner]:
    state_kwargs = {
        "coef": coefficients,
        "trunc": truncation,
        "axis": -2,
        "block_size": block_size,
        "accumulate": accumulate,
        "initial_state": initial_state,
        "core": core,
        "seq_core": seq_core,
    }

    def automatic(projected):
        return fssk_state_from_coef(projected, **state_kwargs)

    def forced_native(projected):
        call = _prepare_fssk(
            projected,
            coefficients=coefficients,
            core=core,
            seq_core=seq_core,
            truncation=truncation,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial_state,
        )
        states = fssk_state_update._try_fssk_q1_wordwise(call)
        if states is None:
            raise RuntimeError(
                "the private native FSSK runner rejected a supported case"
            )
        return fssk_state_update._finalize_wordwise_prepared_fssk_state(
            call,
            states,
        )

    def forced_portable(projected):
        call = _prepare_fssk(
            projected,
            coefficients=coefficients,
            core=core,
            seq_core=seq_core,
            truncation=truncation,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial_state,
        )
        return fssk_state_update._execute_portable_prepared_fssk_state(call)

    state_runners = {
        "automatic-public-eager": _Runner(automatic, eager=True),
        "forced-private-native-jit": _Runner(forced_native),
        "forced-portable-jit": _Runner(forced_portable),
    }
    if output_kind == "state":
        return state_runners
    if output_kind != "readout":
        raise AssertionError(f"unknown FSSK output kind {output_kind!r}")

    def automatic_readout(increments):
        return fssk_vsig(
            increments,
            kernel=kernel,
            dt=dt,
            trunc=truncation,
            axis=-2,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial_state,
            increment_input=True,
            core=core,
            seq_core=seq_core,
        )

    def forced_native_readout(increments):
        call = _prepare_fssk_path(
            increments,
            kernel=kernel,
            dt=dt,
            core=core,
            seq_core=seq_core,
            truncation=truncation,
            maximum_order=maximum_order,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial_state,
        )
        if block_size is None:
            weights = (
                fssk_state_update._prepare_fssk_q1_wordwise_readout_weights(
                    call,
                    kernel,
                    0.0,
                )
            )
            if weights is None:
                raise RuntimeError(
                    "the fused native FSSK readout rejected terminal output"
                )
            signature = fssk_state_update._try_fssk_q1_wordwise_readout(
                call,
                weights,
            )
            if signature is None:
                raise RuntimeError(
                    "the private native FSSK readout rejected a supported "
                    "case"
                )
            return (
                fssk_state_update._finalize_wordwise_prepared_fssk_readout(
                    call,
                    signature,
                )
            )
        states = fssk_state_update._try_fssk_q1_wordwise(call)
        if states is None:
            raise RuntimeError(
                "the private native FSSK runner rejected a supported case"
            )
        return fssk_state_update._finalize_wordwise_prepared_fssk_vsig(
            call,
            states,
            kernel,
            0.0,
        )

    return {
        "automatic-public-eager": _Runner(automatic_readout, eager=True),
        "forced-private-native-jit": _Runner(forced_native_readout),
        "forced-portable-jit": _Runner(automatic_readout),
    }


def _synchronize(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()
    return value


def _clear_execution_caches() -> None:
    from tensordev._wordwise.pallas_fssk_q1 import (
        clear_pallas_fssk_q1_cache,
    )
    from tensordev._wordwise.pallas_fssk_quotient import (
        clear_pallas_fssk_quotient_cache,
    )

    clear_layout_plan_cache()
    clear_pallas_ordinary_cache()
    clear_pallas_quotient_cache()
    clear_pallas_fssk_q1_cache()
    clear_pallas_fssk_quotient_cache()
    jax.clear_caches()


def _timed_call(function: Callable[[Any], Any], argument: Any):
    started = time.perf_counter()
    value = function(argument)
    _synchronize(value)
    return value, time.perf_counter() - started


def _distribution(samples: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(values.size),
        "median_seconds": float(statistics.median(samples)),
        "p10_seconds": float(np.percentile(values, 10)),
        "p90_seconds": float(np.percentile(values, 90)),
        "minimum_seconds": float(values.min()),
        "maximum_seconds": float(values.max()),
        "samples_seconds": [float(value) for value in samples],
    }


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _device_memory_stats(device: Any) -> dict[str, Any] | None:
    memory_stats = getattr(device, "memory_stats", None)
    if memory_stats is None:
        return None
    try:
        stats = memory_stats()
    except Exception:
        return None
    if stats is None:
        return None
    if not isinstance(stats, Mapping):
        return None
    return {
        str(key): _json_compatible(value) for key, value in stats.items()
    }


def _bootstrap_paired_median_ratio_interval(
    native: np.ndarray,
    portable: np.ndarray,
    *,
    seed: int = _BOOTSTRAP_SEED,
    resamples: int = _BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    if native.ndim != 1 or portable.ndim != 1 or native.size == 0:
        raise ValueError("bootstrap samples must be non-empty vectors")
    if native.shape != portable.shape:
        raise ValueError("paired bootstrap sample counts must match")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    generator = np.random.default_rng(seed)
    ratios = np.empty(resamples, dtype=np.float64)
    chunk_size = 256
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = generator.integers(
            0,
            native.size,
            size=(stop - start, native.size),
        )
        ratios[start:stop] = np.median(
            native[indices],
            axis=1,
        ) / np.median(portable[indices], axis=1)
    lower, upper = np.percentile(ratios, (2.5, 97.5))
    return float(lower), float(upper)


def _paired_native_portable_gate(
    native_samples: Sequence[float],
    portable_samples: Sequence[float],
) -> dict[str, Any]:
    native = np.asarray(native_samples, dtype=np.float64)
    portable = np.asarray(portable_samples, dtype=np.float64)
    if native.ndim != 1 or portable.ndim != 1 or native.size == 0:
        raise ValueError("timing samples must be non-empty vectors")
    if native.shape != portable.shape:
        raise ValueError("native and portable timing sample counts must match")
    if np.any(native <= 0.0) or np.any(portable <= 0.0):
        raise ValueError("timing samples must be positive")

    paired_ratios = native / portable
    ratio_estimate = float(np.median(native) / np.median(portable))
    ratio_lower, ratio_upper = _bootstrap_paired_median_ratio_interval(
        native,
        portable,
    )
    speedup_estimate = 1.0 - ratio_estimate
    speedup_lower = 1.0 - ratio_upper
    speedup_upper = 1.0 - ratio_lower
    median_passed = speedup_estimate >= _MEDIAN_SPEEDUP_GATE
    confidence_passed = speedup_lower >= _CONFIDENCE_SPEEDUP_GATE
    passed = bool(median_passed and confidence_passed)

    failures = []
    if not median_passed:
        failures.append("median native speedup is below 20%")
    if not confidence_passed:
        failures.append("95% confidence lower bound is below 10% speedup")

    return {
        "statistic": "median(native_seconds) / median(portable_seconds)",
        "pairing": (
            "native and portable calls interleaved by repetition with "
            "alternating execution order"
        ),
        "sample_count": int(paired_ratios.size),
        "bootstrap": {
            "seed": _BOOTSTRAP_SEED,
            "resamples": _BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
        },
        "native_to_portable_ratio": {
            "estimate": ratio_estimate,
            "confidence_interval_95": [ratio_lower, ratio_upper],
            "paired_sample_ratios": [
                float(value) for value in paired_ratios
            ],
        },
        "native_speedup_fraction": {
            "estimate": speedup_estimate,
            "confidence_interval_95": [speedup_lower, speedup_upper],
        },
        "thresholds": {
            "minimum_median_speedup_fraction": _MEDIAN_SPEEDUP_GATE,
            "minimum_confidence_lower_bound_fraction": (
                _CONFIDENCE_SPEEDUP_GATE
            ),
        },
        "checks": {
            "median_native_speedup_at_least_20_percent": bool(
                median_passed
            ),
            "confidence_lower_bound_at_least_10_percent": bool(
                confidence_passed
            ),
        },
        "passed": passed,
        "failures": failures,
    }


def _measure_automatic(
    function: Callable[[Any], Any],
    argument: Any,
    *,
    device: Any,
    warmups: int,
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    _clear_execution_caches()
    memory_before = _device_memory_stats(device)
    first_value, first_seconds = _timed_call(function, argument)
    for _ in range(warmups):
        _timed_call(function, argument)
    warm_samples = [_timed_call(function, argument)[1] for _ in range(repeats)]
    memory_after = _device_memory_stats(device)
    return first_value, {
        "trace_lower_seconds": None,
        "compile_seconds": None,
        "first_execution_seconds": first_seconds,
        "first_execution_includes_compilation": True,
        "compile_note": (
            "The automatic public route is eager. Its internal compilation is "
            "included in the synchronized first call because an outer jit "
            "intentionally selects the portable route."
        ),
        "warm": _distribution(warm_samples),
        "device_memory_stats": {
            "before": memory_before,
            "after": memory_after,
        },
    }


def _prepare_jitted_measurement(
    function: Callable[[Any], Any],
    argument: Any,
    *,
    device: Any,
) -> tuple[Any, dict[str, Any], Callable[[Any], Any]]:
    _clear_execution_caches()
    memory_before = _device_memory_stats(device)
    jitted = jax.jit(function)

    started = time.perf_counter()
    lowered = jitted.lower(argument)
    lower_seconds = time.perf_counter() - started

    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started

    first_value, first_seconds = _timed_call(executable, argument)
    return first_value, {
        "trace_lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_seconds,
        "first_execution_includes_compilation": False,
        "compile_note": None,
        "warm": None,
        "device_memory_stats": {
            "before": memory_before,
            "after": None,
        },
    }, executable


def _measure_runners(
    runners: dict[str, _Runner],
    argument: Any,
    *,
    oracle_label: str,
    dtype: np.dtype,
    device: Any,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    oracle_runner = runners[oracle_label]
    oracle_function = (
        oracle_runner.function
        if oracle_runner.eager
        else jax.jit(oracle_runner.function)
    )
    oracle = _synchronize(oracle_function(argument))
    measurements = {}
    compiled_runners = {}
    for label, runner in runners.items():
        if runner.eager:
            output, timing = _measure_automatic(
                runner.function,
                argument,
                device=device,
                warmups=warmups,
                repeats=repeats,
            )
        else:
            output, timing, executable = _prepare_jitted_measurement(
                runner.function,
                argument,
                device=device,
            )
            compiled_runners[label] = executable
        measurements[label] = {
            "timing": timing,
            "correctness": _correctness(output, oracle, dtype=dtype),
            "output_bytes": _output_bytes(output),
        }

    compiled_labels = tuple(compiled_runners)
    if compiled_labels:
        for warmup in range(warmups):
            labels = (
                compiled_labels
                if warmup % 2 == 0
                else tuple(reversed(compiled_labels))
            )
            for label in labels:
                _timed_call(compiled_runners[label], argument)

        warm_samples = {label: [] for label in compiled_labels}
        for repetition in range(repeats):
            labels = (
                compiled_labels
                if repetition % 2 == 0
                else tuple(reversed(compiled_labels))
            )
            for label in labels:
                _, seconds = _timed_call(compiled_runners[label], argument)
                warm_samples[label].append(seconds)

        memory_after = _device_memory_stats(device)
        for label in compiled_labels:
            timing = measurements[label]["timing"]
            timing["warm"] = _distribution(warm_samples[label])
            timing["device_memory_stats"]["after"] = memory_after
    return measurements


def _output_bytes(value: Any) -> int:
    return sum(
        int(leaf.size) * np.dtype(leaf.dtype).itemsize
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _correctness(
    actual: Any,
    expected: Any,
    *,
    dtype: np.dtype,
) -> dict[str, Any]:
    if hasattr(actual, "spec") or hasattr(expected, "spec"):
        if actual.spec != expected.spec:
            raise AssertionError(
                f"tensor specifications differ: {actual.spec!r} != "
                f"{expected.spec!r}"
            )
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    if len(actual_leaves) != len(expected_leaves):
        raise AssertionError(
            f"output leaf counts differ: {len(actual_leaves)} != "
            f"{len(expected_leaves)}"
        )

    tolerance = 3e-5 if dtype == np.dtype(np.float32) else 3e-11
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        actual_host = np.asarray(actual_leaf)
        expected_host = np.asarray(expected_leaf)
        difference = np.abs(actual_host - expected_host)
        if difference.size:
            maximum_absolute = max(maximum_absolute, float(difference.max()))
            denominator = np.maximum(np.abs(expected_host), tolerance)
            maximum_relative = max(
                maximum_relative,
                float((difference / denominator).max()),
            )
        np.testing.assert_allclose(
            actual_host,
            expected_host,
            atol=tolerance,
            rtol=tolerance,
        )
    return {
        "oracle": "forced portable implementation",
        "atol": tolerance,
        "rtol": tolerance,
        "maximum_absolute_error": maximum_absolute,
        "maximum_scaled_relative_error": maximum_relative,
        "passed": True,
    }


def _device_metadata(device: Any) -> dict[str, Any]:
    capability = _normalized_compute_capability(device)
    client = getattr(device, "client", None)
    return {
        "id": int(getattr(device, "id", 0)),
        "local_hardware_id": int(getattr(device, "local_hardware_id", 0)),
        "platform": str(getattr(device, "platform", "unknown")),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "compute_capability": (None if capability is None else list(capability)),
        "platform_version": str(getattr(client, "platform_version", "unknown")),
        "runtime_type": str(getattr(client, "runtime_type", "unknown")),
    }


def _metadata(args: argparse.Namespace, device: Any) -> dict[str, Any]:
    return {
        "scope": "wordwise ordinary signatures and scalar-FSSK recursion",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform_module.python_version(),
        "host_platform": platform_module.platform(),
        "tensordev_version": _package_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "numpy_version": np.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "selected_device": _device_metadata(device),
        "automatic_native_dispatch": {
            "enabled": _automatic_wordwise_release_eligible(),
            "reason": (
                "enabled only after exact target and workload regions pass "
                "the recorded correctness and performance gates"
            ),
        },
        "value_and_gradient": {
            "measured": False,
            "reason": (
                "The forced native executors do not yet expose the custom "
                "derivative bridge required for a native gradient. Public "
                "differentiation intentionally selects the portable route."
            ),
        },
        "device_memory_stats_note": (
            "Each runner records every field exposed by JAX before and after "
            "its measurement. Jitted native and portable warm calls are "
            "interleaved, so they share the final snapshot. Null means the "
            "backend exposes no memory_stats mapping; process-lifetime peak "
            "fields may include earlier runners when the backend cannot "
            "reset them."
        ),
        "configuration": {
            "workloads": list(args.workloads),
            "families": list(args.families),
            "dtypes": list(args.dtypes),
            "block_modes": list(args.block_modes),
            "batch": args.batch,
            "steps": args.steps,
            "block_size": args.block_size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "tile_words": args.tile_words,
            "tile_prime_words": args.tile_prime_words,
            "fssk_outputs": list(args.fssk_outputs),
            "coefficient_modes": list(args.coefficient_modes),
            "initial_states": list(args.initial_states),
            "state_dim": args.state_dim,
            "quad_order": args.quad_order,
            "seed": args.seed,
            "device_index": args.device_index,
        },
    }


def _case_metadata(
    *,
    workload: str,
    core: Any,
    truncation: int | tuple[int, int],
    family: str,
    dtype_name: str,
    block_mode: str,
    block_size: int | None,
    accumulate: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "family": family,
        "partially_symmetrized": bool(
            getattr(core, "partially_symmetrized", False)
        ),
        "coordinates": getattr(core, "coordinates", "standard"),
        "grading": getattr(core, "grading", "unknown"),
        "truncation": (
            list(truncation) if isinstance(truncation, tuple) else truncation
        ),
        "dtype": dtype_name,
        "batch": args.batch,
        "steps": args.steps,
        "block_mode": block_mode,
        "block_size": block_size,
        "accumulate": accumulate,
    }


def _layout_metadata(plan: Any, construction_seconds: float) -> dict[str, Any]:
    return {
        "construction_seconds": construction_seconds,
        "retained_metadata_bytes": plan.memory_bytes(),
        "output_coordinates": plan.output_size,
    }


def _measure_ordinary_case(
    *,
    family: str,
    dtype_name: str,
    block_mode: str,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    dtype = np.dtype(dtype_name)
    core, truncation = _case(family)
    block_size, accumulate = _block_options(block_mode, args.block_size)
    increments = _make_input(
        batch=args.batch,
        steps=args.steps,
        dtype=dtype,
        seed=args.seed,
        device=device,
    )

    probe = _prepare(
        increments,
        core=core,
        truncation=truncation,
        block_size=block_size,
        accumulate=accumulate,
    )
    if not ordinary_wordwise_candidate_eligible(probe):
        raise RuntimeError(
            f"native wordwise execution is unexpectedly ineligible for "
            f"{family}, {dtype_name}, {block_mode}"
        )

    clear_layout_plan_cache()
    started = time.perf_counter()
    plan = build_layout_plan(core, truncation, alphabet_dim=2)
    plan_seconds = time.perf_counter() - started

    runners = _ordinary_runners(
        core=core,
        truncation=truncation,
        block_size=block_size,
        accumulate=accumulate,
        tile_words=args.tile_words,
        tile_prime_words=args.tile_prime_words,
    )
    measurements = _measure_runners(
        runners,
        increments,
        oracle_label="forced-portable-jit",
        dtype=dtype,
        device=device,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    native_portable_gate = _paired_native_portable_gate(
        measurements["forced-pallas-jit"]["timing"]["warm"][
            "samples_seconds"
        ],
        measurements["forced-portable-jit"]["timing"]["warm"][
            "samples_seconds"
        ],
    )

    return {
        **_case_metadata(
            workload="ordinary",
            core=core,
            truncation=truncation,
            family=family,
            dtype_name=dtype_name,
            block_mode=block_mode,
            block_size=block_size,
            accumulate=accumulate,
            args=args,
        ),
        "layout_plan": _layout_metadata(plan, plan_seconds),
        "native_portable_warm_gate": native_portable_gate,
        "measurements": measurements,
    }


def _fssk_case_resources(
    *,
    plan: Any,
    batch: int,
    steps: int,
    block_size: int | None,
    state_dim: int,
    coefficient_steps: int,
    maximum_order: int,
) -> dict[str, int]:
    emitted = 1 if block_size is None else steps // block_size
    positive_coordinates = plan.output_size - 1
    input_elements = batch * steps * plan.alphabet_dim
    state_elements = batch * state_dim * positive_coordinates
    emitted_state_elements = emitted * state_elements
    coefficient_elements = coefficient_steps * (
        state_dim**2
        + maximum_order * state_dim
        + max(maximum_order - 1, 0) * state_dim**2
    )
    bounded_elements = (
        input_elements
        + state_elements
        + emitted_state_elements
        + coefficient_elements
    )
    if bounded_elements > _MAX_CASE_ELEMENTS:
        raise SystemExit(
            "FSSK case exceeds the benchmark resource guard: "
            f"{bounded_elements:,} estimated array elements > "
            f"{_MAX_CASE_ELEMENTS:,}. Reduce --batch, --steps, or --state-dim."
        )
    return {
        "input_elements": input_elements,
        "state_elements": state_elements,
        "emitted_state_elements": emitted_state_elements,
        "coefficient_elements": coefficient_elements,
        "bounded_elements": bounded_elements,
        "guard_elements": _MAX_CASE_ELEMENTS,
    }


def _measure_fssk_case(
    *,
    family: str,
    dtype_name: str,
    block_mode: str,
    coefficient_mode: str,
    initial_state_mode: str,
    output_kind: str,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    dtype = np.dtype(dtype_name)
    core, truncation = _case(family)
    core, seq_core = fssk_state_update._resolve_fssk_core_pair(core, None)
    block_size, accumulate = _block_options(block_mode, args.block_size)
    projected = _make_input(
        batch=args.batch,
        steps=args.steps,
        dtype=dtype,
        seed=args.seed,
        device=device,
    )

    clear_layout_plan_cache()
    started = time.perf_counter()
    plan = build_layout_plan(core, truncation, alphabet_dim=2)
    plan_seconds = time.perf_counter() - started
    maximum_order = sum(truncation) if isinstance(truncation, tuple) else truncation
    coefficient_steps = 1 if coefficient_mode == "uniform" else args.steps
    resources = _fssk_case_resources(
        plan=plan,
        batch=args.batch,
        steps=args.steps,
        block_size=block_size,
        state_dim=args.state_dim,
        coefficient_steps=coefficient_steps,
        maximum_order=maximum_order,
    )
    kernel = _make_fssk_kernel(
        state_dim=args.state_dim,
        dtype=dtype,
        quad_order=args.quad_order,
        device=device,
    )
    dt, coefficients = _make_fssk_time_and_coefficients(
        kernel,
        mode=coefficient_mode,
        steps=args.steps,
        maximum_order=maximum_order,
        dtype=dtype,
        device=device,
    )
    initial_state = _make_fssk_initial_state(
        mode=initial_state_mode,
        core=core,
        truncation=truncation,
        plan=plan,
        state_dim=args.state_dim,
        dtype=dtype,
        device=device,
    )
    if not fssk_q1_wordwise_candidate_eligible(
        core=core,
        seq_core=seq_core,
        q=1,
        reference=projected,
        differentiable_inputs=(projected, coefficients, initial_state),
    ):
        raise RuntimeError(
            "native scalar-FSSK execution is unexpectedly ineligible for "
            f"{family}, {dtype_name}, {block_mode}, {coefficient_mode}, "
            f"{initial_state_mode}"
        )

    runners = _fssk_runners(
        output_kind=output_kind,
        coefficients=coefficients,
        dt=dt,
        kernel=kernel,
        core=core,
        seq_core=seq_core,
        truncation=truncation,
        maximum_order=maximum_order,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
    )
    measurements = _measure_runners(
        runners,
        projected,
        oracle_label="forced-portable-jit",
        dtype=dtype,
        device=device,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    native_portable_gate = _paired_native_portable_gate(
        measurements["forced-private-native-jit"]["timing"]["warm"][
            "samples_seconds"
        ],
        measurements["forced-portable-jit"]["timing"]["warm"][
            "samples_seconds"
        ],
    )

    return {
        **_case_metadata(
            workload="fssk",
            core=core,
            truncation=truncation,
            family=family,
            dtype_name=dtype_name,
            block_mode=block_mode,
            block_size=block_size,
            accumulate=accumulate,
            args=args,
        ),
        "fssk_output": output_kind,
        "timed_scope": (
            "public fssk_state_from_coef recursion"
            if output_kind == "state"
            else "public fssk_vsig path including coefficients and readout"
        ),
        "private_native_readout_path": (
            None
            if output_kind == "state"
            else (
                "fused terminal signature emission"
                if block_size is None
                else "native state followed by readout"
            )
        ),
        "coefficient_mode": coefficient_mode,
        "coefficient_steps": coefficient_steps,
        "coefficient_bytes": _output_bytes(coefficients),
        "initial_state": initial_state_mode,
        "state_dim": args.state_dim,
        "quad_order": args.quad_order,
        "resource_estimate": resources,
        "layout_plan": _layout_metadata(plan, plan_seconds),
        "native_portable_warm_gate": native_portable_gate,
        "measurements": measurements,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    device = _select_device(args.device_index)
    print(
        "TensorDev GPU benchmark: synchronized wordwise workloads across "
        "all supported layouts.",
        file=sys.stderr,
    )

    cases = []
    for workload in args.workloads:
        for family in args.families:
            for dtype_name in args.dtypes:
                for block_mode in args.block_modes:
                    if workload == "ordinary":
                        print(
                            "measuring ordinary, "
                            f"{family}, {dtype_name}, {block_mode}",
                            file=sys.stderr,
                        )
                        cases.append(
                            _measure_ordinary_case(
                                family=family,
                                dtype_name=dtype_name,
                                block_mode=block_mode,
                                args=args,
                                device=device,
                            )
                        )
                        continue

                    for coefficient_mode in args.coefficient_modes:
                        for initial_state_mode in args.initial_states:
                            for output_kind in args.fssk_outputs:
                                print(
                                    "measuring fssk, "
                                    f"{output_kind}, {family}, {dtype_name}, "
                                    f"{block_mode}, {coefficient_mode}, "
                                    f"{initial_state_mode}",
                                    file=sys.stderr,
                                )
                                cases.append(
                                    _measure_fssk_case(
                                        family=family,
                                        dtype_name=dtype_name,
                                        block_mode=block_mode,
                                        coefficient_mode=coefficient_mode,
                                        initial_state_mode=initial_state_mode,
                                        output_kind=output_kind,
                                        args=args,
                                        device=device,
                                    )
                                )

    result = {
        "metadata": _metadata(args, device),
        "cases": cases,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
