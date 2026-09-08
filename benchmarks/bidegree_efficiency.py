#!/usr/bin/env python3
"""Benchmark native bidegree pruning against total-degree execution.

This is a diagnostic benchmark, not a timing test.  It deliberately reports
both cold and steady-state phases and synchronizes every output PyTree leaf.
The two matched comparisons have the same maximum total word length:

* total degree 6 versus bidegree ``(3, 3)`` on a ``(2, 2)`` split;
* total degree 8 versus bidegree ``(4, 4)`` on a ``(2, 2)`` split.

The optional total-6 versus ``(4, 4)`` comparison is labelled non-equivalent:
the latter retains words of lengths 7 and 8 and has a substantially larger
payload, so it is not a sensible speed requirement.

Examples
--------
Quick smoke run::

    PYTHONPATH=src python benchmarks/bidegree_efficiency.py \
        --preset smoke --cases matched6 --modes outer-jit

Standard direct-public and whole-call-JIT run::

    PYTHONPATH=src python benchmarks/bidegree_efficiency.py \
        --preset standard --output benchmarks/results/bidegree_efficiency.json

Include the intentionally mismatched diagnostic::

    PYTHONPATH=src python benchmarks/bidegree_efficiency.py \
        --preset standard --include-mismatched
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


# Always benchmark the checkout containing this script, even when a different
# tensordev release is installed in the active environment.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(_REPOSITORY_ROOT / "notebooks"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import tensordev as td  # noqa: E402
from _validation_util.timing_utils import (  # noqa: E402
    block_until_ready as _block_until_ready,
    time_call as _measure_call,
)


_DIMS = (2, 2)
_DIMENSION = sum(_DIMS)


@dataclass(frozen=True)
class Scenario:
    """One total/bidegree comparison."""

    name: str
    total_truncation: int
    bidegree_truncation: tuple[int, int]
    matched_max_order: bool
    note: str


SCENARIOS = {
    "matched6": Scenario(
        name="matched_order_6",
        total_truncation=6,
        bidegree_truncation=(3, 3),
        matched_max_order=True,
        note="Same maximum word length; bidegree is the rectangular projection.",
    ),
    "matched8": Scenario(
        name="matched_order_8",
        total_truncation=8,
        bidegree_truncation=(4, 4),
        matched_max_order=True,
        note="Same maximum word length; bidegree is the rectangular projection.",
    ),
    "mismatched6v44": Scenario(
        name="mismatched_total_6_vs_bidegree_4_4",
        total_truncation=6,
        bidegree_truncation=(4, 4),
        matched_max_order=False,
        note=(
            "Non-equivalent diagnostic: bidegree (4,4) also retains words of "
            "lengths 7 and 8."
        ),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "standard"),
        default="standard",
        help=(
            "smoke uses 16/4 steps and one hot sample; standard uses 512/24 "
            "steps and seven hot samples (default: standard)."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("matched6", "matched8"),
        default=("matched6", "matched8"),
        help="Matched maximum-order comparisons to run.",
    )
    parser.add_argument(
        "--include-mismatched",
        action="store_true",
        help="Also report total degree 6 versus bidegree (4,4).",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("signature", "volterra"),
        default=("signature", "volterra"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("direct", "outer-jit"),
        default=("direct", "outer-jit"),
        help=(
            "direct calls the public API normally; outer-jit compiles the "
            "complete public call as one executable."
        ),
    )
    parser.add_argument(
        "--signature-steps",
        type=int,
        default=None,
        help="Override the preset's number of ordinary-signature increments.",
    )
    parser.add_argument(
        "--volterra-steps",
        type=int,
        default=None,
        help="Override the preset's number of Volterra-signature increments.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=None,
        help="Override the preset's untimed warm-up count.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override the preset's number of synchronized hot samples.",
    )
    parser.add_argument(
        "--volterra-scheme",
        choices=("quadratic", "fft"),
        default="quadratic",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--clear-caches",
        action="store_true",
        help=(
            "Call jax.clear_caches() before every row. This makes cold phases "
            "more isolated but considerably increases total benchmark time."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path (or directory). A sibling CSV containing "
            "the flat rows is written as well."
        ),
    )
    args = parser.parse_args()

    for name in ("signature_steps", "volterra_steps", "repeats"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups is not None and args.warmups < 0:
        parser.error("--warmups must be non-negative")
    return args


def _time_call(function: Callable[[jax.Array], Any], argument: jax.Array):
    result, record = _measure_call(function, argument)
    return result, record.wall_s


def _path(steps: int, dtype) -> jax.Array:
    """Build the same deterministic four-dimensional path for every core."""

    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    values = np.stack(
        (
            t + 0.08 * np.sin(6.0 * np.pi * t),
            0.35 * np.sin(2.0 * np.pi * t) + 0.12 * t,
            0.28 * np.cos(3.0 * np.pi * t) - 0.28,
            0.22 * np.sin(5.0 * np.pi * t) + 0.17 * t**2,
        ),
        axis=-1,
    )
    return _block_until_ready(jnp.asarray(values, dtype=dtype))


def _total_coordinates(dimension: int, truncation: int) -> int:
    return sum(dimension**degree for degree in range(truncation + 1))


def _bidegree_coordinates(
    dims: tuple[int, int], truncation: tuple[int, int]
) -> int:
    d_prime, d_doubleprime = dims
    n_max, m_max = truncation
    return sum(
        math.comb(n + m, n) * d_prime**n * d_doubleprime**m
        for n in range(n_max + 1)
        for m in range(m_max + 1)
    )


def _tree_payload(value: Any) -> tuple[int, int, int]:
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(value)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    ]
    elements = sum(int(leaf.size) for leaf in leaves)
    nbytes = sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in leaves)
    return len(leaves), elements, nbytes


def _sum_cost_analysis(raw: Any) -> dict[str, float]:
    """Normalize JAX's dict or per-device list-of-dicts cost formats."""

    dictionaries: Iterable[dict]
    if isinstance(raw, dict):
        dictionaries = (raw,)
    elif isinstance(raw, (tuple, list)):
        dictionaries = (item for item in raw if isinstance(item, dict))
    else:
        return {}

    result: dict[str, float] = {}
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            result[key] = result.get(key, 0.0) + number
    return result


def _compiled_analysis_placeholder() -> dict[str, Any]:
    return {
        "xla_flops_raw": None,
        "xla_flops_scan_scaled": None,
        "xla_bytes_accessed_raw": None,
        "xla_bytes_accessed_scan_scaled": None,
        "xla_transcendentals_raw": None,
        "xla_argument_bytes": None,
        "xla_output_bytes": None,
        "xla_temporary_bytes": None,
        "xla_generated_code_bytes": None,
        "stablehlo_operations": None,
        "stablehlo_text_bytes": None,
        "cost_analysis_error": None,
        "memory_analysis_error": None,
        "stablehlo_analysis_error": None,
    }


def _compiled_analysis(compiled, lowered, steps: int) -> dict[str, Any]:
    result = _compiled_analysis_placeholder()

    try:
        cost = _sum_cost_analysis(compiled.cost_analysis())
        flops = cost.get("flops")
        accessed = cost.get("bytes accessed")
        result.update(
            xla_flops_raw=flops,
            # On the JAX CPU backend cost_analysis reports a scan/while body
            # once.  Keep the raw value authoritative and expose this clearly
            # labelled diagnostic estimate rather than silently correcting it.
            xla_flops_scan_scaled=None if flops is None else flops * steps,
            xla_bytes_accessed_raw=accessed,
            xla_bytes_accessed_scan_scaled=(
                None if accessed is None else accessed * steps
            ),
            xla_transcendentals_raw=cost.get("transcendentals"),
        )
    except Exception as error:  # backend/version diagnostic, not fatal
        result["cost_analysis_error"] = repr(error)

    try:
        memory = compiled.memory_analysis()
        result.update(
            xla_argument_bytes=getattr(memory, "argument_size_in_bytes", None),
            xla_output_bytes=getattr(memory, "output_size_in_bytes", None),
            xla_temporary_bytes=getattr(memory, "temp_size_in_bytes", None),
            xla_generated_code_bytes=getattr(
                memory, "generated_code_size_in_bytes", None
            ),
        )
    except Exception as error:  # backend/version diagnostic, not fatal
        result["memory_analysis_error"] = repr(error)

    try:
        hlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
        result["stablehlo_operations"] = hlo_text.count("stablehlo.")
        result["stablehlo_text_bytes"] = len(hlo_text.encode("utf-8"))
    except Exception as error:  # backend/version diagnostic, not fatal
        result["stablehlo_analysis_error"] = repr(error)
    return result


def _build_core(kind: str, truncation: int | tuple[int, int]):
    start = time.perf_counter()
    if kind == "total":
        core = td.Jax()
        seq_core = td.JaxSequentialCore()
        plan_memory_bytes = 0
    else:
        core = td.make_core(
            dims=_DIMS,
            max_trunc=truncation,
            default_trunc=truncation,
            precompute_shuffle=False,
        )
        seq_core = core.make_sequential_core()
        plan_memory_bytes = int(core.memory_bytes())
    setup_s = float(time.perf_counter() - start)
    return core, seq_core, plan_memory_bytes, setup_s


def _make_public_call(
    workload: str,
    *,
    truncation,
    core,
    seq_core,
    steps: int,
    dtype,
    volterra_scheme: str,
):
    kernel_setup_s = 0.0
    start = time.perf_counter()
    if workload == "signature":

        def public_call(path):
            return td.path_signature(
                path,
                trunc=truncation,
                axis=-2,
                parallel=False,
                core=core,
                seq_core=seq_core,
            )

    else:
        kernel_start = time.perf_counter()
        kernel = td.FractionalKernel(
            beta=jnp.asarray([0.7], dtype=dtype),
            A=jnp.eye(_DIMENSION, dtype=dtype)[None, :, :],
        )
        _block_until_ready(kernel)
        kernel_setup_s = float(time.perf_counter() - kernel_start)
        dt = jnp.asarray(1.0 / steps, dtype=dtype)
        _block_until_ready(dt)

        def public_call(path):
            return td.vsig(
                path,
                kernel=kernel,
                trunc=truncation,
                dt=dt,
                axis=-2,
                order=0,
                scheme=volterra_scheme,
                core=core,
                seq_core=seq_core,
            )

    callable_setup_s = float(time.perf_counter() - start) - kernel_setup_s
    return public_call, kernel_setup_s, max(callable_setup_s, 0.0)


def _run_row(
    *,
    scenario: Scenario,
    kind: str,
    workload: str,
    mode: str,
    path: jax.Array,
    steps: int,
    warmups: int,
    repeats: int,
    dtype,
    volterra_scheme: str,
    clear_caches: bool,
) -> dict[str, Any]:
    if clear_caches:
        jax.clear_caches()

    truncation = (
        scenario.total_truncation
        if kind == "total"
        else scenario.bidegree_truncation
    )
    retained_coordinates = (
        _total_coordinates(_DIMENSION, truncation)
        if kind == "total"
        else _bidegree_coordinates(_DIMS, truncation)
    )
    max_order = (
        int(truncation)
        if kind == "total"
        else int(sum(truncation))
    )
    dense_coordinates = _total_coordinates(_DIMENSION, max_order)

    core, seq_core, plan_memory_bytes, core_setup_s = _build_core(
        kind, truncation
    )
    public_call, kernel_setup_s, callable_setup_s = _make_public_call(
        workload,
        truncation=truncation,
        core=core,
        seq_core=seq_core,
        steps=steps,
        dtype=dtype,
        volterra_scheme=volterra_scheme,
    )

    lower_s = None
    compile_s = None
    analysis = _compiled_analysis_placeholder()
    if mode == "outer-jit":
        whole_call = jax.jit(public_call)
        start = time.perf_counter()
        lowered = whole_call.lower(path)
        lower_s = float(time.perf_counter() - start)
        start = time.perf_counter()
        runner = lowered.compile()
        compile_s = float(time.perf_counter() - start)
        analysis = _compiled_analysis(runner, lowered, steps)
    else:
        runner = public_call

    first_result, first_s = _time_call(runner, path)
    for _ in range(warmups):
        _time_call(runner, path)
    hot_s = [_time_call(runner, path)[1] for _ in range(repeats)]

    leaf_count, output_elements, output_bytes = _tree_payload(first_result)
    if output_elements != retained_coordinates:
        raise AssertionError(
            f"retained-coordinate payload mismatch for "
            f"{scenario.name}/{kind}/{workload}: "
            f"expected {retained_coordinates} elements, got {output_elements}"
        )

    return {
        "scenario": scenario.name,
        "comparison_is_matched_max_order": scenario.matched_max_order,
        "scenario_note": scenario.note,
        "kind": kind,
        "workload": workload,
        "mode": mode,
        "volterra_scheme": volterra_scheme if workload == "volterra" else None,
        "dtype": np.dtype(dtype).name,
        "steps": steps,
        "dimension": _DIMENSION,
        "bidegree_dims": list(_DIMS) if kind == "bidegree" else None,
        "truncation": list(truncation) if kind == "bidegree" else truncation,
        "max_total_order": max_order,
        "retained_coordinates": retained_coordinates,
        "dense_total_coordinates_at_max_order": dense_coordinates,
        "retained_fraction": retained_coordinates / dense_coordinates,
        "output_leaf_count": leaf_count,
        "output_elements": output_elements,
        "output_bytes": output_bytes,
        "core_plan_memory_bytes": plan_memory_bytes,
        "core_setup_s": core_setup_s,
        "kernel_setup_s": kernel_setup_s,
        "callable_setup_s": callable_setup_s,
        "lower_s": lower_s,
        "compile_s": compile_s,
        "first_execution_s": first_s,
        "warm_samples_s": hot_s,
        "warm_median_s": statistics.median(hot_s),
        "warm_mean_s": statistics.fmean(hot_s),
        "warm_min_s": min(hot_s),
        "warm_max_s": max(hot_s),
        **analysis,
    }


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = row["scenario"], row["workload"], row["mode"]
        grouped.setdefault(key, {})[row["kind"]] = row

    comparisons = []
    for (scenario, workload, mode), pair in grouped.items():
        if set(pair) != {"total", "bidegree"}:
            continue
        total = pair["total"]
        bidegree = pair["bidegree"]
        comparisons.append(
            {
                "scenario": scenario,
                "comparison_is_matched_max_order": total[
                    "comparison_is_matched_max_order"
                ],
                "workload": workload,
                "mode": mode,
                "bidegree_over_total_payload": _safe_ratio(
                    bidegree["output_elements"], total["output_elements"]
                ),
                "total_over_bidegree_warm_speedup": _safe_ratio(
                    total["warm_median_s"], bidegree["warm_median_s"]
                ),
                "bidegree_over_total_warm_time": _safe_ratio(
                    bidegree["warm_median_s"], total["warm_median_s"]
                ),
                "bidegree_over_total_raw_flops": _safe_ratio(
                    bidegree["xla_flops_raw"], total["xla_flops_raw"]
                ),
                "bidegree_over_total_output_bytes": _safe_ratio(
                    bidegree["output_bytes"], total["output_bytes"]
                ),
            }
        )
    return comparisons


def _seconds(value) -> str:
    return "-" if value is None else f"{1e3 * value:9.3f}"


def _print_results(
    rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> None:
    print()
    print(
        "scenario                             workload  mode       kind      "
        "payload    setup ms   lower ms compile ms   first ms    warm ms"
    )
    print("-" * 128)
    for row in rows:
        setup = row["core_setup_s"] + row["kernel_setup_s"] + row["callable_setup_s"]
        print(
            f"{row['scenario']:<36} "
            f"{row['workload']:<9} "
            f"{row['mode']:<10} "
            f"{row['kind']:<9} "
            f"{row['output_elements']:>9,d} "
            f"{_seconds(setup)} "
            f"{_seconds(row['lower_s'])} "
            f"{_seconds(row['compile_s'])} "
            f"{_seconds(row['first_execution_s'])} "
            f"{_seconds(row['warm_median_s'])}"
        )

    print()
    print(
        "Comparison ratios (speedup > 1 means native bidegree is faster; "
        "only matched rows are performance expectations):"
    )
    for comparison in comparisons:
        label = "matched" if comparison["comparison_is_matched_max_order"] else "NON-EQUIVALENT"
        speedup = comparison["total_over_bidegree_warm_speedup"]
        speedup_text = "-" if speedup is None else f"{speedup:.3f}x"
        payload = comparison["bidegree_over_total_payload"]
        print(
            f"  {comparison['scenario']} / {comparison['workload']} / "
            f"{comparison['mode']}: speedup={speedup_text}, "
            f"payload ratio={payload:.4f} ({label})"
        )


def _write_results(
    output: Path,
    *,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> tuple[Path, Path]:
    json_path = output
    if output.suffix.lower() != ".json":
        json_path = output / "bidegree_efficiency.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = json_path.with_suffix(".csv")

    document = {
        "metadata": metadata,
        "rows": rows,
        "comparisons": comparisons,
    }
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["truncation"] = json.dumps(flat["truncation"])
        flat["bidegree_dims"] = json.dumps(flat["bidegree_dims"])
        flat["warm_samples_s"] = json.dumps(flat["warm_samples_s"])
        csv_rows.append(flat)
    fieldnames = list(csv_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return json_path, csv_path


def main() -> None:
    args = _parse_args()
    if args.preset == "smoke":
        signature_steps = 16
        volterra_steps = 4
        warmups = 0
        repeats = 1
    else:
        signature_steps = 512
        volterra_steps = 24
        warmups = 2
        repeats = 7
    signature_steps = args.signature_steps or signature_steps
    volterra_steps = args.volterra_steps or volterra_steps
    warmups = warmups if args.warmups is None else args.warmups
    repeats = args.repeats or repeats
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32

    scenario_keys = list(dict.fromkeys(args.cases))
    if args.include_mismatched:
        scenario_keys.append("mismatched6v44")
    workloads = list(dict.fromkeys(args.workloads))
    modes = list(dict.fromkeys(args.modes))

    paths = {
        "signature": _path(signature_steps, dtype),
        "volterra": _path(volterra_steps, dtype),
    }
    steps_by_workload = {
        "signature": signature_steps,
        "volterra": volterra_steps,
    }

    print(f"JAX backend     : {jax.default_backend()}")
    print(f"JAX devices     : {[str(device) for device in jax.devices()]}")
    print(f"tensordev source: {Path(td.__file__).resolve()}")
    print(f"dtype           : {np.dtype(dtype).name}")
    print(f"scenarios       : {scenario_keys}")
    print(f"workloads       : {workloads}")
    print(f"modes           : {modes}")
    print(f"warmups/repeats : {warmups}/{repeats}")

    rows: list[dict[str, Any]] = []
    for scenario_key in scenario_keys:
        scenario = SCENARIOS[scenario_key]
        for workload in workloads:
            for mode_name in modes:
                mode = mode_name
                for kind in ("total", "bidegree"):
                    print(
                        f"running {scenario.name} / {workload} / {mode} / {kind}",
                        flush=True,
                    )
                    rows.append(
                        _run_row(
                            scenario=scenario,
                            kind=kind,
                            workload=workload,
                            mode=mode,
                            path=paths[workload],
                            steps=steps_by_workload[workload],
                            warmups=warmups,
                            repeats=repeats,
                            dtype=dtype,
                            volterra_scheme=args.volterra_scheme,
                            clear_caches=args.clear_caches,
                        )
                    )

    comparisons = _comparisons(rows)
    _print_results(rows, comparisons)

    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "tensordev_source": str(Path(td.__file__).resolve()),
        "dtype": np.dtype(dtype).name,
        "preset": args.preset,
        "signature_steps": signature_steps,
        "volterra_steps": volterra_steps,
        "warmups": warmups,
        "repeats": repeats,
        "clear_caches_between_rows": args.clear_caches,
        "volterra_scheme": args.volterra_scheme,
        "cost_analysis_note": (
            "Raw XLA cost is backend-provided. The scan-scaled fields multiply "
            "raw values by the number of steps because CPU cost analysis reports "
            "a scan/while body once; they are diagnostic estimates, not timings."
        ),
    }
    if args.output is not None:
        json_path, csv_path = _write_results(
            args.output,
            metadata=metadata,
            rows=rows,
            comparisons=comparisons,
        )
        print(f"\nWrote {json_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
