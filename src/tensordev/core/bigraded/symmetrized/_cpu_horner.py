"""Optional fused CPU execution for partially symmetrized Horner updates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import importlib
import math
from threading import Lock
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from tensordev.core.bigraded.types import Bidegree, BigradedTensor


_TARGETS = {
    np.dtype(np.float32): "tensordev_cpu_sym_horner_f32_v2",
    np.dtype(np.float64): "tensordev_cpu_sym_horner_f64_v2",
}
_MIN_SCALAR_WORK = 32_768
_MIN_SCALAR_WORK_PER_BATCH_Q2 = 3_000
_MIN_SCALAR_WORK_PER_BATCH_Q3_PLUS = 1_500
_INT32_MAX = np.iinfo(np.int32).max
_REGISTRATION_LOCK = Lock()
_REGISTRATION_STATE: bool | None = None
_STATE_TYPE_NAME = "tensordev.ragged_horner_state.v1"


class _UnsupportedNativePlan(ValueError):
    pass


@dataclass(frozen=True, slots=True, eq=False)
class _NativeHornerPlan:
    spec: object
    grades: tuple[Bidegree, ...]
    widths: tuple[int, ...]
    metadata: np.ndarray
    selected_edges: np.ndarray
    collision_targets: np.ndarray


def _readonly_int32(values) -> np.ndarray:
    array = np.asarray(values)
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < np.iinfo(np.int32).min or maximum > _INT32_MAX:
            raise _UnsupportedNativePlan("native Horner indices exceed signed int32")
    result = np.asarray(array, dtype=np.int32)
    result.setflags(write=False)
    return result


def _cpu_backend_enabled() -> bool:
    platforms = jax.config.jax_platforms
    return not platforms or "cpu" in platforms.split(",")


def _native_registrations():
    required = frozenset(_TARGETS.values())
    for module_name in ("tensordev._native_cpu", "tensordev_native_cpu"):
        try:
            extension = importlib.import_module(module_name)
            registrations = extension.registrations()
            type_registrations = extension.type_registrations()
        except (ImportError, OSError, AttributeError):
            continue
        if not isinstance(registrations, Mapping):
            raise TypeError(f"{module_name}.registrations() must return a mapping")
        if not isinstance(type_registrations, Mapping):
            raise TypeError(
                f"{module_name}.type_registrations() must return a mapping"
            )
        if not required.issubset(registrations):
            continue
        if type_registrations.get(_STATE_TYPE_NAME) is None:
            continue
        return registrations, type_registrations
    return None


def _register_targets() -> bool:
    global _REGISTRATION_STATE
    if not _cpu_backend_enabled():
        return False
    if _REGISTRATION_STATE is not None:
        return _REGISTRATION_STATE
    with _REGISTRATION_LOCK:
        if _REGISTRATION_STATE is not None:
            return _REGISTRATION_STATE
        registration_maps = _native_registrations()
        if registration_maps is None:
            _REGISTRATION_STATE = False
            return False
        registrations, type_registrations = registration_maps
        state_type = type_registrations[_STATE_TYPE_NAME]
        jax.devices("cpu")
        try:
            jax.ffi.register_ffi_type(
                _STATE_TYPE_NAME,
                state_type,
                platform="cpu",
            )
        except ValueError as error:
            if "already registered" not in str(error):
                raise
        for name in frozenset(_TARGETS.values()):
            try:
                jax.ffi.register_ffi_target(
                    name,
                    registrations[name],
                    platform="cpu",
                    api_version=1,
                )
            except ValueError as error:
                if "already registered" not in str(error):
                    raise
        _REGISTRATION_STATE = True
        return True


def _destination_arrays(plan) -> tuple[np.ndarray, np.ndarray, int]:
    source_count = int(plan.source_rank_count)
    prefix_count = int(plan.doubleprime_rank_count)
    edge_count = int(plan.d_doubleprime * source_count)
    head_start = (int(plan.d_doubleprime) - 1) * source_count

    destination = plan.destination_plan
    if destination is not None:
        if (
            destination.primary_head_start != head_start
            or destination.primary_head_count != source_count
        ):
            raise _UnsupportedNativePlan(
                "the native Horner kernel requires the terminal letter head"
            )
        return (
            _readonly_int32(destination.selected_edge_ids),
            _readonly_int32(destination.collision_target_ranks),
            int(destination.tail_primary_count),
        )

    if plan.target_ranks is None:
        raise AssertionError("missing double-prime generator plan")
    targets = np.asarray(plan.target_ranks, dtype=np.int64).reshape(-1)
    if targets.size != edge_count:
        raise _UnsupportedNativePlan("invalid generator edge map")
    head = np.arange(head_start, edge_count, dtype=np.int64)
    if not np.array_equal(targets[head], np.arange(source_count)):
        raise _UnsupportedNativePlan(
            "the generator edge map has no terminal-letter primary head"
        )

    primary = np.zeros(edge_count, dtype=np.bool_)
    primary[head] = True
    tail_edges = []
    for target in range(source_count, prefix_count):
        candidates = np.flatnonzero((targets == target) & ~primary)
        if candidates.size == 0:
            raise _UnsupportedNativePlan(
                f"double-prime rank {target} has no primary edge"
            )
        edge = int(candidates[0])
        primary[edge] = True
        tail_edges.append(edge)

    collision_edges = np.flatnonzero(~primary)
    order = np.argsort(targets[collision_edges], kind="stable")
    collision_edges = collision_edges[order]
    selected = np.concatenate(
        (
            np.asarray(tail_edges, dtype=np.int64),
            collision_edges,
        )
    )
    if selected.size != edge_count - source_count:
        raise _UnsupportedNativePlan("invalid destination edge partition")
    return (
        _readonly_int32(selected),
        _readonly_int32(targets[collision_edges]),
        len(tail_edges),
    )


def _scalar_work_per_batch(layout, truncation: Bidegree) -> int:
    max_order = sum(truncation)
    return sum(
        (max_order - sum(grade) + 1) * int(layout.block_width(grade))
        for grade in layout.grades
        if sum(grade)
    )


@lru_cache(maxsize=128)
def _compile_plan(plan_store, truncation: Bidegree) -> _NativeHornerPlan:
    layout = plan_store.resolve(
        truncation,
        include_scalar=True,
        coordinates="standard",
    )
    grades = tuple(layout.grades)
    grade_index = {grade: index for index, grade in enumerate(grades)}
    widths = tuple(int(layout.block_width(grade)) for grade in grades)
    if max(widths, default=0) > _INT32_MAX:
        raise _UnsupportedNativePlan("native Horner block width exceeds int32")

    rows = np.zeros((len(grades), 16), dtype=np.int64)
    selected_parts = []
    collision_parts = []
    selected_offset = 0
    collision_offset = 0
    d_prime, d_doubleprime = plan_store.dims
    max_order = sum(truncation)

    for index, grade in enumerate(grades):
        n, m = grade
        grade_plan = plan_store.grade_plan(grade)
        row = rows[index]
        row[0] = n + m
        row[1] = grade_plan.block_width
        row[2] = grade_plan.rank_count
        row[3] = grade_plan.dense_shape[1]
        row[4] = grade_index[(n - 1, m)] if n else -1
        row[5] = grade_index[(n, m - 1)] if m else -1
        row[13] = max_order
        row[14] = d_prime
        row[15] = d_doubleprime
        if not m:
            continue

        generator_plan = plan_store.doubleprime_generator_plan(grade)
        selected, collisions, tail_count = _destination_arrays(generator_plan)
        row[6] = generator_plan.doubleprime_rank_count
        row[7] = generator_plan.source_rank_count
        row[8] = selected_offset
        row[9] = selected.size
        row[10] = collision_offset
        row[11] = collisions.size
        row[12] = tail_count
        selected_parts.append(selected)
        collision_parts.append(collisions)
        selected_offset += selected.size
        collision_offset += collisions.size

    if rows.size and (rows.max() > _INT32_MAX or rows.min() < -1):
        raise _UnsupportedNativePlan("native Horner metadata exceeds int32")
    selected_edges = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty(0, dtype=np.int32)
    )
    collision_targets = (
        np.concatenate(collision_parts)
        if collision_parts
        else np.empty(0, dtype=np.int32)
    )
    return _NativeHornerPlan(
        spec=layout.spec,
        grades=grades,
        widths=widths,
        metadata=_readonly_int32(rows),
        selected_edges=_readonly_int32(selected_edges),
        collision_targets=_readonly_int32(collision_targets),
    )


def _portable_horner_blocks(core, blocks, z, truncation):
    from tensordev.core.bigraded.standard import StandardBigradedCore

    plan = _compile_plan(core.plan_store, truncation)
    result = StandardBigradedCore._fmexp_first_level(
        core,
        BigradedTensor(tuple(blocks), plan.spec),
        z,
        trunc=truncation,
    )
    return result.blocks


def _native_horner_blocks(core, blocks, z, truncation):
    plan = _compile_plan(core.plan_store, truncation)
    batch = blocks[0].shape[:-1]
    dtype = blocks[0].dtype
    flat_batch = math.prod(batch) if batch else 1
    flat_blocks = tuple(
        block.reshape((flat_batch, width)) for block, width in zip(blocks, plan.widths)
    )
    generator = z.reshape((flat_batch, sum(core.dims)))
    result_specs = tuple(
        jax.ShapeDtypeStruct((flat_batch, width), dtype) for width in plan.widths
    )
    call = jax.ffi.ffi_call(
        _TARGETS[np.dtype(dtype)],
        result_specs,
        vmap_method="expand_dims",
    )
    outputs = call(
        generator,
        *flat_blocks,
        metadata=plan.metadata.reshape(-1),
        selected_edges=plan.selected_edges,
        collision_targets=plan.collision_targets,
    )
    return tuple(
        output.reshape(batch + (width,)) for output, width in zip(outputs, plan.widths)
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 3))
def _fused_horner_blocks(core, blocks, z, truncation):
    return jax.lax.platform_dependent(
        blocks,
        z,
        cpu=lambda left, generator: _native_horner_blocks(
            core, left, generator, truncation
        ),
        default=lambda left, generator: _portable_horner_blocks(
            core, left, generator, truncation
        ),
    )


@_fused_horner_blocks.defjvp
def _fused_horner_blocks_jvp(core, truncation, primals, tangents):
    blocks, z = primals
    blocks_dot, z_dot = tangents
    primal = _fused_horner_blocks(core, blocks, z, truncation)
    _, tangent = jax.jvp(
        lambda left, generator: _portable_horner_blocks(
            core, left, generator, truncation
        ),
        (blocks, z),
        (blocks_dot, z_dot),
    )
    return primal, tangent


def try_fused_horner(core, g, z, truncation):
    """Return a fused standard-coordinate update or ``None`` if unsupported."""
    if core.coordinates != "standard" or not g.spec.include_scalar:
        return None
    if core.dims[0] != 1 or core.dims[1] <= 1:
        return None
    if not _cpu_backend_enabled():
        return None
    try:
        layout = core.plan_store.resolve(
            truncation,
            include_scalar=True,
            coordinates="standard",
        )
        scalar_work_per_batch = _scalar_work_per_batch(layout, truncation)
        batch = jnp.broadcast_shapes(g.batch_shape, jnp.shape(z)[:-1])
        batch_count = math.prod(int(size) for size in batch) if batch else 1
        dtype = np.dtype(jnp.result_type(g.blocks[0], jnp.asarray(z)))
    except (_UnsupportedNativePlan, TypeError, ValueError, OverflowError):
        return None
    if dtype not in _TARGETS:
        return None
    per_batch_threshold = (
        _MIN_SCALAR_WORK_PER_BATCH_Q2
        if core.dims[1] == 2
        else _MIN_SCALAR_WORK_PER_BATCH_Q3_PLUS
    )
    if scalar_work_per_batch < per_batch_threshold:
        return None
    if scalar_work_per_batch * batch_count < _MIN_SCALAR_WORK:
        return None
    if not _register_targets():
        return None
    try:
        plan = _compile_plan(core.plan_store, truncation)
    except (_UnsupportedNativePlan, TypeError, ValueError, OverflowError):
        return None
    z = jnp.asarray(z)
    batch = jnp.broadcast_shapes(g.batch_shape, z.shape[:-1])
    blocks = []
    for grade, width in zip(plan.grades, plan.widths):
        if g.spec.contains(grade):
            block = jnp.broadcast_to(g[grade], batch + (width,))
            block = block.astype(dtype)
        else:
            block = jnp.zeros(batch + (width,), dtype=dtype)
        blocks.append(block)
    generator = jnp.broadcast_to(
        z,
        batch + (sum(core.dims),),
    ).astype(dtype)
    outputs = _fused_horner_blocks(
        core,
        tuple(blocks),
        generator,
        truncation,
    )
    return BigradedTensor(tuple(outputs), plan.spec)


__all__ = []
