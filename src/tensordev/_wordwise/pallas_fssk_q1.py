"""Ordered scalar-FSSK Pallas execution in standard coordinates.

The executor consumes canonical arrays, keeps the complete initial state in
one compact operand, and gathers the prefix states required by each word
through static int32 metadata.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from threading import RLock
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tensordev._wordwise.dispatch import _colocate_array, _concrete_single_device
from tensordev._wordwise.layout import WordwiseLayoutPlan
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryError,
    PallasOrdinaryResourceError,
    PallasOrdinaryUnsupportedError,
    _compiler_warps,
    _effective_tile_words,
    _load_pallas,
    _normalize_block_size,
    _padded_word_count,
    _require_integer,
)
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.utils.precompute import _readonly


Array = jax.Array

_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_DEGREE = 32
_MAX_STATE_DIM = 128
_MAX_TILE_WORDS = 256
_MAX_LOCAL_BYTES = 64 * 1024
_MAX_PREFIX_INDEX_ELEMENTS = 8_000_000
_MAX_PREFIX_PLAN_BYTES = 64 * 1024**2
_EXECUTION_PLAN_CACHE_SIZE = 32
_EXECUTION_PLAN_CACHE_BYTES = 64 * 1024**2


class PallasFSSKPlanError(PallasOrdinaryError):
    """Raised when an ordered layout cannot supply an exact FSSK plan."""


@dataclass(frozen=True, slots=True)
class FSSKPlanLimits:
    """Bounds for the extra prefix-index metadata used by scalar FSSK."""

    max_prefix_index_elements: int = _MAX_PREFIX_INDEX_ELEMENTS
    max_plan_bytes: int = _MAX_PREFIX_PLAN_BYTES

    def __post_init__(self) -> None:
        for name in ("max_prefix_index_elements", "max_plan_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a positive integer.")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")


DEFAULT_FSSK_PLAN_LIMITS = FSSKPlanLimits()


@dataclass(frozen=True, slots=True, eq=False)
class OrderedFSSKGroupPlan:
    """Decoder and compact-seed lookup metadata for one degree group."""

    total_degree: int
    execution_group: int
    block_indices: tuple[int, ...]
    block_slices: tuple[tuple[int, int], ...]
    width: int
    prefix_indices: np.ndarray | None

    def memory_bytes(self) -> int:
        return 0 if self.prefix_indices is None else int(self.prefix_indices.nbytes)


@dataclass(frozen=True, slots=True, eq=False)
class OrderedFSSKExecutionPlan:
    """Additional ordered FSSK metadata attached to a wordwise layout."""

    fingerprint: tuple[object, ...]
    positive_width: int
    groups: tuple[OrderedFSSKGroupPlan, ...]

    def memory_bytes(self) -> int:
        return sum(group.memory_bytes() for group in self.groups)


_EXECUTION_PLAN_CACHE: OrderedDict[
    tuple[object, ...], OrderedFSSKExecutionPlan
] = OrderedDict()
_EXECUTION_PLAN_LOCK = RLock()


def _check_layout(plan: WordwiseLayoutPlan) -> None:
    if plan.partially_symmetrized:
        raise PallasFSSKPlanError(
            "the ordered scalar-FSSK executor requires an ordered layout plan."
        )
    if not plan.blocks or plan.blocks[0].total_degree != 0:
        raise PallasFSSKPlanError("the layout plan must begin with its scalar block.")
    if plan.blocks[0].flat_offset != 0 or plan.blocks[0].width != 1:
        raise PallasFSSKPlanError("the scalar block must occupy flat coordinate zero.")
    if len(plan.blocks) == 1:
        raise PallasFSSKPlanError(
            "scalar-FSSK execution requires a positive truncation."
        )


def _prefix_metadata_estimate(plan: WordwiseLayoutPlan) -> tuple[int, int]:
    if plan.grading == "total_degree":
        return 0, 0
    elements = sum(
        group.width * group.total_degree
        for group in plan.execution_groups
        if group.total_degree > 0
    )
    # Conservatively cover retained prefix tables and the code/index/sort
    # temporaries used to build the shared lookup for each degree.
    working_bytes = 4 * elements + 32 * max(plan.output_size - 1, 0)
    return elements, working_bytes


def _check_prefix_resources(
        plan: WordwiseLayoutPlan,
        limits: FSSKPlanLimits,
) -> None:
    elements, working_bytes = _prefix_metadata_estimate(plan)
    if elements > min(limits.max_prefix_index_elements, _INT32_MAX):
        raise PallasOrdinaryResourceError(
            "ordered scalar-FSSK prefix metadata requires "
            f"{elements} int32 entries; limit is "
            f"{min(limits.max_prefix_index_elements, _INT32_MAX)}."
        )
    if working_bytes > limits.max_plan_bytes:
        raise PallasOrdinaryResourceError(
            "ordered scalar-FSSK prefix planning requires an estimated "
            f"{working_bytes} bytes; limit is {limits.max_plan_bytes}."
        )
    if plan.output_size - 1 > _INT32_MAX:
        raise PallasOrdinaryResourceError(
            "ordered scalar-FSSK state exceeds signed int32 indexing."
        )


def _bidegree_prefix_lookups(
        plan: WordwiseLayoutPlan,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blocks_by_degree: dict[int, list[Any]] = {}
    for block in plan.blocks:
        if block.total_degree > 0:
            blocks_by_degree.setdefault(block.total_degree, []).append(block)

    lookups = {}
    for degree, blocks in blocks_by_degree.items():
        codes = np.concatenate(
            tuple(np.asarray(block.word_codes, dtype=np.int32) for block in blocks)
        )
        positive_indices = np.concatenate(
            tuple(
                np.arange(
                    block.flat_offset - 1,
                    block.flat_stop - 1,
                    dtype=np.int32,
                )
                for block in blocks
            )
        )
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        if sorted_codes.size > 1 and np.any(sorted_codes[1:] == sorted_codes[:-1]):
            raise PallasFSSKPlanError(
                f"degree-{degree} decoder codes are not unique."
            )
        lookups[degree] = (
            _readonly(sorted_codes),
            _readonly(positive_indices[order]),
        )
    return lookups


def _bidegree_prefix_indices(
        lookups: dict[int, tuple[np.ndarray, np.ndarray]],
        *,
        alphabet_dim: int,
        degree: int,
        target_codes: np.ndarray,
) -> np.ndarray:
    indices = np.empty((target_codes.size, degree), dtype=np.int32)
    for prefix_degree in range(1, degree + 1):
        lookup = lookups.get(prefix_degree)
        if lookup is None:
            raise PallasFSSKPlanError(
                f"the retained word set has no degree-{prefix_degree} prefixes."
            )
        sorted_codes, sorted_indices = lookup
        divisor = alphabet_dim ** (degree - prefix_degree)
        prefix_codes = target_codes // divisor
        positions = np.searchsorted(sorted_codes, prefix_codes)
        valid = positions < sorted_codes.size
        if not np.all(valid):
            raise PallasFSSKPlanError(
                "the retained bidegree word set is not prefix-closed."
            )
        if not np.array_equal(sorted_codes[positions], prefix_codes):
            raise PallasFSSKPlanError(
                "the retained bidegree word set is not prefix-closed."
            )
        indices[:, prefix_degree - 1] = sorted_indices[positions]
    return _readonly(indices)


def build_ordered_fssk_execution_plan(
        plan: WordwiseLayoutPlan,
        *,
        limits: FSSKPlanLimits = DEFAULT_FSSK_PLAN_LIMITS,
) -> OrderedFSSKExecutionPlan:
    """Build bounded prefix lookups without keying on a core object's identity."""
    _check_layout(plan)
    _check_prefix_resources(plan, limits)
    key = plan.fingerprint
    with _EXECUTION_PLAN_LOCK:
        cached = _EXECUTION_PLAN_CACHE.get(key)
        if cached is not None:
            _EXECUTION_PLAN_CACHE.move_to_end(key)
            return cached

    groups = []
    prefix_lookups = (
        _bidegree_prefix_lookups(plan)
        if plan.grading == "bidegree"
        else {}
    )
    for execution_group, group in enumerate(plan.execution_groups):
        if group.total_degree == 0:
            continue
        decoder = group.decoder_codes
        if plan.grading == "bidegree":
            if decoder is None or decoder.dtype != np.dtype(np.int32):
                raise PallasFSSKPlanError(
                    "an ordered bidegree group requires an int32 word decoder."
                )
            prefix_indices = _bidegree_prefix_indices(
                prefix_lookups,
                alphabet_dim=plan.alphabet_dim,
                degree=group.total_degree,
                target_codes=np.asarray(decoder, dtype=np.int32),
            )
        else:
            if decoder is not None:
                raise PallasFSSKPlanError(
                    "a total-degree group must use its analytic word decoder."
                )
            prefix_indices = None
        groups.append(
            OrderedFSSKGroupPlan(
                total_degree=group.total_degree,
                execution_group=execution_group,
                block_indices=group.block_indices,
                block_slices=group.block_slices,
                width=group.width,
                prefix_indices=prefix_indices,
            )
        )

    result = OrderedFSSKExecutionPlan(
        fingerprint=plan.fingerprint,
        positive_width=plan.output_size - 1,
        groups=tuple(groups),
    )
    with _EXECUTION_PLAN_LOCK:
        existing = _EXECUTION_PLAN_CACHE.get(key)
        if existing is not None:
            _EXECUTION_PLAN_CACHE.move_to_end(key)
            return existing
        _EXECUTION_PLAN_CACHE[key] = result
        _EXECUTION_PLAN_CACHE.move_to_end(key)
        while (
            len(_EXECUTION_PLAN_CACHE) > _EXECUTION_PLAN_CACHE_SIZE
            or sum(
                cached.memory_bytes()
                for cached in _EXECUTION_PLAN_CACHE.values()
            ) > _EXECUTION_PLAN_CACHE_BYTES
        ):
            _EXECUTION_PLAN_CACHE.popitem(last=False)
    return result


def _right_matrix(values: Array, matrix: Array) -> Array:
    return jnp.sum(values[..., :, None] * matrix, axis=-2)


def _check_seed_resources(
        *,
        seed_batch: int,
        positive_width: int,
        state_extent: int,
) -> None:
    seed_elements = seed_batch * positive_width * state_extent
    if seed_elements > _INT32_MAX:
        raise PallasOrdinaryResourceError(
            "ordered scalar-FSSK seed contains "
            f"{seed_elements} elements, exceeding the int32 execution guard."
        )


def _check_kernel_resources(
        *,
        flat_batch: int,
        steps: int,
        block_count: int,
        alphabet_dim: int,
        coefficient_steps: int,
        coefficient_order: int,
        state_dim: int,
        degree: int,
        word_count: int,
        padded_word_count: int,
        tile_words: int,
        itemsize: int,
        emit_readout: bool,
) -> None:
    for name, value in (
        ("flat batch size", flat_batch),
        ("step count", steps),
        ("block count", block_count),
        ("alphabet dimension", alphabet_dim),
        ("coefficient step count", coefficient_steps),
        ("coefficient order", coefficient_order),
        ("state dimension", state_dim),
        ("word count", word_count),
        ("padded word count", padded_word_count),
    ):
        if value > _INT32_MAX:
            raise PallasOrdinaryResourceError(
                f"{name} {value} exceeds signed int32 indexing."
            )
    if degree > _MAX_DEGREE:
        raise PallasOrdinaryResourceError(
            f"degree {degree} exceeds the {_MAX_DEGREE}-degree kernel guard."
        )
    if state_dim > _MAX_STATE_DIM:
        raise PallasOrdinaryResourceError(
            f"state dimension {state_dim} exceeds the "
            f"{_MAX_STATE_DIM}-dimensional kernel guard."
        )
    # The coefficient loads materialize E, every active phi matrix, and every
    # psi vector alongside the prefix tile.  Count all of them so a high
    # degree/state-dimension combination is rejected before Triton lowering.
    local_bytes = (
        itemsize
        * (
            degree * tile_words * state_dim
            + 2 * tile_words * state_dim
            + degree * tile_words
            + degree * state_dim * state_dim
            + degree * state_dim
            + (state_dim if emit_readout else 0)
        )
        + np.dtype(np.int32).itemsize * degree * tile_words
    )
    if local_bytes > _MAX_LOCAL_BYTES:
        raise PallasOrdinaryResourceError(
            "ordered scalar-FSSK word tile requires an estimated "
            f"{local_bytes} local bytes, exceeding the "
            f"{_MAX_LOCAL_BYTES}-byte guard; reduce tile_words."
        )
    output_state_width = 1 if emit_readout else state_dim
    output_elements = (
        flat_batch * block_count * output_state_width * padded_word_count
    )
    if output_elements > _INT32_MAX:
        raise PallasOrdinaryResourceError(
            "ordered scalar-FSSK padded output contains "
            f"{output_elements} elements, exceeding the int32 execution guard."
        )


def _load_matrix(
        ref,
        pltriton,
        *,
        batch_index,
        time_index,
        matrix_index: int | None,
        state_dim: int,
) -> Array:
    positions = jnp.arange(state_dim, dtype=jnp.int32)
    rows = positions[:, None]
    columns = positions[None, :]
    if matrix_index is None:
        return pltriton.load(
            ref.at[batch_index, time_index, rows, columns]
        )
    return pltriton.load(
        ref.at[
            batch_index,
            time_index,
            matrix_index,
            rows,
            columns,
        ]
    )


@lru_cache(maxsize=128)
def _ordered_fssk_call(
        shape: tuple[int, int, int],
        dtype: np.dtype,
        coefficient_steps: int,
        coefficient_singletons: tuple[bool, bool, bool],
        state_dim: int,
        degree: int,
        word_count: int,
        padded_word_count: int,
        tile_words: int,
        emission_block_size: int | None,
        accumulate: bool,
        explicit_plan: bool,
        seed_singleton: bool,
        readout_batch: int,
        interpret: bool,
):
    pl, pltriton = _load_pallas()
    flat_batch, steps, alphabet_dim = shape
    block_count = (
        1 if emission_block_size is None else steps // emission_block_size
    )
    if readout_batch and emission_block_size is not None:
        raise AssertionError("readout emission is terminal-only.")
    divisors = tuple(
        alphabet_dim ** power for power in range(degree - 1, -1, -1)
    )
    positive_offsets = tuple(
        sum(alphabet_dim**order for order in range(1, prefix_degree))
        for prefix_degree in range(1, degree + 1)
    )

    def run_kernel(
            y_ref,
            E_ref,
            psi_ref,
            phi_ref,
            initial_ref,
            codes_ref,
            prefix_indices_ref,
            weights_ref,
            output_ref,
    ):
        batch_index = pl.program_id(0)
        E_batch_index = 0 if coefficient_singletons[0] else batch_index
        psi_batch_index = 0 if coefficient_singletons[1] else batch_index
        phi_batch_index = 0 if coefficient_singletons[2] else batch_index
        initial_batch_index = 0 if seed_singleton else batch_index
        word_positions = (
            pl.program_id(1) * tile_words
            + jnp.arange(tile_words, dtype=jnp.int32)
        )
        valid_words = word_positions < word_count
        codes = (
            pltriton.load(codes_ref.at[word_positions])
            if explicit_plan
            else word_positions
        )
        letters = tuple((codes // divisor) % alphabet_dim for divisor in divisors)
        state_positions = jnp.arange(state_dim, dtype=jnp.int32)

        initial_prefixes = []
        for prefix_index in range(degree):
            if explicit_plan:
                compact_indices = pltriton.load(
                    prefix_indices_ref.at[word_positions, prefix_index]
                )
            else:
                prefix_code = codes // divisors[prefix_index]
                compact_indices = positive_offsets[prefix_index] + prefix_code
            initial_prefixes.append(
                pltriton.load(
                    initial_ref.at[
                        initial_batch_index,
                        compact_indices[:, None],
                        state_positions[None, :],
                    ],
                    mask=valid_words[:, None],
                    other=0,
                )
            )
        initial_prefixes = jnp.stack(initial_prefixes, axis=0)

        def time_step(time_index, prefixes):
            coefficient_time = 0 if coefficient_steps == 1 else time_index
            word_increments = tuple(
                pltriton.load(
                    y_ref.at[batch_index, time_index, letter],
                    mask=valid_words,
                    other=0,
                )
                for letter in letters
            )
            E_step = _load_matrix(
                E_ref,
                pltriton,
                batch_index=E_batch_index,
                time_index=coefficient_time,
                matrix_index=None,
                state_dim=state_dim,
            )
            psi_step = tuple(
                pltriton.load(
                    psi_ref.at[
                        psi_batch_index,
                        coefficient_time,
                        prefix_length,
                        state_positions,
                    ]
                )
                for prefix_length in range(degree)
            )
            phi_step = tuple(
                _load_matrix(
                    phi_ref,
                    pltriton,
                    batch_index=phi_batch_index,
                    time_index=coefficient_time,
                    matrix_index=order,
                    state_dim=state_dim,
                )
                for order in range(max(degree - 1, 0))
            )

            for prefix_length in range(degree, 0, -1):
                horner = jnp.broadcast_to(
                    psi_step[prefix_length - 1][None, :],
                    (tile_words, state_dim),
                )
                for prefix_index in range(1, prefix_length):
                    horner = (
                        horner * word_increments[prefix_index - 1][..., None]
                        + _right_matrix(
                            prefixes[prefix_index - 1],
                            phi_step[prefix_length - 1 - prefix_index],
                        )
                    )
                updated = (
                    _right_matrix(prefixes[prefix_length - 1], E_step)
                    + horner * word_increments[prefix_length - 1][..., None]
                )
                prefixes = prefixes.at[prefix_length - 1].set(updated)
            return prefixes

        def store(block_index, prefixes):
            terminal = prefixes[degree - 1]
            if readout_batch:
                weights_batch_index = 0 if readout_batch == 1 else batch_index
                weights = pltriton.load(
                    weights_ref.at[weights_batch_index, state_positions]
                )
                values = jnp.sum(terminal * weights[None, :], axis=-1)
            else:
                values = jnp.moveaxis(terminal, -1, 0)
            if emission_block_size is None:
                if readout_batch:
                    pltriton.store(
                        output_ref.at[batch_index, word_positions],
                        values,
                        mask=valid_words,
                    )
                else:
                    pltriton.store(
                        output_ref.at[
                            batch_index,
                            state_positions[:, None],
                            word_positions[None, :],
                        ],
                        values,
                        mask=valid_words[None, :],
                    )
            else:
                pltriton.store(
                    output_ref.at[
                        batch_index,
                        block_index,
                        state_positions[:, None],
                        word_positions[None, :],
                    ],
                    values,
                    mask=valid_words[None, :],
                )

        if emission_block_size is None:
            terminal = jax.lax.fori_loop(0, steps, time_step, initial_prefixes)
            store(0, terminal)
            return

        def block_step(block_index, current):
            start = block_index * emission_block_size
            block_initial = current if accumulate else initial_prefixes

            def local_time_step(local_index, prefixes):
                return time_step(start + local_index, prefixes)

            terminal = jax.lax.fori_loop(
                0,
                emission_block_size,
                local_time_step,
                block_initial,
            )
            store(block_index, terminal)
            return terminal if accumulate else current

        jax.lax.fori_loop(0, block_count, block_step, initial_prefixes)

    if explicit_plan:
        if readout_batch:
            kernel = run_kernel
        else:
            def kernel(
                    y_ref,
                    E_ref,
                    psi_ref,
                    phi_ref,
                    initial_ref,
                    codes_ref,
                    prefix_indices_ref,
                    output_ref,
            ):
                return run_kernel(
                    y_ref,
                    E_ref,
                    psi_ref,
                    phi_ref,
                    initial_ref,
                    codes_ref,
                    prefix_indices_ref,
                    None,
                    output_ref,
                )
    elif readout_batch:
        def kernel(
                y_ref,
                E_ref,
                psi_ref,
                phi_ref,
                initial_ref,
                weights_ref,
                output_ref,
        ):
            return run_kernel(
                y_ref,
                E_ref,
                psi_ref,
                phi_ref,
                initial_ref,
                None,
                None,
                weights_ref,
                output_ref,
            )
    else:
        def kernel(y_ref, E_ref, psi_ref, phi_ref, initial_ref, output_ref):
            return run_kernel(
                y_ref,
                E_ref,
                psi_ref,
                phi_ref,
                initial_ref,
                None,
                None,
                None,
                output_ref,
            )

    if readout_batch:
        output_shape = (flat_batch, padded_word_count)
    else:
        output_shape = (
            (flat_batch, state_dim, padded_word_count)
            if emission_block_size is None
            else (flat_batch, block_count, state_dim, padded_word_count)
        )
    mode = (
        "readout"
        if readout_batch
        else "terminal"
        if emission_block_size is None
        else "accumulated" if accumulate else "independent"
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
        grid=(flat_batch, padded_word_count // tile_words),
        compiler_params=pltriton.CompilerParams(
            num_warps=_compiler_warps(tile_words),
            num_stages=1,
        ),
        interpret=interpret,
        name=f"tensordev_ordered_fssk_q1_{mode}",
    )


@dataclass(frozen=True, slots=True)
class _CanonicalFSSKInput:
    y: Array
    E: Array
    psi: Array
    phi: Array
    flat_batch: int
    steps: int
    alphabet_dim: int
    coefficient_batches: tuple[int, int, int]
    coefficient_steps: int
    coefficient_order: int
    state_dim: int


def _canonicalize_inputs(y: Any, E: Any, psi: Any, phi: Any) -> _CanonicalFSSKInput:
    y = jnp.asarray(y)
    if y.ndim != 3:
        raise ValueError(
            "y must have canonical shape (flat_batch, steps, alphabet), got "
            f"{y.shape}."
        )
    flat_batch, steps, alphabet_dim = map(int, y.shape)
    if flat_batch <= 0:
        raise PallasOrdinaryUnsupportedError(
            "the Pallas executor requires a positive flat batch size."
        )
    if steps <= 0:
        raise PallasOrdinaryUnsupportedError(
            "the Pallas executor requires at least one increment."
        )
    dtype = np.dtype(y.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise PallasOrdinaryUnsupportedError(
            f"y must have dtype float32 or float64, got {dtype}."
        )

    coefficient_shapes = tuple(
        tuple(value.shape) if hasattr(value, "shape") else np.shape(value)
        for value in (E, psi, phi)
    )
    E_shape, psi_shape, phi_shape = coefficient_shapes
    if len(E_shape) != 4 or E_shape[-1] != E_shape[-2]:
        raise ValueError("E must have shape (batch, coefficient_steps, R, R).")
    state_dim = int(E_shape[-1])
    if state_dim <= 0:
        raise ValueError("E must have a positive state dimension.")
    if state_dim > _MAX_STATE_DIM:
        raise PallasOrdinaryResourceError(
            f"state dimension {state_dim} exceeds the "
            f"{_MAX_STATE_DIM}-dimensional kernel guard."
        )
    if len(psi_shape) != 4 or psi_shape[-1] != state_dim:
        raise ValueError("psi must have shape (batch, coefficient_steps, N, R).")
    coefficient_order = int(psi_shape[-2])
    if len(phi_shape) != 5 or phi_shape[-3:] != (
        max(coefficient_order - 1, 0),
        state_dim,
        state_dim,
    ):
        raise ValueError(
            "phi must have shape (batch, coefficient_steps, N - 1, R, R)."
        )
    coefficient_steps = int(E_shape[1])
    if coefficient_steps not in (1, steps):
        raise ValueError(
            f"coefficient_steps must be 1 or {steps}, got {coefficient_steps}."
        )
    if (
        psi_shape[1] != coefficient_steps
        or phi_shape[1] != coefficient_steps
    ):
        raise ValueError("E, psi, and phi must use the same coefficient step count.")
    for name, shape in zip(("E", "psi", "phi"), coefficient_shapes):
        if shape[0] not in (1, flat_batch):
            raise ValueError(
                f"{name} batch size must be 1 or {flat_batch}, got "
                f"{shape[0]}."
            )
    if coefficient_order <= 0:
        raise ValueError("psi must contain at least one coefficient order.")

    E = jnp.asarray(_colocate_array(E, y), dtype=y.dtype)
    psi = jnp.asarray(_colocate_array(psi, y), dtype=y.dtype)
    phi = jnp.asarray(_colocate_array(phi, y), dtype=y.dtype)
    return _CanonicalFSSKInput(
        y=y,
        E=E,
        psi=psi,
        phi=phi,
        flat_batch=flat_batch,
        steps=steps,
        alphabet_dim=alphabet_dim,
        coefficient_batches=(E.shape[0], psi.shape[0], phi.shape[0]),
        coefficient_steps=coefficient_steps,
        coefficient_order=coefficient_order,
        state_dim=state_dim,
    )


def _pack_initial_state(
        initial_state: Any,
        *,
        plan: WordwiseLayoutPlan,
        flat_batch: int,
        state_dim: int,
        dtype: Any,
        reference: Any = None,
        state_extent: int | None = None,
) -> Array:
    positive_blocks = tuple(block for block in plan.blocks if block.total_degree > 0)
    resource_extent = state_dim if state_extent is None else state_extent
    if initial_state is None:
        _check_seed_resources(
            seed_batch=1,
            positive_width=plan.output_size - 1,
            state_extent=resource_extent,
        )
        return jnp.zeros(
            (1, plan.output_size - 1, state_dim),
            dtype=dtype,
            device=_concrete_single_device(reference),
        )
    if isinstance(initial_state, BigradedTensor):
        if plan.grading != "bidegree":
            raise TypeError("a total-degree plan requires a tuple initial state.")
        if initial_state.spec.include_scalar:
            raise ValueError("initial_state must use first-on format.")
        if initial_state.spec.coordinates != "standard":
            raise ValueError("initial_state must use standard coordinates.")
        if initial_state.spec.partially_symmetrized != plan.partially_symmetrized:
            raise ValueError(
                "initial_state partial-symmetrization disagrees with the "
                "layout plan."
            )
        if initial_state.spec.dims != plan.dims:
            raise ValueError("initial_state dimensions disagree with the layout plan.")
        if initial_state.spec.truncation != plan.truncation:
            raise ValueError("initial_state truncation disagrees with the layout plan.")
        blocks = initial_state.blocks
    else:
        if plan.grading != "total_degree":
            raise TypeError("a bidegree plan requires a BigradedTensor initial state.")
        blocks = tuple(initial_state)
    if len(blocks) != len(positive_blocks):
        raise ValueError(
            f"initial_state must contain {len(positive_blocks)} blocks, got "
            f"{len(blocks)}."
        )

    sources = []
    seed_batch = 1
    for value, block in zip(blocks, positive_blocks):
        shape = tuple(value.shape) if hasattr(value, "shape") else np.shape(value)
        tail = (1, 1, state_dim, block.width)
        unbatched = shape == tail
        canonical_shape = (1,) + shape if unbatched else shape
        if (
            len(canonical_shape) != 5
            or canonical_shape[1:] != tail
            or canonical_shape[0] not in (1, flat_batch)
        ):
            raise ValueError(
                f"initial_state block {block.grade!r} must have canonical shape "
                f"(1 or {flat_batch},) + {tail}, got {shape}."
            )
        seed_batch = max(seed_batch, int(canonical_shape[0]))
        sources.append((value, unbatched))

    _check_seed_resources(
        seed_batch=seed_batch,
        positive_width=plan.output_size - 1,
        state_extent=resource_extent,
    )
    packed = []
    for value, unbatched in sources:
        value = jnp.asarray(_colocate_array(value, reference), dtype=dtype)
        if unbatched:
            value = value[None]
        value = jnp.broadcast_to(value, (seed_batch,) + value.shape[1:])
        packed.append(jnp.moveaxis(value[:, 0, 0], -1, -2))
    result = jnp.concatenate(packed, axis=1)
    if result.shape != (seed_batch, plan.output_size - 1, state_dim):
        raise AssertionError("packed scalar-FSSK initial state has the wrong shape.")
    return result


def _fssk_state_extent(state_dim: int) -> int:
    state_extent = 1 << (state_dim - 1).bit_length()
    if state_extent > _MAX_STATE_DIM:
        raise PallasOrdinaryResourceError(
            f"state dimension {state_dim} requires padded extent "
            f"{state_extent}, exceeding the {_MAX_STATE_DIM}-dimensional "
            "kernel guard."
        )
    return state_extent


def _pad_fssk_state_operands(
        canonical: _CanonicalFSSKInput,
        initial: Array,
        *,
        state_extent: int,
) -> tuple[Array, Array, Array, Array]:
    """Pad state axes to the power-of-two extent used by Triton vectors."""
    state_padding = state_extent - canonical.state_dim
    if state_padding:
        E_operand = jnp.pad(
            canonical.E,
            ((0, 0), (0, 0), (0, state_padding), (0, state_padding)),
        )
        psi_operand = jnp.pad(
            canonical.psi,
            ((0, 0), (0, 0), (0, 0), (0, state_padding)),
        )
        phi_operand = jnp.pad(
            canonical.phi,
            (
                (0, 0),
                (0, 0),
                (0, 0),
                (0, state_padding),
                (0, state_padding),
            ),
        )
        initial = jnp.pad(initial, ((0, 0), (0, 0), (0, state_padding)))
    else:
        E_operand = canonical.E
        psi_operand = canonical.psi
        phi_operand = canonical.phi

    # Zero-length Pallas operands are not portable across supported JAX lines.
    if phi_operand.shape[2] == 0:
        phi_operand = jnp.zeros(
            (
                canonical.phi.shape[0],
                canonical.coefficient_steps,
                1,
                state_extent,
                state_extent,
            ),
            dtype=canonical.y.dtype,
            device=_concrete_single_device(canonical.y),
        )
    return E_operand, psi_operand, phi_operand, initial


def _canonicalize_readout_weights(
        weights: Any,
        canonical: _CanonicalFSSKInput,
        *,
        state_extent: int,
) -> tuple[Array, int]:
    shape = tuple(weights.shape) if hasattr(weights, "shape") else np.shape(weights)
    if len(shape) != 2 or shape[-1] != canonical.state_dim:
        raise ValueError(
            "readout weights must have shape (batch, R), got "
            f"{shape}."
        )
    weight_batch = int(shape[0])
    if weight_batch not in (1, canonical.flat_batch):
        raise ValueError(
            "readout weight batch size must be 1 or "
            f"{canonical.flat_batch}, got {weight_batch}."
        )
    operand = jnp.asarray(
        _colocate_array(weights, canonical.y),
        dtype=canonical.y.dtype,
    )
    if state_extent != canonical.state_dim:
        operand = jnp.pad(
            operand,
            ((0, 0), (0, state_extent - canonical.state_dim)),
        )
    return operand, 1 if weight_batch == 1 else 2


def _assemble_readout_signature(
        plan: WordwiseLayoutPlan,
        blocks: list[Any],
        canonical: _CanonicalFSSKInput,
):
    unit = jnp.ones(
        (canonical.flat_batch, 1),
        dtype=canonical.y.dtype,
        device=_concrete_single_device(canonical.y),
    )
    return plan.assemble_signature((unit, *blocks))


def _ordered_fssk_q1_execute(
        y: Any,
        E: Any,
        psi: Any,
        phi: Any,
        *,
        plan: WordwiseLayoutPlan,
        initial_state: Any = None,
        block_size: int | None = None,
        accumulate: bool = True,
        tile_words: int = 32,
        interpret: bool = False,
        limits: FSSKPlanLimits = DEFAULT_FSSK_PLAN_LIMITS,
        readout_weights: Any = None,
):
    """Execute the shared ordered scalar-FSSK recurrence with Pallas.

    Numerical inputs are canonical: ``y`` is ``(flat_batch, steps, m)``;
    ``E``, ``psi``, and ``phi`` start with ``(batch, coefficient_steps)``,
    where each batch is one or ``flat_batch``; and ``coefficient_steps`` is
    either one or the path step count.  An unbatched initial state likewise
    remains a singleton kernel operand.  The returned first-on tensor retains
    the flat batch axis and inserts the existing singleton FSSK axes before
    ``(R, coordinates)``.
    """
    _check_layout(plan)
    canonical = _canonicalize_inputs(y, E, psi, phi)
    emit_readout = readout_weights is not None
    if canonical.alphabet_dim != plan.alphabet_dim:
        raise ValueError(
            f"y alphabet width {canonical.alphabet_dim} disagrees with plan "
            f"dimension {plan.alphabet_dim}."
        )
    maximum_degree = max(block.total_degree for block in plan.blocks)
    if maximum_degree > canonical.coefficient_order:
        raise ValueError(
            f"coefficients cover order {canonical.coefficient_order}, but the "
            f"layout requires order {maximum_degree}."
        )
    if not isinstance(accumulate, bool):
        raise TypeError("accumulate must be a bool.")
    if not isinstance(interpret, bool):
        raise TypeError("interpret must be a bool.")
    tile_words = _require_integer(tile_words, name="tile_words", minimum=1)
    if tile_words > _MAX_TILE_WORDS or tile_words & (tile_words - 1):
        raise PallasOrdinaryResourceError(
            "tile_words must be a power of two no greater than "
            f"{_MAX_TILE_WORDS}, got {tile_words}."
        )

    normalized_block_size, block_count = _normalize_block_size(
        block_size,
        steps=canonical.steps,
    )
    if emit_readout and block_count != 1:
        raise PallasOrdinaryUnsupportedError(
            "scalar-FSSK readout emission is terminal-only."
        )
    state_extent = _fssk_state_extent(canonical.state_dim)
    execution = build_ordered_fssk_execution_plan(plan, limits=limits)
    group_resources = []
    for group in execution.groups:
        effective_tile = _effective_tile_words(group.width, tile_words)
        padded_width = _padded_word_count(group.width, effective_tile)
        _check_kernel_resources(
            flat_batch=canonical.flat_batch,
            steps=canonical.steps,
            block_count=block_count,
            alphabet_dim=canonical.alphabet_dim,
            coefficient_steps=canonical.coefficient_steps,
            coefficient_order=canonical.coefficient_order,
            state_dim=state_extent,
            degree=group.total_degree,
            word_count=group.width,
            padded_word_count=padded_width,
            tile_words=effective_tile,
            itemsize=np.dtype(canonical.y.dtype).itemsize,
            emit_readout=emit_readout,
        )
        group_resources.append((group, effective_tile, padded_width))

    initial = _pack_initial_state(
        initial_state,
        plan=plan,
        flat_batch=canonical.flat_batch,
        state_dim=canonical.state_dim,
        dtype=canonical.y.dtype,
        reference=canonical.y,
        state_extent=state_extent,
    )
    E_operand, psi_operand, phi_operand, initial = _pad_fssk_state_operands(
        canonical,
        initial,
        state_extent=state_extent,
    )
    if emit_readout:
        weights_operand, readout_batch = _canonicalize_readout_weights(
            readout_weights,
            canonical,
            state_extent=state_extent,
        )
    else:
        weights_operand, readout_batch = None, 0

    output_blocks: list[Any | None] = [None] * (len(plan.blocks) - 1)
    for group, effective_tile, padded_width in group_resources:
        decoder_codes = plan.execution_groups[group.execution_group].decoder_codes
        explicit_plan = decoder_codes is not None
        operands = (
            canonical.y,
            E_operand,
            psi_operand,
            phi_operand,
            initial,
        )
        if explicit_plan:
            codes = decoder_codes
            prefix_indices = group.prefix_indices
            if padded_width != group.width:
                pad = padded_width - group.width
                codes = np.pad(codes, (0, pad))
                prefix_indices = np.pad(
                    prefix_indices,
                    ((0, pad), (0, 0)),
                )
            codes = _colocate_array(codes, canonical.y)
            prefix_indices = _colocate_array(prefix_indices, canonical.y)
        call = _ordered_fssk_call(
            (
                canonical.flat_batch,
                canonical.steps,
                canonical.alphabet_dim,
            ),
            np.dtype(canonical.y.dtype),
            canonical.coefficient_steps,
            tuple(batch == 1 for batch in canonical.coefficient_batches),
            state_extent,
            group.total_degree,
            group.width,
            padded_width,
            effective_tile,
            None if block_count == 1 else normalized_block_size,
            True if block_count == 1 else accumulate,
            explicit_plan,
            initial.shape[0] == 1,
            readout_batch,
            interpret,
        )
        if explicit_plan:
            values = (
                call(*operands, codes, prefix_indices, weights_operand)
                if emit_readout
                else call(*operands, codes, prefix_indices)
            )
        else:
            values = (
                call(*operands, weights_operand)
                if emit_readout
                else call(*operands)
            )
        if emit_readout:
            values = values[..., :group.width]
        else:
            values = values[..., :canonical.state_dim, :group.width]
            values = (
                values[:, None, None, :, :]
                if values.ndim == 3
                else values[:, :, None, None, :, :]
            )
        for block_index, (start, stop) in zip(
            group.block_indices,
            group.block_slices,
        ):
            output_blocks[block_index - 1] = values[..., start:stop]

    if any(block is None for block in output_blocks):
        raise AssertionError("ordered scalar-FSSK execution missed a state block.")
    if emit_readout:
        return _assemble_readout_signature(plan, output_blocks, canonical)
    return plan.assemble_first_on(output_blocks)


def ordered_fssk_q1_pallas(
        y: Any,
        E: Any,
        psi: Any,
        phi: Any,
        *,
        plan: WordwiseLayoutPlan,
        initial_state: Any = None,
        block_size: int | None = None,
        accumulate: bool = True,
        tile_words: int = 32,
        interpret: bool = False,
        limits: FSSKPlanLimits = DEFAULT_FSSK_PLAN_LIMITS,
):
    """Evolve all ordered scalar-FSSK state coordinates with Pallas."""
    return _ordered_fssk_q1_execute(
        y,
        E,
        psi,
        phi,
        plan=plan,
        initial_state=initial_state,
        block_size=block_size,
        accumulate=accumulate,
        tile_words=tile_words,
        interpret=interpret,
        limits=limits,
    )


def ordered_fssk_q1_readout_pallas(
        y: Any,
        E: Any,
        psi: Any,
        phi: Any,
        weights: Any,
        *,
        plan: WordwiseLayoutPlan,
        initial_state: Any = None,
        tile_words: int = 32,
        interpret: bool = False,
        limits: FSSKPlanLimits = DEFAULT_FSSK_PLAN_LIMITS,
):
    """Emit the terminal q=1 readout directly from the ordered recurrence."""
    return _ordered_fssk_q1_execute(
        y,
        E,
        psi,
        phi,
        plan=plan,
        initial_state=initial_state,
        tile_words=tile_words,
        interpret=interpret,
        limits=limits,
        readout_weights=weights,
    )


def clear_pallas_fssk_q1_cache() -> None:
    """Clear the bounded host-plan and Pallas-call caches."""
    _ordered_fssk_call.cache_clear()
    with _EXECUTION_PLAN_LOCK:
        _EXECUTION_PLAN_CACHE.clear()


__all__ = [
    "DEFAULT_FSSK_PLAN_LIMITS",
    "FSSKPlanLimits",
    "OrderedFSSKExecutionPlan",
    "OrderedFSSKGroupPlan",
    "PallasFSSKPlanError",
    "build_ordered_fssk_execution_plan",
    "clear_pallas_fssk_q1_cache",
    "ordered_fssk_q1_pallas",
    "ordered_fssk_q1_readout_pallas",
]
