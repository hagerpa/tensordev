"""Immutable, standard-coordinate layout plans for wordwise execution."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from math import comb
from numbers import Integral
from threading import RLock
from typing import Any, Literal, Sequence

import numpy as np

from tensordev._wordwise.plans import PrefixGraphPlan, build_prefix_graph_plan
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
)
from tensordev.core.bigraded.types import BigradedSpec, BigradedTensor
from tensordev.core.utils.precompute import _readonly


Grade = int | tuple[int, int]
Grading = Literal["total_degree", "bidegree"]
_INT32_MAX = int(np.iinfo(np.int32).max)


class PlanResourceError(RuntimeError):
    """Raised when a wordwise plan exceeds a deterministic resource guard."""

    def __init__(self, message: str, *, estimate: "PlanEstimate") -> None:
        super().__init__(message)
        self.estimate = estimate


@dataclass(frozen=True, slots=True)
class PlanLimits:
    """Host-plan limits checked before any decoder or graph allocation."""

    max_output_coordinates: int = _INT32_MAX
    max_metadata_elements: int = 8_000_000
    max_metadata_bytes: int = 64 * 1024**2
    max_blocks: int = 4_096
    max_prefix_graphs: int = 100_000
    max_prefix_nodes: int = 8_192
    max_prefix_edges: int = 65_536

    def __post_init__(self) -> None:
        for name in (
            "max_output_coordinates",
            "max_metadata_elements",
            "max_metadata_bytes",
            "max_blocks",
            "max_prefix_graphs",
            "max_prefix_nodes",
            "max_prefix_edges",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a positive integer.")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")


DEFAULT_PLAN_LIMITS = PlanLimits()


@dataclass(frozen=True, slots=True)
class PlanEstimate:
    """Closed-form structural size of one active wordwise plan."""

    grading: Grading
    dims: tuple[int, ...]
    truncation: int | tuple[int, int]
    partially_symmetrized: bool
    output_coordinates: int
    metadata_elements: int
    metadata_bytes: int
    block_count: int
    prefix_graphs: int = 0
    prefix_nodes: int = 0
    prefix_edges: int = 0
    max_prefix_nodes: int = 0
    max_prefix_edges: int = 0
    maximum_word_code: int = 0
    exact: bool = True


@dataclass(frozen=True, slots=True, eq=False)
class WordwiseBlockPlan:
    """Output and decoder metadata for one homogeneous block."""

    grade: Grade
    total_degree: int
    width: int
    flat_offset: int
    dense_prime_width: int
    execution_group: int = 0
    group_offset: int = 0
    word_codes: np.ndarray | None = None
    prefix_plan: PrefixGraphPlan | None = None

    @property
    def flat_stop(self) -> int:
        return self.flat_offset + self.width

    @property
    def partially_symmetrized(self) -> bool:
        return self.prefix_plan is not None

    def memory_bytes(self) -> int:
        return (
            0 if self.word_codes is None else int(self.word_codes.nbytes)
        ) + (0 if self.prefix_plan is None else self.prefix_plan.memory_bytes())


@dataclass(frozen=True, slots=True, eq=False)
class WordwiseExecutionGroup:
    """Blocks of one total degree sharing one kernel launch."""

    total_degree: int
    block_indices: tuple[int, ...]
    block_slices: tuple[tuple[int, int], ...]
    width: int
    decoder_codes: np.ndarray | None = None


@dataclass(frozen=True, slots=True, eq=False)
class WordwiseLayoutPlan:
    """One active standard-coordinate output layout.

    Coordinates deliberately do not occur in this object or its fingerprint.
    A shear core uses this same plan and transforms the completed standard
    tensor at the operation boundary.
    """

    grading: Grading
    dims: tuple[int, ...]
    truncation: int | tuple[int, int]
    partially_symmetrized: bool
    blocks: tuple[WordwiseBlockPlan, ...]
    execution_groups: tuple[WordwiseExecutionGroup, ...]
    estimate: PlanEstimate

    @property
    def alphabet_dim(self) -> int:
        return sum(self.dims)

    @property
    def grades(self) -> tuple[Grade, ...]:
        return tuple(block.grade for block in self.blocks)

    @property
    def output_size(self) -> int:
        return self.estimate.output_coordinates

    def memory_bytes(self) -> int:
        """Retained NumPy payload, excluding referenced Python objects."""
        return sum(block.memory_bytes() for block in self.blocks)

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.grading,
            self.dims,
            self.truncation,
            self.partially_symmetrized,
        )

    def block(self, grade: Grade) -> WordwiseBlockPlan:
        for block in self.blocks:
            if block.grade == grade:
                return block
        raise KeyError(f"grade {grade!r} is not present in this plan.")

    def assemble_signature(self, blocks: Sequence[Any]):
        """Assemble all planned blocks in standard coordinates."""
        return self._assemble(blocks, include_scalar=True)

    def assemble_first_on(self, blocks: Sequence[Any]):
        """Assemble non-scalar blocks as a first-on tensor."""
        return self._assemble(blocks, include_scalar=False)

    def _assemble(self, blocks: Sequence[Any], *, include_scalar: bool):
        blocks = tuple(blocks)
        expected_plans = tuple(
            block
            for block in self.blocks
            if include_scalar or block.total_degree > 0
        )
        if len(blocks) != len(expected_plans):
            raise ValueError(
                f"expected {len(expected_plans)} blocks, got {len(blocks)}."
            )
        for value, block_plan in zip(blocks, expected_plans):
            if getattr(value, "ndim", 0) == 0 or value.shape[-1] != block_plan.width:
                width = None if getattr(value, "ndim", 0) == 0 else value.shape[-1]
                raise ValueError(
                    f"block {block_plan.grade!r} has width {width}, expected "
                    f"{block_plan.width}."
                )
        if self.grading == "total_degree":
            return blocks
        spec = BigradedSpec(
            self.dims[0],
            self.dims[1],
            self.truncation,
            coordinates="standard",
            include_scalar=include_scalar,
            partially_symmetrized=self.partially_symmetrized,
        )
        return BigradedTensor(blocks, spec)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _bidegree(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a pair of non-negative integers.")
    return (
        _non_negative_int(value[0], name=f"{name}[0]"),
        _non_negative_int(value[1], name=f"{name}[1]"),
    )


def _balanced_product(total: int, parts: int) -> int:
    """Maximum of ``prod(a_i + 1)`` over ``sum(a_i) == total``."""
    quotient, remainder = divmod(total, parts)
    return (quotient + 2) ** remainder * (quotient + 1) ** (parts - remainder)


def _power_capped(base: int, exponent: int, cap: int) -> int:
    """Return ``base**exponent`` saturated at ``cap`` in logarithmic time."""
    result = 1
    factor = min(base, cap)
    power = exponent
    while power:
        if power & 1:
            result *= factor
            if result >= cap:
                return cap
        power >>= 1
        if power:
            factor *= factor
            if factor >= cap:
                factor = cap
    return result


def _total_width_capped(dimension: int, truncation: int, cap: int) -> int:
    if dimension == 1:
        return min(truncation + 1, cap)
    numerator = _power_capped(dimension, truncation + 1, cap * (dimension - 1))
    if numerator >= cap * (dimension - 1):
        return cap
    return min((numerator - 1) // (dimension - 1), cap)


def estimate_layout_plan(
    *,
    grading: Grading,
    dims: int | tuple[int, int],
    truncation: int | tuple[int, int],
    partially_symmetrized: bool = False,
) -> PlanEstimate:
    """Compute plan sizes without enumerating words, ranks, or prefixes."""

    if grading == "total_degree":
        if partially_symmetrized:
            raise ValueError("total-degree partially symmetrized layouts do not exist.")
        dimension = _positive_int(dims, name="dims")
        trunc = _non_negative_int(truncation, name="truncation")
        output = _total_width_capped(dimension, trunc, _INT32_MAX + 1)
        maximum_word_code = max(
            _power_capped(dimension, trunc, _INT32_MAX + 2) - 1,
            0,
        )
        return PlanEstimate(
            grading=grading,
            dims=(dimension,),
            truncation=trunc,
            partially_symmetrized=False,
            output_coordinates=output,
            metadata_elements=0,
            metadata_bytes=0,
            block_count=trunc + 1,
            maximum_word_code=maximum_word_code,
            exact=(
                output <= _INT32_MAX
                and maximum_word_code <= _INT32_MAX
            ),
        )

    if grading != "bidegree":
        raise ValueError(
            "grading must be either 'total_degree' or 'bidegree', "
            f"got {grading!r}."
        )
    split = _bidegree(dims, name="dims")
    if split[0] <= 0 or split[1] <= 0:
        raise ValueError(f"dims must be strictly positive, got {split}.")
    trunc = _bidegree(truncation, name="truncation")
    spec = BigradedSpec(
        *split,
        trunc,
        partially_symmetrized=partially_symmetrized,
    )
    dimension = sum(split)
    grade_count_closed = (trunc[0] + 1) * (trunc[1] + 1)
    # A plan with this many Python blocks is ineligible under every supported
    # execution profile.  Return conservative lower bounds without walking an
    # enormous grade rectangle; ``check_plan_resources`` rejects on blocks.
    if grade_count_closed > 100_000:
        return PlanEstimate(
            grading=grading,
            dims=split,
            truncation=trunc,
            partially_symmetrized=partially_symmetrized,
            output_coordinates=min(grade_count_closed, _INT32_MAX + 1),
            metadata_elements=min(grade_count_closed, _INT32_MAX + 1),
            metadata_bytes=min(4 * grade_count_closed, 4 * (_INT32_MAX + 1)),
            block_count=grade_count_closed,
            prefix_graphs=(
                min(grade_count_closed, _INT32_MAX + 1)
                if partially_symmetrized
                else 0
            ),
            maximum_word_code=max(
                _power_capped(
                    split[0] if partially_symmetrized else dimension,
                    trunc[0] if partially_symmetrized else sum(trunc),
                    _INT32_MAX + 2,
                )
                - 1,
                0,
            ),
            exact=False,
        )
    output = 0
    graph_count = 0
    total_nodes = 0
    total_edges = 0
    max_nodes = 0
    max_edges = 0
    decoder_elements = 0
    grade_count = 0
    for n, m in spec.grades:
        grade_count += 1
        output += spec.block_width((n, m))
        if not partially_symmetrized:
            decoder_elements += spec.block_width((n, m))
            continue

        rank_count = multiset_placement_count(split[1], (n, m))
        parts = (n + 1) * split[1]
        # Sum of the manuscript's prefix-set cardinality over every weak
        # composition of m into ``parts`` entries.
        nodes = (n + 1) * comb(m + parts + split[1] - 1, m)
        doubleprime_edges = (
            0
            if m == 0
            else (n + 1)
            * split[1]
            * comb(m + parts + split[1] - 2, m - 1)
        )
        edges = n * rank_count + doubleprime_edges
        graph_count += rank_count
        total_nodes += nodes
        total_edges += edges
        node_bound = (n + 1) * _balanced_product(m, split[1])
        edge_bound = split[1] * node_bound + n
        max_nodes = max(max_nodes, node_bound)
        max_edges = max(max_edges, edge_bound)

    if partially_symmetrized:
        # Four int32 node fields (including incoming-edge CSR offsets), four
        # int32 edge fields, two graph-level CSR offset arrays, and one
        # terminal index per graph.  Python object overhead is excluded.
        metadata_elements = (
            4 * total_nodes
            + 4 * total_edges
            + 4 * graph_count
            + 2 * grade_count
        )
        metadata_bytes = 4 * metadata_elements
    else:
        metadata_elements = decoder_elements
        metadata_bytes = 4 * decoder_elements
    return PlanEstimate(
        grading=grading,
        dims=split,
        truncation=trunc,
        partially_symmetrized=partially_symmetrized,
        output_coordinates=output,
        metadata_elements=metadata_elements,
        metadata_bytes=metadata_bytes,
        block_count=grade_count,
        prefix_graphs=graph_count,
        prefix_nodes=total_nodes,
        prefix_edges=total_edges,
        max_prefix_nodes=max_nodes,
        max_prefix_edges=max_edges,
        maximum_word_code=max(
            _power_capped(
                split[0] if partially_symmetrized else dimension,
                trunc[0] if partially_symmetrized else sum(trunc),
                _INT32_MAX + 2,
            )
            - 1,
            0,
        ),
    )


def check_plan_resources(
    estimate: PlanEstimate,
    limits: PlanLimits = DEFAULT_PLAN_LIMITS,
) -> None:
    """Reject an unsafe plan from closed-form counts alone."""

    checks = (
        (
            estimate.output_coordinates,
            min(limits.max_output_coordinates, _INT32_MAX),
            "output coordinates",
        ),
        (estimate.metadata_elements, limits.max_metadata_elements, "metadata elements"),
        (estimate.metadata_bytes, limits.max_metadata_bytes, "metadata bytes"),
        (estimate.block_count, limits.max_blocks, "homogeneous blocks"),
        (estimate.prefix_graphs, limits.max_prefix_graphs, "prefix graphs"),
        (estimate.max_prefix_nodes, limits.max_prefix_nodes, "nodes in one graph"),
        (estimate.max_prefix_edges, limits.max_prefix_edges, "edges in one graph"),
        (estimate.prefix_nodes, _INT32_MAX, "packed prefix nodes"),
        (estimate.prefix_edges, _INT32_MAX, "packed prefix edges"),
        (estimate.maximum_word_code, _INT32_MAX, "maximum word code"),
    )
    for actual, maximum, label in checks:
        if actual > maximum:
            raise PlanResourceError(
                f"wordwise plan requires {actual} {label}; limit is {maximum}.",
                estimate=estimate,
            )


_PLAN_CACHE_MAXSIZE = 32
_PLAN_CACHE_MAX_BYTES = 128 * 1024**2
_PLAN_CACHE: OrderedDict[tuple[object, ...], WordwiseLayoutPlan] = OrderedDict()
_PLAN_CACHE_LOCK = RLock()
_PLAN_CACHE_BYTES = 0


def clear_layout_plan_cache() -> None:
    global _PLAN_CACHE_BYTES
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE.clear()
        _PLAN_CACHE_BYTES = 0


def _resolve_structure(
    core: Any,
    truncation: object,
    alphabet_dim: int | None,
) -> tuple[Grading, tuple[int, ...], int | tuple[int, int], bool]:
    grading = getattr(core, "grading", None)
    if grading not in {"total_degree", "bidegree"}:
        raise TypeError(f"unsupported core grading {grading!r}.")
    active = core.normalize_truncation(truncation)
    partially_symmetrized = bool(
        getattr(core, "partially_symmetrized", False)
    )
    if grading == "total_degree":
        configured = getattr(core, "d", None)
        if alphabet_dim is None:
            if configured is None:
                raise ValueError(
                    "alphabet_dim is required for a dimension-free total-degree core."
                )
            alphabet_dim = configured
        dimension = _positive_int(alphabet_dim, name="alphabet_dim")
        if configured is not None and int(configured) != dimension:
            raise ValueError(
                f"alphabet_dim={dimension} disagrees with core dimension {configured}."
            )
        return grading, (dimension,), int(active), partially_symmetrized

    dims = tuple(getattr(core, "dims", ()))
    split = _bidegree(dims, name="core.dims")
    if alphabet_dim is not None and int(alphabet_dim) != sum(split):
        raise ValueError(
            f"alphabet_dim={alphabet_dim} disagrees with core dimensions {split}."
        )
    return grading, split, tuple(active), partially_symmetrized


def build_layout_plan(
    core: Any,
    truncation: object = None,
    *,
    alphabet_dim: int | None = None,
    limits: PlanLimits = DEFAULT_PLAN_LIMITS,
) -> WordwiseLayoutPlan:
    """Return a bounded plan for the core's active standard layout.

    Construction is lazy and cached by structural content.  Neither core
    identity, coordinate system, shuffle scope, nor inactive capacity enters
    the cache key.
    """

    grading, dims, active, partially_symmetrized = _resolve_structure(
        core,
        truncation,
        alphabet_dim,
    )
    estimate = estimate_layout_plan(
        grading=grading,
        dims=dims[0] if grading == "total_degree" else dims,
        truncation=active,
        partially_symmetrized=partially_symmetrized,
    )
    check_plan_resources(estimate, limits)
    key = grading, dims, active, partially_symmetrized
    global _PLAN_CACHE_BYTES
    with _PLAN_CACHE_LOCK:
        cached = _PLAN_CACHE.get(key)
        if cached is not None:
            _PLAN_CACHE.move_to_end(key)
            return cached

    blocks = []
    offset = 0
    if grading == "total_degree":
        dimension = dims[0]
        for degree in range(int(active) + 1):
            width = dimension**degree
            blocks.append(
                WordwiseBlockPlan(
                    grade=degree,
                    total_degree=degree,
                    width=width,
                    flat_offset=offset,
                    dense_prime_width=width,
                )
            )
            offset += width
    else:
        spec = BigradedSpec(
            dims[0],
            dims[1],
            active,
            partially_symmetrized=partially_symmetrized,
        )
        store = getattr(core, "plan_store", None)
        if store is None:
            raise TypeError("a bidegree core must expose its plan_store.")
        for grade in spec.grades:
            n, _m = grade
            source = store.grade_plan(grade)
            width = spec.block_width(grade)
            if source.block_width != width:
                raise ValueError(
                    f"core plan width at grade {grade} is inconsistent."
                )
            if partially_symmetrized:
                if source.placements.shape[0] != spec.rank_count(grade):
                    raise ValueError(
                        f"core rank plan at grade {grade} is inconsistent."
                    )
                prefix_plan = build_prefix_graph_plan(
                    source.placements,
                    grade=grade,
                    d_prime=dims[0],
                )
                word_codes = None
            else:
                word_codes = source.block_to_total_indices
                if word_codes.shape != (width,):
                    raise ValueError(
                        f"core decoder at grade {grade} has shape "
                        f"{word_codes.shape}, expected {(width,)}."
                    )
                if word_codes.size and int(word_codes.max()) > _INT32_MAX:
                    raise PlanResourceError(
                        "ordered bidegree decoder exceeds signed int32 range.",
                        estimate=estimate,
                    )
                prefix_plan = None
            blocks.append(
                WordwiseBlockPlan(
                    grade=grade,
                    total_degree=sum(grade),
                    width=width,
                    flat_offset=offset,
                    dense_prime_width=dims[0] ** n,
                    word_codes=word_codes,
                    prefix_plan=prefix_plan,
                )
            )
            offset += width

    if offset != estimate.output_coordinates:
        raise AssertionError("layout estimate and constructed width disagree.")

    # Bidegrees of the same total order are contiguous in the canonical
    # layout.  Ordered decoders are packed once per total degree so the
    # executor can issue one launch without either one operand per grade or a
    # second device-side concatenation.  Block decoders become zero-copy views
    # into this authoritative wordwise buffer.
    grouped_indices: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        grouped_indices.setdefault(block.total_degree, []).append(index)
    execution_groups = []
    for group_index, (degree, indices_list) in enumerate(grouped_indices.items()):
        indices = tuple(indices_list)
        starts = []
        group_width = 0
        for index in indices:
            starts.append(group_width)
            group_width += blocks[index].width
        slices = tuple(
            (start, start + blocks[index].width)
            for start, index in zip(starts, indices)
        )
        if grading == "bidegree" and not partially_symmetrized:
            decoder = _readonly(
                np.concatenate(
                    tuple(
                        np.asarray(blocks[index].word_codes, dtype=np.int32)
                        for index in indices
                    )
                )
            )
            for index, (start, stop) in zip(indices, slices):
                blocks[index] = replace(
                    blocks[index],
                    execution_group=group_index,
                    group_offset=start,
                    word_codes=decoder[start:stop],
                )
        else:
            decoder = None
            for index, (start, _stop) in zip(indices, slices):
                blocks[index] = replace(
                    blocks[index],
                    execution_group=group_index,
                    group_offset=start,
                )
        execution_groups.append(
            WordwiseExecutionGroup(
                total_degree=degree,
                block_indices=indices,
                block_slices=slices,
                width=group_width,
                decoder_codes=decoder,
            )
        )
    result = WordwiseLayoutPlan(
        grading=grading,
        dims=dims,
        truncation=active,
        partially_symmetrized=partially_symmetrized,
        blocks=tuple(blocks),
        execution_groups=tuple(execution_groups),
        estimate=estimate,
    )
    with _PLAN_CACHE_LOCK:
        existing = _PLAN_CACHE.get(key)
        if existing is not None:
            _PLAN_CACHE.move_to_end(key)
            return existing
        _PLAN_CACHE[key] = result
        _PLAN_CACHE_BYTES += result.memory_bytes()
        _PLAN_CACHE.move_to_end(key)
        while (
            len(_PLAN_CACHE) > _PLAN_CACHE_MAXSIZE
            or _PLAN_CACHE_BYTES > _PLAN_CACHE_MAX_BYTES
        ):
            _evicted_key, evicted = _PLAN_CACHE.popitem(last=False)
            _PLAN_CACHE_BYTES -= evicted.memory_bytes()
    return result


__all__ = [
    "DEFAULT_PLAN_LIMITS",
    "PlanEstimate",
    "PlanLimits",
    "PlanResourceError",
    "WordwiseBlockPlan",
    "WordwiseExecutionGroup",
    "WordwiseLayoutPlan",
    "build_layout_plan",
    "check_plan_resources",
    "clear_layout_plan_cache",
    "estimate_layout_plan",
]
