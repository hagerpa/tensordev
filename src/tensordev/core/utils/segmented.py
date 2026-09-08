"""Shared immutable rank-accumulation plans for many-to-one block kernels.

Layout-specific builders supply the target maps.  This module owns
the common runtime shape contract: an already-vectorized source-pair axis is
accumulated into an output-rank axis through a static map.  Concrete backends
provide the functional scatter-add primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _readonly_indices(values, *, maximum: int) -> np.ndarray:
    """Return a compact immutable unsigned index array."""
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if maximum <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif maximum <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif maximum <= np.iinfo(np.uint32).max:
        dtype = np.uint32
    elif maximum <= np.iinfo(np.uint64).max:
        dtype = np.uint64
    else:
        raise OverflowError(f"index maximum {maximum} exceeds uint64")
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class SegmentedRankPlan:
    """Static many-to-one map from a source-pair axis to output ranks.

    ``target_ranks[j]`` is the output rank receiving source row ``j``.  The
    compact plan is shared by concatenation and shuffle kernels while allowing
    each backend to supply its own accumulation primitive.
    """

    target_ranks: np.ndarray
    source_count: int
    output_rank_count: int

    @classmethod
    def from_targets(
        cls,
        target_ranks,
        *,
        output_rank_count: int,
    ) -> "SegmentedRankPlan":
        output_rank_count = int(output_rank_count)
        if output_rank_count <= 0:
            raise ValueError("output_rank_count must be positive")
        flat = np.asarray(target_ranks).reshape(-1)
        if flat.size:
            if not np.issubdtype(flat.dtype, np.integer):
                raise TypeError("target ranks must be integers")
            minimum = int(flat.min())
            maximum = int(flat.max())
            if minimum < 0 or maximum >= output_rank_count:
                raise ValueError(
                    "target ranks must lie in [0, output_rank_count), got "
                    f"[{minimum}, {maximum}] for {output_rank_count} ranks"
                )
        else:
            maximum = 0
        targets = _readonly_indices(
            flat,
            maximum=max(maximum, output_rank_count - 1),
        )
        return cls(
            target_ranks=targets,
            source_count=int(targets.size),
            output_rank_count=output_rank_count,
        )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def memory_bytes(self) -> int:
        return int(self.target_ranks.nbytes)


@dataclass(frozen=True, slots=True, eq=False)
class DestinationRankPlan:
    """Compact inverse map for one destination-ordered rank action.

    Edge values use letter-major order.  A contiguous edge interval supplies
    the first output ranks directly; ``selected_edge_ids`` contains the
    remaining primary edges followed by the collision corrections.
    """

    selected_edge_ids: np.ndarray
    collision_target_ranks: np.ndarray
    edge_count: int
    output_rank_count: int
    primary_head_start: int
    primary_head_count: int
    tail_primary_count: int

    @classmethod
    def from_edges(
        cls,
        selected_edge_ids,
        collision_target_ranks,
        *,
        edge_count: int,
        output_rank_count: int,
        primary_head_start: int,
        primary_head_count: int,
        tail_primary_count: int,
    ) -> "DestinationRankPlan":
        """Build a compact immutable destination plan."""
        selected = np.asarray(selected_edge_ids).reshape(-1)
        collisions = np.asarray(collision_target_ranks).reshape(-1)
        if selected.size and not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("selected edge ids must be integers")
        if collisions.size and not np.issubdtype(collisions.dtype, np.integer):
            raise TypeError("collision target ranks must be integers")
        selected_maximum = max(int(edge_count) - 1, 0)
        collision_maximum = max(int(output_rank_count) - 1, 0)
        return cls(
            selected_edge_ids=_readonly_indices(
                selected,
                maximum=selected_maximum,
            ),
            collision_target_ranks=_readonly_indices(
                collisions,
                maximum=collision_maximum,
            ),
            edge_count=int(edge_count),
            output_rank_count=int(output_rank_count),
            primary_head_start=int(primary_head_start),
            primary_head_count=int(primary_head_count),
            tail_primary_count=int(tail_primary_count),
        )

    def __post_init__(self) -> None:
        selected = np.asarray(self.selected_edge_ids)
        collisions = np.asarray(self.collision_target_ranks)
        if selected.ndim != 1 or collisions.ndim != 1:
            raise ValueError(
                "destination rank-plan arrays must be one-dimensional"
            )
        if not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("selected edge ids must be integers")
        if not np.issubdtype(collisions.dtype, np.integer):
            raise TypeError("collision target ranks must be integers")
        if self.edge_count <= 0 or self.output_rank_count <= 0:
            raise ValueError("destination rank-plan counts must be positive")
        if self.primary_head_count <= 0:
            raise ValueError("primary head count must be positive")
        if self.primary_head_start < 0:
            raise ValueError("primary head start must be non-negative")
        if self.primary_head_start + self.primary_head_count > self.edge_count:
            raise ValueError("primary head lies outside the edge axis")
        if not 0 <= self.tail_primary_count <= selected.size:
            raise ValueError("tail primary count exceeds the selected edge axis")
        collision_count = selected.size - self.tail_primary_count
        if collisions.size != collision_count:
            raise ValueError(
                "collision targets must match the selected collision edges"
            )
        if (
            self.primary_head_count + self.tail_primary_count
            != self.output_rank_count
        ):
            raise ValueError("primary head and tail do not fill the output ranks")
        if selected.size != self.edge_count - self.primary_head_count:
            raise ValueError("selected edges must complement the primary head")
        if selected.size:
            if np.issubdtype(selected.dtype, np.signedinteger):
                if int(selected.min()) < 0:
                    raise ValueError("selected edge ids must be non-negative")
            if int(selected.max()) >= self.edge_count:
                raise ValueError("selected edge ids exceed the edge axis")
        if collisions.size:
            if np.issubdtype(collisions.dtype, np.signedinteger):
                if int(collisions.min()) < 0:
                    raise ValueError(
                        "collision target ranks must be non-negative"
                    )
            if int(collisions.max()) >= self.output_rank_count:
                raise ValueError("collision target ranks exceed the output axis")
            if np.any(collisions[1:] < collisions[:-1]):
                raise ValueError("collision target ranks must be sorted")

    @property
    def collision_count(self) -> int:
        return int(self.selected_edge_ids.size - self.tail_primary_count)

    def memory_bytes(self) -> int:
        return int(
            self.selected_edge_ids.nbytes
            + self.collision_target_ranks.nbytes
        )


def apply_segmented_rank_plan(
    xp: Any,
    values,
    plan: SegmentedRankPlan,
    *,
    scatter_add: Callable[[Any, Any, Any], Any],
):
    """Accumulate ``values[..., source, dense]`` by a static rank map."""
    if values.ndim < 2:
        raise ValueError("rank accumulation requires source and dense axes")
    if values.shape[-2] != plan.source_count:
        raise ValueError(
            f"source axis has length {values.shape[-2]}, expected "
            f"{plan.source_count}"
        )
    output = xp.zeros(
        values.shape[:-2] + (plan.output_rank_count, values.shape[-1]),
        dtype=values.dtype,
    )
    targets = xp.asarray(plan.target_ranks)
    return scatter_add(output, targets, values)


def apply_destination_rank_plan(
    xp: Any,
    edge_values,
    plan: DestinationRankPlan,
    *,
    scatter_add: Callable[[Any, Any, Any], Any],
):
    """Assemble destination ranks from compact letter-major edge values."""
    if edge_values.ndim < 2:
        raise ValueError("destination execution requires edge and dense axes")
    if edge_values.shape[-2] != plan.edge_count:
        raise ValueError(
            f"edge axis has length {edge_values.shape[-2]}, expected "
            f"{plan.edge_count}"
        )

    start = plan.primary_head_start
    stop = start + plan.primary_head_count
    head = edge_values[..., start:stop, :]
    if plan.selected_edge_ids.size == 0:
        return head

    selected = xp.take(
        edge_values,
        xp.asarray(plan.selected_edge_ids),
        axis=-2,
    )
    if plan.tail_primary_count:
        output = xp.concatenate(
            (head, selected[..., :plan.tail_primary_count, :]),
            axis=-2,
        )
    else:
        output = head
    if plan.collision_count:
        output = scatter_add(
            output,
            xp.asarray(plan.collision_target_ranks),
            selected[..., plan.tail_primary_count:, :],
        )
    return output


__all__ = [
    "DestinationRankPlan",
    "SegmentedRankPlan",
    "apply_destination_rank_plan",
    "apply_segmented_rank_plan",
]
