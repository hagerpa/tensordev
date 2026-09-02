"""Shared immutable rank-accumulation plans for many-to-one block kernels.

Representation-specific builders supply the target maps.  This module owns
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


__all__ = ["SegmentedRankPlan", "apply_segmented_rank_plan"]
