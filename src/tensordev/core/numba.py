"""Numba-backed implementations of the total-degree tensor cores."""

from __future__ import annotations

import numba as nb
import numpy as np

from .shuffle import (
    TotalDegreeShufflePlanStore,
    _direct_homogeneous_shuffle,
    _homogeneous_outer_product,
    _normalize_precompute_shuffle,
    _prepare_homogeneous_shuffle_inputs,
    _sum_homogeneous_axis_permutations,
)
from .universal import Universal


@nb.njit(fastmath=True, cache=True)
def _permutation_shuffle_nb(
        left: np.ndarray,
        right: np.ndarray,
        axis_permutations: np.ndarray,
        dimension: int,
        left_degree: int,
) -> np.ndarray:
    """Apply a compact homogeneous shuffle plan to flattened batches.

    ``axis_permutations[k]`` describes which input tensor axis occupies each
    output position for shuffle ``k``. Inverting that small table once per call
    lets the hot loop recover the left and right flat coordinates without
    materialising a coordinate-to-coordinate gather map.
    """
    permutation_count, output_degree = axis_permutations.shape
    inverse_permutations = np.empty_like(axis_permutations)
    for permutation_index in range(permutation_count):
        for output_axis in range(output_degree):
            input_axis = axis_permutations[permutation_index, output_axis]
            inverse_permutations[permutation_index, input_axis] = output_axis

    # Coordinate-major buffers make the batch loop contiguous for Numba
    # vectorization.  Each output word and its shuffled input coordinates are
    # therefore decoded once rather than once per batch item.
    left_by_coordinate = np.ascontiguousarray(left.T)
    right_by_coordinate = np.ascontiguousarray(right.T)
    output_width = left.shape[1] * right.shape[1]
    result = np.zeros((output_width, left.shape[0]), dtype=left.dtype)
    digits = np.empty(output_degree, dtype=np.intp)

    for output_index in range(output_width):
        remainder = output_index
        for axis in range(output_degree - 1, -1, -1):
            digits[axis] = remainder % dimension
            remainder //= dimension

        for permutation_index in range(permutation_count):
            left_index = 0
            for input_axis in range(left_degree):
                output_axis = inverse_permutations[
                    permutation_index, input_axis
                ]
                left_index = left_index * dimension + digits[output_axis]

            right_index = 0
            for input_axis in range(left_degree, output_degree):
                output_axis = inverse_permutations[
                    permutation_index, input_axis
                ]
                right_index = right_index * dimension + digits[output_axis]

            for batch_index in range(left.shape[0]):
                result[output_index, batch_index] += (
                    left_by_coordinate[left_index, batch_index]
                    * right_by_coordinate[right_index, batch_index]
                )

    return np.ascontiguousarray(result.T)


class NumbaTotalDegreeShufflePlanStore(TotalDegreeShufflePlanStore):
    """Ordinary shuffle plans executed by the compiled Numba kernel."""

    NUMPY_FALLBACK_BATCH_THRESHOLD = 8
    NUMPY_FALLBACK_OUTPUT_WIDTH_THRESHOLD = 1024

    def apply(self, xp, left, right, i: int, j: int):
        """Apply the precomputed homogeneous plan to broadcast input batches."""
        del xp
        plan = self.plan(i, j)

        # The compiled Numba shuffle contract uses float64 arrays irrespective
        # of input dtype.
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        batch_shape, left, right = _prepare_homogeneous_shuffle_inputs(
            np,
            left,
            right,
            plan,
        )

        if plan.uses_direct_scaling:
            return _direct_homogeneous_shuffle(left, right, plan)

        flat_batch_size = int(np.prod(batch_shape, dtype=np.int64))
        if (
            flat_batch_size < self.NUMPY_FALLBACK_BATCH_THRESHOLD
            and plan.output_width >= self.NUMPY_FALLBACK_OUTPUT_WIDTH_THRESHOLD
        ):
            outer = _homogeneous_outer_product(
                np,
                left,
                right,
                batch_shape,
                plan,
            )
            return _sum_homogeneous_axis_permutations(
                np,
                outer,
                batch_shape,
                plan,
            )

        left_flat = left.reshape(-1, plan.left_width)
        right_flat = right.reshape(-1, plan.right_width)
        output_flat = _permutation_shuffle_nb(
            left_flat,
            right_flat,
            plan.axis_permutations,
            plan.dimension,
            plan.left_degree,
        )
        return output_flat.reshape(batch_shape + (plan.output_width,))


class Numba(Universal[np.ndarray]):
    """NumPy tensor algebra with individually Numba-backed kernels where useful."""

    def __init__(
        self,
        *,
        d: int | None = None,
        max_trunc: int | None = None,
        default_trunc: int | None = None,
        precompute_shuffle: bool = False,
        shuffle_plan_store: TotalDegreeShufflePlanStore | None = None,
    ) -> None:
        shuffle_scope = _normalize_precompute_shuffle(
            precompute_shuffle,
            allow_generator=False,
        )
        if shuffle_plan_store is not None:
            if shuffle_scope != "none":
                raise ValueError(
                    "precompute_shuffle and shuffle_plan_store are mutually "
                    "exclusive."
                )
            if not isinstance(
                shuffle_plan_store,
                NumbaTotalDegreeShufflePlanStore,
            ):
                raise TypeError(
                    "shuffle_plan_store must be a Numba shuffle plan store, "
                    f"got {type(shuffle_plan_store).__name__}."
                )

        super().__init__(
            np,
            d=d,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            shuffle_plan_store=shuffle_plan_store,
        )
        if shuffle_scope == "full":
            if self.d is None or self.max_truncation is None:
                raise ValueError(
                    "precompute_shuffle requires both d and max_trunc."
                )
            self.shuffle_plan_store = NumbaTotalDegreeShufflePlanStore(
                self.d,
                self.max_truncation,
            )
