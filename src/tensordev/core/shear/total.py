"""Packed dense-total plans for ordered shear coordinates.

This module deliberately has no dependency on the bidegree package.  Letter
classes are static rectangular regions of one ordinary dense total-degree
level; they are encoded by bit masks and never become runtime PyTree leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from numbers import Integral
from operator import mul
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np
from numba import njit

from tensordev.core.shuffle import (
    _axis_dtype,
    _normalize_precompute_shuffle,
)
from tensordev.core.shear._compiled_support import (
    compile_forward_shear_support,
    compile_inverse_shear_support,
)
from tensordev.core.shear.symbolic import (
    gamma_axis_permutation,
    gamma_shuffle_support,
    psi_inverse_support,
    psi_support,
    right_generator_support,
    symbolic_plan_compilation_scope,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


ExecutionStrategy = Literal[
    "direct", "transpose_sum", "flat_gather", "coefficient_gather"
]

_MAX_TOTAL_SHEAR_TRANSFORM_DEGREE = np.iinfo(np.uint64).bits


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be a non-negative integer, got {value!r}."
        )
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _dims(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            "dims must be a pair (d_prime, d_doubleprime) of positive integers."
        )
    return (
        _positive_int(value[0], name="dims[0]"),
        _positive_int(value[1], name="dims[1]"),
    )


def _product(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


def _mask(positions: Sequence[int]) -> int:
    result = 0
    for position in positions:
        result |= 1 << int(position)
    return result


def _radices_for_mask(
    mask: int,
    degree: int,
    d_prime: int,
    d_doubleprime: int,
) -> tuple[int, ...]:
    return tuple(
        d_prime if mask & (1 << axis) else d_doubleprime
        for axis in range(degree)
    )


def _mixed_digits(radices: Sequence[int]) -> np.ndarray:
    radices = tuple(int(radix) for radix in radices)
    width = _product(radices)
    if not radices:
        return np.empty((1, 0), dtype=np.int64)
    divisors = np.asarray(
        [_product(radices[index + 1 :]) for index in range(len(radices))],
        dtype=np.int64,
    )
    coordinates = np.arange(width, dtype=np.int64)[:, None]
    return (coordinates // divisors[None, :]) % np.asarray(radices)[None, :]


@njit(cache=True, nogil=True)
def _lift_transform_permutations(
    output_masks: np.ndarray,
    input_masks: np.ndarray,
    doubleprime_permutations: np.ndarray,
    group_offsets: np.ndarray,
    prime_count: int,
    doubleprime_count: int,
) -> np.ndarray:
    """Lift grouped row permutations to ordinary full-axis permutations."""
    degree = prime_count + doubleprime_count
    permutations = np.empty((output_masks.size, degree), dtype=np.uint8)
    input_prime_axes = np.empty(prime_count, dtype=np.uint8)
    input_doubleprime_axes = np.empty(doubleprime_count, dtype=np.uint8)
    inverse_doubleprime = np.empty(doubleprime_count, dtype=np.uint8)

    for group in range(doubleprime_permutations.shape[0]):
        for input_axis in range(doubleprime_count):
            output_label = doubleprime_permutations[group, input_axis]
            inverse_doubleprime[output_label] = input_axis
        for term in range(group_offsets[group], group_offsets[group + 1]):
            prime_axis = 0
            doubleprime_axis = 0
            for axis in range(degree):
                if input_masks[term] & (
                    np.uint64(1) << np.uint64(axis)
                ):
                    input_prime_axes[prime_axis] = axis
                    prime_axis += 1
                else:
                    input_doubleprime_axes[doubleprime_axis] = axis
                    doubleprime_axis += 1

            prime_axis = 0
            doubleprime_axis = 0
            for output_axis in range(degree):
                if output_masks[term] & (
                    np.uint64(1) << np.uint64(output_axis)
                ):
                    permutations[term, output_axis] = input_prime_axes[
                        prime_axis
                    ]
                    prime_axis += 1
                else:
                    permutations[term, output_axis] = (
                        input_doubleprime_axes[
                            inverse_doubleprime[doubleprime_axis]
                        ]
                    )
                    doubleprime_axis += 1
    return permutations


@njit(cache=True, nogil=True)
def _mask_parity_coefficients(
    output_masks: np.ndarray,
    input_masks: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Return inverse-transform signs from primitive placement masks."""
    coefficients = np.ones(output_masks.size, dtype=np.int8)
    for term in range(output_masks.size):
        parity = 0
        for axis in range(1, degree, 2):
            bit = np.uint64(1) << np.uint64(axis)
            parity ^= int(bool(output_masks[term] & bit))
            parity ^= int(bool(input_masks[term] & bit))
        if parity:
            coefficients[term] = -1
    return coefficients


@dataclass(frozen=True, slots=True, eq=False)
class TotalMaskedPermutationExecutionGroup:
    """One output-mask group with a construction-time execution strategy."""

    output_mask: int
    term_start: int
    term_stop: int
    strategy: ExecutionStrategy
    chunk_size: int
    output_radices: tuple[int, ...]
    flat_indices: np.ndarray | None = None
    affine_offsets: np.ndarray | None = None
    affine_coefficients: np.ndarray | None = None
    validity_masks: np.ndarray | None = None
    coefficient_signs: np.ndarray | None = None

    @property
    def term_count(self) -> int:
        return self.term_stop - self.term_start

    def memory_bytes_by_category(self) -> dict[str, int]:
        def size(array) -> int:
            return 0 if array is None else int(array.nbytes)

        return {
            "derived_execution_maps": size(self.flat_indices),
            "derived_execution_coefficients": (
                size(self.affine_offsets)
                + size(self.affine_coefficients)
                + size(self.coefficient_signs)
            ),
            "derived_execution_masks": size(self.validity_masks),
        }


@dataclass(frozen=True, slots=True, eq=False)
class _PlanOrientation:
    """Traversal ordering and derived execution data for one direction."""

    term_order: np.ndarray | None
    group_masks: np.ndarray
    group_offsets: np.ndarray
    groups: tuple[TotalMaskedPermutationExecutionGroup, ...]

    def memory_bytes_by_category(self) -> dict[str, int]:
        categories = {
            "transpose_order": (
                0 if self.term_order is None else int(self.term_order.nbytes)
            ),
            "group_indices": int(self.group_masks.nbytes + self.group_offsets.nbytes),
            "derived_execution_maps": 0,
            "derived_execution_coefficients": 0,
            "derived_execution_masks": 0,
        }
        for group in self.groups:
            for name, size in group.memory_bytes_by_category().items():
                categories[name] += size
        return categories


@dataclass(frozen=True, slots=True, eq=False)
class TotalMaskedPermutationPlan:
    """Packed authoritative support for one dense masked-permutation map."""

    degree: int
    d_prime: int
    d_doubleprime: int
    source_domains: tuple[Literal["full", "prime", "doubleprime"], ...]
    source_radices: tuple[int, ...]
    permutation_table: np.ndarray
    inverse_permutation_table: np.ndarray
    term_output_masks: np.ndarray
    term_input_masks: np.ndarray
    term_permutation_ids: np.ndarray
    coefficient_mode: Literal["positive", "mask_parity"]
    forward: _PlanOrientation
    transpose: _PlanOrientation | None

    @property
    def term_count(self) -> int:
        return int(self.term_output_masks.size)

    @property
    def permutation_count(self) -> int:
        return int(self.permutation_table.shape[0])

    def memory_bytes_by_category(self) -> dict[str, int]:
        categories = {
            "axis_permutations": int(
                self.permutation_table.nbytes
                + self.inverse_permutation_table.nbytes
            ),
            "term_masks": int(
                self.term_output_masks.nbytes + self.term_input_masks.nbytes
            ),
            "term_permutation_ids": int(self.term_permutation_ids.nbytes),
            "transpose_order": 0,
            "group_indices": 0,
            "derived_execution_maps": 0,
            "derived_execution_coefficients": 0,
            "derived_execution_masks": 0,
        }
        for orientation in (self.forward, self.transpose):
            if orientation is None:
                continue
            for name, size in orientation.memory_bytes_by_category().items():
                categories[name] += size
        return categories

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())


class TotalShearPlanBuilder:
    """Compile host-side shear support into packed dense execution plans."""

    STATIC_TERM_THRESHOLD = 12
    FLAT_GATHER_MAX_BYTES = 256 * 1024
    COEFFICIENT_CHUNK_SIZE = 8

    def __init__(self, dims: tuple[int, int]):
        self.d_prime, self.d_doubleprime = _dims(dims)
        self.d = self.d_prime + self.d_doubleprime

    def _source_radices(self, domains) -> tuple[int, ...]:
        return tuple(
            self.d
            if domain == "full"
            else self.d_prime
            if domain == "prime"
            else self.d_doubleprime
            for domain in domains
        )

    @classmethod
    def _execution_strategy(
        cls,
        *,
        term_count: int,
        output_width: int,
    ) -> ExecutionStrategy:
        """Select the retained execution layout from scalar size metadata."""
        if term_count == 1:
            return "direct"
        if term_count <= cls.STATIC_TERM_THRESHOLD:
            return "transpose_sum"
        if (
            term_count * output_width * np.dtype(np.int32).itemsize
            <= cls.FLAT_GATHER_MAX_BYTES
        ):
            return "flat_gather"
        return "coefficient_gather"

    @classmethod
    def _padded_coefficient_count(cls, term_count: int) -> int:
        chunk = cls.COEFFICIENT_CHUNK_SIZE
        return ((term_count + chunk - 1) // chunk) * chunk

    @staticmethod
    def _affine_dtype(source_width: int):
        return (
            np.int32
            if source_width - 1 <= np.iinfo(np.int32).max
            else np.int64
        )

    def _orientation_structure(
        self,
        *,
        output_masks,
        input_masks,
        permutations,
        source_domains,
        source_radices,
        transpose: bool,
    ):
        """Return shared stable grouping and affine metadata for one direction."""
        target_masks = np.asarray(input_masks if transpose else output_masks)
        source_masks = np.asarray(output_masks if transpose else input_masks)
        count = int(target_masks.size)
        order = np.argsort(target_masks, kind="stable").astype(
            np.int32,
            copy=False,
        )
        if count:
            ordered_masks = target_masks[order]
            starts = np.flatnonzero(
                np.concatenate(
                    (
                        np.ones(1, dtype=np.bool_),
                        ordered_masks[1:] != ordered_masks[:-1],
                    )
                )
            )
            masks = tuple(map(int, ordered_masks[starts]))
            offsets = tuple(map(int, starts)) + (count,)
        else:
            masks = ()
            offsets = (0,)

        degree = len(source_domains)
        permutation_array = np.asarray(permutations, dtype=np.int64).reshape(
            len(permutations), degree
        )
        if transpose:
            permutation_array = np.argsort(permutation_array, axis=1)
        source_strides = np.asarray(
            [
                _product(source_radices[index + 1 :])
                for index in range(degree)
            ],
            dtype=np.int64,
        )
        full_domain_offsets = np.asarray(
            [
                self.d_prime * source_strides[axis]
                if domain == "full"
                else 0
                for axis, domain in enumerate(source_domains)
            ],
            dtype=np.int64,
        )
        axis_bits = np.left_shift(
            np.uint64(1),
            np.arange(degree, dtype=np.uint64),
        )
        return (
            order,
            masks,
            offsets,
            source_masks,
            permutation_array,
            source_strides,
            full_domain_offsets,
            axis_bits,
        )

    @staticmethod
    def _orientation_affine_arrays(
        *,
        source_masks,
        term_ids,
        permutation_ids,
        permutation_array,
        source_domains,
        source_strides,
        full_domain_offsets,
        axis_bits,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return vectorized offsets and coefficient rows for one mask group."""
        source_prime = (
            np.asarray(source_masks[term_ids], dtype=np.uint64)[:, None]
            & axis_bits[None, :]
        ) != 0
        for axis, domain in enumerate(source_domains):
            if domain == "prime" and not np.all(source_prime[:, axis]):
                raise ValueError("a prime-restricted source axis has a D mask bit")
            if domain == "doubleprime" and np.any(source_prime[:, axis]):
                raise ValueError("a D-restricted source axis has a prime mask bit")
        offsets = (~source_prime).astype(np.int64) @ full_domain_offsets
        term_permutation_ids = np.asarray(permutation_ids)[term_ids].astype(
            np.intp,
            copy=False,
        )
        coefficients = source_strides[
            permutation_array[term_permutation_ids]
        ]
        return offsets, coefficients

    def _orientation(
        self,
        *,
        output_masks: np.ndarray,
        input_masks: np.ndarray,
        permutation_ids: np.ndarray,
        permutations: tuple[tuple[int, ...], ...],
        coefficients: tuple[int, ...],
        source_domains,
        source_radices,
        transpose: bool,
    ) -> _PlanOrientation:
        (
            order,
            masks,
            offsets,
            source_masks,
            permutation_array,
            source_strides,
            full_domain_offsets,
            axis_bits,
        ) = self._orientation_structure(
            output_masks=output_masks,
            input_masks=input_masks,
            permutations=permutations,
            source_domains=source_domains,
            source_radices=source_radices,
            transpose=transpose,
        )

        groups = []
        for group_index, output_mask in enumerate(masks):
            start, stop = offsets[group_index], offsets[group_index + 1]
            term_ids = order[start:stop].astype(np.intp, copy=False)
            output_radices = _radices_for_mask(
                output_mask,
                len(source_domains),
                self.d_prime,
                self.d_doubleprime,
            )
            term_count = len(term_ids)
            output_width = _product(output_radices)
            strategy = self._execution_strategy(
                term_count=term_count,
                output_width=output_width,
            )

            flat_indices = None
            affine_offsets = None
            affine_coefficients = None
            validity = None
            signs = np.asarray(coefficients, dtype=np.int8)[
                term_ids
            ]
            if strategy in ("flat_gather", "coefficient_gather"):
                offsets_array, coefficient_rows = (
                    self._orientation_affine_arrays(
                        source_masks=source_masks,
                        term_ids=term_ids,
                        permutation_ids=permutation_ids,
                        permutation_array=permutation_array,
                        source_domains=source_domains,
                        source_strides=source_strides,
                        full_domain_offsets=full_domain_offsets,
                        axis_bits=axis_bits,
                    )
                )
                if strategy == "flat_gather":
                    digits = _mixed_digits(output_radices)
                    indices = offsets_array[:, None] + coefficient_rows @ digits.T
                    if indices.size and int(indices.max()) > np.iinfo(np.int32).max:
                        index_dtype = np.int64
                    else:
                        index_dtype = np.int32
                    flat_indices = _readonly(indices.astype(index_dtype))
                else:
                    chunk = self.COEFFICIENT_CHUNK_SIZE
                    padded = self._padded_coefficient_count(term_count)
                    pad = padded - term_count
                    affine_dtype = self._affine_dtype(
                        _product(source_radices)
                    )
                    affine_offsets = _readonly(
                        np.pad(offsets_array, (0, pad))
                        .astype(affine_dtype)
                        .reshape(-1, chunk)
                    )
                    affine_coefficients = _readonly(
                        np.pad(coefficient_rows, ((0, pad), (0, 0)))
                        .astype(affine_dtype)
                        .reshape(-1, chunk, len(source_domains))
                    )
                    validity = _readonly(
                        (np.arange(padded) < term_count).reshape(-1, chunk)
                    )
                    signs = np.pad(signs, (0, pad)).reshape(-1, chunk)
            groups.append(
                TotalMaskedPermutationExecutionGroup(
                    output_mask=output_mask,
                    term_start=start,
                    term_stop=stop,
                    strategy=strategy,
                    chunk_size=(
                        self.COEFFICIENT_CHUNK_SIZE
                        if strategy == "coefficient_gather"
                        else term_count
                    ),
                    output_radices=output_radices,
                    flat_indices=flat_indices,
                    affine_offsets=(
                        None if affine_offsets is None else _readonly(affine_offsets)
                    ),
                    affine_coefficients=(
                        None
                        if affine_coefficients is None
                        else _readonly(affine_coefficients)
                    ),
                    validity_masks=validity,
                    coefficient_signs=_readonly(np.asarray(signs, dtype=np.int8)),
                )
            )
        forward_order = None if not transpose else _readonly(order)
        return _PlanOrientation(
            term_order=forward_order,
            group_masks=_readonly(
                np.asarray(
                    masks,
                    dtype=_unsigned_index_dtype(max(masks, default=0)),
                )
            ),
            group_offsets=_readonly(np.asarray(offsets, dtype=np.int32)),
            groups=tuple(groups),
        )

    def _expected_orientation_memory(
        self,
        *,
        output_masks: tuple[int, ...],
        input_masks: tuple[int, ...],
        permutation_ids: tuple[int, ...],
        permutations: tuple[tuple[int, ...], ...],
        source_domains,
        source_radices,
        transpose: bool,
    ) -> dict[str, int]:
        """Count one orientation's retained buffers without building them."""
        count = len(output_masks)
        (
            order,
            masks,
            offsets,
            source_masks,
            permutation_array,
            source_strides,
            full_domain_offsets,
            axis_bits,
        ) = self._orientation_structure(
            output_masks=output_masks,
            input_masks=input_masks,
            permutations=permutations,
            source_domains=source_domains,
            source_radices=source_radices,
            transpose=transpose,
        )
        categories = {
            "transpose_order": count * np.dtype(np.int32).itemsize if transpose else 0,
            "group_indices": (
                len(masks)
                * np.dtype(
                    _unsigned_index_dtype(max(masks, default=0))
                ).itemsize
                + (len(masks) + 1) * np.dtype(np.int32).itemsize
            ),
            "derived_execution_maps": 0,
            "derived_execution_coefficients": 0,
            "derived_execution_masks": 0,
        }
        degree = len(source_domains)
        source_width = _product(source_radices)
        for group_index, output_mask in enumerate(masks):
            start, stop = offsets[group_index], offsets[group_index + 1]
            term_ids = order[start:stop].astype(np.intp, copy=False)
            output_radices = _radices_for_mask(
                output_mask,
                degree,
                self.d_prime,
                self.d_doubleprime,
            )
            term_count = len(term_ids)
            output_width = _product(output_radices)
            strategy = self._execution_strategy(
                term_count=term_count,
                output_width=output_width,
            )
            retained_count = term_count
            if strategy in ("flat_gather", "coefficient_gather"):
                offsets_array, coefficient_rows = (
                    self._orientation_affine_arrays(
                        source_masks=source_masks,
                        term_ids=term_ids,
                        permutation_ids=permutation_ids,
                        permutation_array=permutation_array,
                        source_domains=source_domains,
                        source_strides=source_strides,
                        full_domain_offsets=full_domain_offsets,
                        axis_bits=axis_bits,
                    )
                )
            if strategy == "flat_gather":
                largest_indices = offsets_array + coefficient_rows @ (
                    np.asarray(output_radices, dtype=np.int64) - 1
                )
                largest_index = int(np.max(largest_indices, initial=0))
                index_dtype = (
                    np.int64
                    if largest_index > np.iinfo(np.int32).max
                    else np.int32
                )
                categories["derived_execution_maps"] += (
                    term_count
                    * output_width
                    * np.dtype(index_dtype).itemsize
                )
            elif strategy == "coefficient_gather":
                retained_count = self._padded_coefficient_count(term_count)
                affine_itemsize = np.dtype(
                    self._affine_dtype(source_width)
                ).itemsize
                categories["derived_execution_coefficients"] += (
                    retained_count * (degree + 1) * affine_itemsize
                )
                categories["derived_execution_masks"] += (
                    retained_count * np.dtype(np.bool_).itemsize
                )
            # Every group retains one int8 coefficient per real or padded term.
            categories["derived_execution_coefficients"] += (
                retained_count * np.dtype(np.int8).itemsize
            )
        return categories

    def _expected_memory_bytes_by_category(
        self,
        *,
        degree: int,
        records,
        source_domains: Sequence[Literal["full", "prime", "doubleprime"]],
        with_transpose: bool,
    ) -> dict[str, int]:
        """Count the exact arrays retained by :meth:`build`."""
        degree = _non_negative_int(degree, name="degree")
        source_domains = tuple(source_domains)
        if len(source_domains) != degree:
            raise ValueError("source_domains length must equal degree")
        records = tuple(
            sorted(records, key=lambda record: (record[0], record[1], record[2]))
        )
        permutations = tuple(
            dict.fromkeys(tuple(record[2]) for record in records)
        )
        permutation_keys = {
            permutation: index for index, permutation in enumerate(permutations)
        }
        output_masks = tuple(int(record[0]) for record in records)
        input_masks = tuple(int(record[1]) for record in records)
        permutation_ids = tuple(
            permutation_keys[tuple(record[2])] for record in records
        )
        permutation_table = np.asarray(
            permutations,
            dtype=_axis_dtype(max(degree - 1, 0)),
        ).reshape(len(permutations), degree)
        return self._expected_memory_arrays(
            degree=degree,
            output_masks=np.asarray(output_masks),
            input_masks=np.asarray(input_masks),
            permutation_ids=np.asarray(permutation_ids),
            permutation_table=permutation_table,
            source_domains=source_domains,
            with_transpose=with_transpose,
        )

    def _expected_memory_arrays(
        self,
        *,
        degree: int,
        output_masks: np.ndarray,
        input_masks: np.ndarray,
        permutation_ids: np.ndarray,
        permutation_table: np.ndarray,
        source_domains: Sequence[
            Literal["full", "prime", "doubleprime"]
        ],
        with_transpose: bool,
    ) -> dict[str, int]:
        """Count retained buffers from canonical primitive record arrays."""
        source_domains = tuple(source_domains)
        permutations = tuple(
            tuple(map(int, permutation))
            for permutation in permutation_table
        )
        source_radices = self._source_radices(source_domains)

        categories = {
            "axis_permutations": (
                2
                * len(permutations)
                * degree
                * np.dtype(_axis_dtype(max(degree - 1, 0))).itemsize
            ),
            "term_masks": (
                len(output_masks)
                * (
                    np.dtype(
                        _unsigned_index_dtype(
                            int(np.max(output_masks, initial=0))
                        )
                    ).itemsize
                    + np.dtype(
                        _unsigned_index_dtype(
                            int(np.max(input_masks, initial=0))
                        )
                    ).itemsize
                )
            ),
            "term_permutation_ids": (
                len(output_masks)
                * np.dtype(
                    _unsigned_index_dtype(max(len(permutations) - 1, 0))
                ).itemsize
            ),
            "transpose_order": 0,
            "group_indices": 0,
            "derived_execution_maps": 0,
            "derived_execution_coefficients": 0,
            "derived_execution_masks": 0,
        }
        for transpose in ((False, True) if with_transpose else (False,)):
            orientation = self._expected_orientation_memory(
                output_masks=output_masks,
                input_masks=input_masks,
                permutation_ids=permutation_ids,
                permutations=permutations,
                source_domains=("full",) * degree if transpose else source_domains,
                source_radices=(self.d,) * degree if transpose else source_radices,
                transpose=transpose,
            )
            for name, size in orientation.items():
                categories[name] += size
        return categories

    def build(
        self,
        *,
        degree: int,
        records,
        source_domains: Sequence[Literal["full", "prime", "doubleprime"]],
        coefficient_mode: Literal["positive", "mask_parity"],
        with_transpose: bool,
    ) -> TotalMaskedPermutationPlan:
        degree = _non_negative_int(degree, name="degree")
        source_domains = tuple(source_domains)
        if len(source_domains) != degree:
            raise ValueError("source_domains length must equal degree")
        records = tuple(
            sorted(
                records,
                key=lambda record: (record[0], record[1], record[2]),
            )
        )
        permutations = tuple(dict.fromkeys(tuple(record[2]) for record in records))
        permutation_keys = {
            permutation: index
            for index, permutation in enumerate(permutations)
        }
        output_masks = np.asarray(
            [record[0] for record in records],
            dtype=_unsigned_index_dtype(
                max((record[0] for record in records), default=0)
            ),
        )
        input_masks = np.asarray(
            [record[1] for record in records],
            dtype=_unsigned_index_dtype(
                max((record[1] for record in records), default=0)
            ),
        )
        permutation_ids = np.asarray(
            [permutation_keys[tuple(record[2])] for record in records],
            dtype=_unsigned_index_dtype(max(len(permutations) - 1, 0)),
        )
        permutation_dtype = _axis_dtype(max(degree - 1, 0))
        permutation_table = np.asarray(
            permutations,
            dtype=permutation_dtype,
        ).reshape(len(permutations), degree)
        return self._build_arrays(
            degree=degree,
            output_masks=output_masks,
            input_masks=input_masks,
            permutation_ids=permutation_ids,
            permutation_table=permutation_table,
            source_domains=source_domains,
            coefficient_mode=coefficient_mode,
            with_transpose=with_transpose,
        )

    def _build_arrays(
        self,
        *,
        degree: int,
        output_masks: np.ndarray,
        input_masks: np.ndarray,
        permutation_ids: np.ndarray,
        permutation_table: np.ndarray,
        source_domains: Sequence[
            Literal["full", "prime", "doubleprime"]
        ],
        coefficient_mode: Literal["positive", "mask_parity"],
        with_transpose: bool,
    ) -> TotalMaskedPermutationPlan:
        """Build a plan from canonical primitive record arrays."""
        source_domains = tuple(source_domains)
        if len(source_domains) != degree:
            raise ValueError("source_domains length must equal degree")
        output_masks = np.array(
            output_masks,
            dtype=_unsigned_index_dtype(
                int(np.max(output_masks, initial=0))
            ),
            copy=True,
            order="C",
        )
        input_masks = np.array(
            input_masks,
            dtype=_unsigned_index_dtype(
                int(np.max(input_masks, initial=0))
            ),
            copy=True,
            order="C",
        )
        permutation_dtype = _axis_dtype(max(degree - 1, 0))
        permutation_count = int(np.asarray(permutation_table).shape[0])
        permutation_table = np.array(
            permutation_table,
            dtype=permutation_dtype,
            copy=True,
            order="C",
        ).reshape(permutation_count, degree)
        permutation_ids = np.array(
            permutation_ids,
            dtype=_unsigned_index_dtype(
                max(permutation_table.shape[0] - 1, 0)
            ),
            copy=True,
            order="C",
        )
        permutations = tuple(
            tuple(map(int, permutation))
            for permutation in permutation_table
        )
        if coefficient_mode == "positive":
            coefficients = np.ones(output_masks.size, dtype=np.int8)
        else:
            coefficients = _mask_parity_coefficients(
                output_masks.astype(np.uint64, copy=False),
                input_masks.astype(np.uint64, copy=False),
                degree,
            )
        source_radices = self._source_radices(source_domains)
        forward = self._orientation(
            output_masks=output_masks,
            input_masks=input_masks,
            permutation_ids=permutation_ids,
            permutations=permutations,
            coefficients=coefficients,
            source_domains=source_domains,
            source_radices=source_radices,
            transpose=False,
        )
        transpose_orientation = None
        if with_transpose:
            transpose_orientation = self._orientation(
                output_masks=output_masks,
                input_masks=input_masks,
                permutation_ids=permutation_ids,
                permutations=permutations,
                coefficients=coefficients,
                source_domains=("full",) * degree,
                source_radices=(self.d,) * degree,
                transpose=True,
            )
        inverse = np.argsort(permutation_table, axis=1).astype(
            permutation_dtype,
            copy=False,
        )
        return TotalMaskedPermutationPlan(
            degree=degree,
            d_prime=self.d_prime,
            d_doubleprime=self.d_doubleprime,
            source_domains=source_domains,
            source_radices=source_radices,
            permutation_table=_readonly(permutation_table),
            inverse_permutation_table=_readonly(inverse),
            term_output_masks=_readonly(output_masks),
            term_input_masks=_readonly(input_masks),
            term_permutation_ids=_readonly(permutation_ids),
            coefficient_mode=coefficient_mode,
            forward=forward,
            transpose=transpose_orientation,
        )

    def _compiled_transform_arrays(
        self,
        degree: int,
        *,
        inverse: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return canonical total-transform records as primitive arrays."""
        if degree > _MAX_TOTAL_SHEAR_TRANSFORM_DEGREE:
            raise ValueError(
                "total shear transform degree must be <= "
                f"{_MAX_TOTAL_SHEAR_TRANSFORM_DEGREE}, got {degree}."
            )
        compiler = (
            compile_inverse_shear_support
            if inverse
            else compile_forward_shear_support
        )
        output_chunks = []
        input_chunks = []
        permutation_chunks = []
        for prime_count in range(degree + 1):
            support = compiler(prime_count, degree - prime_count)
            output_chunks.append(support.output_masks)
            input_chunks.append(support.input_masks)
            permutation_chunks.append(
                _lift_transform_permutations(
                    support.output_masks,
                    support.input_masks,
                    support.doubleprime_permutations,
                    support.group_offsets,
                    support.prime_count,
                    support.doubleprime_count,
                )
            )

        output_masks = np.concatenate(output_chunks)
        input_masks = np.concatenate(input_chunks)
        term_permutations = np.concatenate(permutation_chunks, axis=0)
        permutation_keys = tuple(
            term_permutations[:, axis]
            for axis in range(degree - 1, -1, -1)
        )
        order = np.lexsort(
            permutation_keys + (input_masks, output_masks)
        )
        output_masks = output_masks[order]
        input_masks = input_masks[order]
        term_permutations = term_permutations[order]

        (
            lexicographic_permutations,
            first_indices,
            lexicographic_ids,
        ) = np.unique(
            term_permutations,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        first_occurrence_order = np.argsort(first_indices, kind="stable")
        permutation_table = lexicographic_permutations[
            first_occurrence_order
        ]
        lexicographic_to_first = np.empty(
            first_occurrence_order.size,
            dtype=np.int64,
        )
        lexicographic_to_first[first_occurrence_order] = np.arange(
            first_occurrence_order.size,
            dtype=np.int64,
        )
        permutation_ids = lexicographic_to_first[lexicographic_ids]
        return (
            output_masks,
            input_masks,
            permutation_ids,
            permutation_table,
        )

    def _transform_records(self, degree: int, *, inverse: bool):
        records = []
        for prime_count in range(degree + 1):
            support = (
                psi_inverse_support(prime_count, degree - prime_count)
                if inverse
                else psi_support(prime_count, degree - prime_count)
            )
            records.extend(
                (
                    _mask(term.output_prime_positions),
                    _mask(term.input_prime_positions),
                    term.axis_permutation,
                )
                for term in support
            )
        return tuple(records)

    def transform(self, degree: int, *, inverse: bool) -> TotalMaskedPermutationPlan:
        degree = _non_negative_int(degree, name="degree")
        (
            output_masks,
            input_masks,
            permutation_ids,
            permutation_table,
        ) = self._compiled_transform_arrays(degree, inverse=inverse)
        return self._build_arrays(
            degree=degree,
            output_masks=output_masks,
            input_masks=input_masks,
            permutation_ids=permutation_ids,
            permutation_table=permutation_table,
            source_domains=("full",) * degree,
            coefficient_mode="mask_parity" if inverse else "positive",
            with_transpose=True,
        )

    def _expected_transform_memory(
        self,
        degree: int,
        *,
        inverse: bool,
    ) -> dict[str, int]:
        degree = _non_negative_int(degree, name="degree")
        (
            output_masks,
            input_masks,
            permutation_ids,
            permutation_table,
        ) = self._compiled_transform_arrays(degree, inverse=inverse)
        return self._expected_memory_arrays(
            degree=degree,
            output_masks=output_masks,
            input_masks=input_masks,
            permutation_ids=permutation_ids,
            permutation_table=permutation_table,
            source_domains=("full",) * degree,
            with_transpose=True,
        )

    def _generator_records(self, output_degree: int):
        return tuple(
            (
                _mask(term.output_prime_positions),
                _mask(term.input_prime_positions) | (1 << (output_degree - 1)),
                term.dense_permutation,
            )
            for term in right_generator_support(output_degree)
            if term.generator_is_prime
        )

    def generator(self, output_degree: int) -> TotalMaskedPermutationPlan:
        return self.build(
            degree=output_degree,
            records=self._generator_records(output_degree),
            source_domains=("full",) * (output_degree - 1) + ("prime",),
            coefficient_mode="positive",
            with_transpose=False,
        )

    def _expected_generator_memory(self, output_degree: int) -> dict[str, int]:
        return self._expected_memory_bytes_by_category(
            degree=output_degree,
            records=self._generator_records(output_degree),
            source_domains=("full",) * (output_degree - 1) + ("prime",),
            with_transpose=False,
        )

    def _gamma_records(self, left_degree: int, right_degree: int):
        return tuple(
            (
                _mask(term.output_prime_positions),
                _mask(term.left_prime_positions)
                | (_mask(term.right_prime_positions) << left_degree),
                gamma_axis_permutation(term, left_degree, right_degree),
            )
            for term in gamma_shuffle_support(left_degree, right_degree)
        )

    def shuffle(
        self,
        left_degree: int,
        right_degree: int,
    ) -> TotalMaskedPermutationPlan:
        degree = left_degree + right_degree
        return self.build(
            degree=degree,
            records=self._gamma_records(left_degree, right_degree),
            source_domains=("full",) * degree,
            coefficient_mode="positive",
            with_transpose=False,
        )

    def _expected_shuffle_memory(
        self,
        left_degree: int,
        right_degree: int,
    ) -> dict[str, int]:
        degree = left_degree + right_degree
        return self._expected_memory_bytes_by_category(
            degree=degree,
            records=self._gamma_records(left_degree, right_degree),
            source_domains=("full",) * degree,
            with_transpose=False,
        )


def _total_gamma_pairs(max_truncation: int, scope: str):
    for total in range(max_truncation + 1):
        for right_degree in range(total // 2 + 1):
            left_degree = total - right_degree
            if (
                scope == "generator"
                and right_degree != 1
                and left_degree != 1
            ):
                continue
            yield left_degree, right_degree


@symbolic_plan_compilation_scope()
def _expected_total_shear_memory_bytes_by_category(
    dims: tuple[int, int],
    max_trunc: int,
    *,
    precompute_shuffle: bool | Literal["generator"] = False,
) -> Mapping[str, int]:
    """Exact retained total-shear payload without constructing plan arrays."""
    normalized_dims = _dims(dims)
    capacity = _non_negative_int(max_trunc, name="max_trunc")
    scope = _normalize_precompute_shuffle(
        precompute_shuffle, allow_generator=True
    )
    builder = TotalShearPlanBuilder(normalized_dims)
    family_memories: dict[str, list[dict[str, int]]] = {
        "forward": [
            builder._expected_transform_memory(degree, inverse=False)
            for degree in range(capacity + 1)
        ],
        "inverse": [
            builder._expected_transform_memory(degree, inverse=True)
            for degree in range(capacity + 1)
        ],
        "generator": [
            builder._expected_generator_memory(degree)
            for degree in range(1, capacity + 1)
        ],
        "shuffle": (
            []
            if scope == "none"
            else [
                builder._expected_shuffle_memory(left_degree, right_degree)
                for left_degree, right_degree in _total_gamma_pairs(
                    capacity, scope
                )
            ]
        ),
    }
    categories: dict[str, int] = {}
    for family, plans in family_memories.items():
        if not plans:
            continue
        names = plans[0]
        for name in names:
            categories[f"{family}_{name}"] = sum(
                plan[name] for plan in plans
            )
    return MappingProxyType(categories)


class TotalShearPlanStore:
    """Required total shear data and optional shuffle plans."""

    @symbolic_plan_compilation_scope()
    def __init__(
        self,
        dims: tuple[int, int],
        max_trunc: int,
        *,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        self.dims = _dims(dims)
        self.d = sum(self.dims)
        self.max_truncation = _non_negative_int(max_trunc, name="max_trunc")
        self.shuffle_scope = _normalize_precompute_shuffle(
            precompute_shuffle, allow_generator=True
        )
        builder = TotalShearPlanBuilder(self.dims)
        self.forward_plans = MappingProxyType(
            {
                degree: builder.transform(degree, inverse=False)
                for degree in range(self.max_truncation + 1)
            }
        )
        self.inverse_plans = MappingProxyType(
            {
                degree: builder.transform(degree, inverse=True)
                for degree in range(self.max_truncation + 1)
            }
        )
        self.generator_plans = MappingProxyType(
            {
                degree: builder.generator(degree)
                for degree in range(1, self.max_truncation + 1)
            }
        )
        shuffle_plans = {}
        if self.shuffle_scope != "none":
            for left_degree, right_degree in _total_gamma_pairs(
                self.max_truncation, self.shuffle_scope
            ):
                shuffle_plans[(left_degree, right_degree)] = builder.shuffle(
                    left_degree, right_degree
                )
        self.shuffle_plans = MappingProxyType(shuffle_plans)

    def shuffle_plan(self, left_degree: int, right_degree: int):
        left_degree = _non_negative_int(left_degree, name="left_degree")
        right_degree = _non_negative_int(right_degree, name="right_degree")
        swapped = left_degree < right_degree
        key = (
            (right_degree, left_degree) if swapped else (left_degree, right_degree)
        )
        try:
            return self.shuffle_plans[key], swapped
        except KeyError as error:
            if self.shuffle_scope == "none":
                raise RuntimeError(
                    "This shear core was constructed with precompute_shuffle=False."
                ) from error
            raise RuntimeError(
                "Arbitrary shear shuffle products require precompute_shuffle=True; "
                "this core contains generator-only plans."
            ) from error

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        categories: dict[str, int] = {}
        for family, plans in (
            ("forward", self.forward_plans),
            ("inverse", self.inverse_plans),
            ("generator", self.generator_plans),
            ("shuffle", self.shuffle_plans),
        ):
            family_categories: dict[str, int] = {}
            for plan in plans.values():
                for name, size in plan.memory_bytes_by_category().items():
                    family_categories[name] = family_categories.get(name, 0) + size
            for name, size in family_categories.items():
                categories[f"{family}_{name}"] = size
        return MappingProxyType(categories)

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def plan_statistics(self) -> Mapping[str, object]:
        families = {
            "forward": self.forward_plans,
            "inverse": self.inverse_plans,
            "generator": self.generator_plans,
            "shuffle": self.shuffle_plans,
        }
        plan_counts = {
            family: len(plans) for family, plans in families.items()
        }
        term_counts = {
            family: sum(plan.term_count for plan in plans.values())
            for family, plans in families.items()
        }
        permutation_counts = {
            family: sum(plan.permutation_count for plan in plans.values())
            for family, plans in families.items()
        }
        execution_groups = {
            family: sum(
                len(orientation.groups)
                for plan in plans.values()
                for orientation in (plan.forward, plan.transpose)
                if orientation is not None
            )
            for family, plans in families.items()
        }
        strategy_counts = {
            strategy: sum(
                group.strategy == strategy
                for plans in families.values()
                for plan in plans.values()
                for orientation in (plan.forward, plan.transpose)
                if orientation is not None
                for group in orientation.groups
            )
            for strategy in (
                "direct",
                "transpose_sum",
                "flat_gather",
                "coefficient_gather",
            )
        }
        categories = dict(self.memory_bytes_by_category())
        authoritative_suffixes = {
            "axis_permutations",
            "term_masks",
            "term_permutation_ids",
        }
        authoritative_bytes = sum(
            size
            for name, size in categories.items()
            if any(name.endswith(suffix) for suffix in authoritative_suffixes)
        )
        derived_bytes = sum(categories.values()) - authoritative_bytes
        return MappingProxyType(
            {
                "dims": self.dims,
                "d": self.d,
                "max_truncation": self.max_truncation,
                "coordinates": "shear",
                "grading": "total_degree",
                "shuffle_scope": self.shuffle_scope,
                "plan_counts": MappingProxyType(plan_counts),
                "term_counts": MappingProxyType(term_counts),
                "permutation_counts": MappingProxyType(permutation_counts),
                "execution_group_counts": MappingProxyType(execution_groups),
                "strategy_counts": MappingProxyType(strategy_counts),
                "forward_terms": term_counts["forward"],
                "inverse_terms": term_counts["inverse"],
                "generator_terms": term_counts["generator"],
                "shuffle_terms": term_counts["shuffle"],
                "authoritative_memory_bytes": authoritative_bytes,
                "derived_execution_memory_bytes": derived_bytes,
                "bytes_by_category": MappingProxyType(categories),
                "memory_bytes": sum(categories.values()),
            }
        )


def _axis_slice(mask: int, axis: int, d_prime: int, domain: str):
    is_prime = bool(mask & (1 << axis))
    if domain == "full":
        return slice(0, d_prime) if is_prime else slice(d_prime, None)
    if domain == "prime":
        if not is_prime:
            raise ValueError("prime source domain paired with a D mask")
        return slice(None)
    if is_prime:
        raise ValueError("double-prime source domain paired with a prime mask")
    return slice(None)


def _output_slices(mask: int, degree: int, d_prime: int):
    return tuple(
        slice(0, d_prime) if mask & (1 << axis) else slice(d_prime, None)
        for axis in range(degree)
    )


def _apply_affine_coefficient_chunks(
    xp,
    flat,
    group: TotalMaskedPermutationExecutionGroup,
    digits,
):
    """Sum bounded mixed-radix gather chunks without unrolling on JAX."""
    batch_ndim = flat.ndim - 1
    output_width = digits.shape[0]
    value = xp.zeros(flat.shape[:-1] + (output_width,), dtype=flat.dtype)
    chunks = (
        xp.asarray(group.affine_offsets),
        xp.asarray(group.affine_coefficients),
        xp.asarray(group.validity_masks),
        xp.asarray(group.coefficient_signs),
    )

    def add_chunk(accumulator, chunk):
        offsets, coefficients, valid, signs = chunk
        indices = offsets[:, None] + coefficients @ digits.T
        values = xp.take(flat, indices, axis=-1)
        weights = xp.asarray(valid, dtype=flat.dtype) * xp.asarray(
            signs, dtype=flat.dtype
        )
        contribution = (
            values
            * weights.reshape((1,) * batch_ndim + (-1, 1))
        ).sum(axis=-2, dtype=flat.dtype)
        return accumulator + contribution

    if getattr(xp, "__name__", "").startswith("jax.numpy"):
        # The number of retained chunks may be large.  A device scan keeps the
        # gather buffer bounded by ``group.chunk_size`` and gives the compiled
        # program one loop body instead of one copy per retained chunk.
        from jax import lax

        value, _ = lax.scan(
            lambda accumulator, chunk: (
                add_chunk(accumulator, chunk),
                None,
            ),
            value,
            chunks,
        )
        return value

    for chunk in zip(*chunks):
        value = add_chunk(value, chunk)
    return value


def apply_total_masked_permutation_plan(
    xp,
    raw,
    plan: TotalMaskedPermutationPlan,
    *,
    transpose: bool = False,
):
    """Apply one packed dense masked-permutation plan on the last axes."""
    if not isinstance(plan, TotalMaskedPermutationPlan):
        raise TypeError("plan must be a TotalMaskedPermutationPlan")
    orientation = plan.transpose if transpose else plan.forward
    if orientation is None:
        raise ValueError("this plan has no transpose traversal")
    degree = plan.degree
    source_domains = ("full",) * degree if transpose else plan.source_domains
    source_radices = (
        (plan.d_prime + plan.d_doubleprime,) * degree
        if transpose
        else plan.source_radices
    )
    expected_width = _product(source_radices)
    if raw.shape[-1] != expected_width:
        raise ValueError(
            f"plan expects source width {expected_width}, got {raw.shape[-1]}."
        )
    batch = raw.shape[:-1]
    tensor = raw.reshape(batch + source_radices)
    flat = raw.reshape(batch + (expected_width,))
    d = plan.d_prime + plan.d_doubleprime
    result = xp.zeros(batch + (d,) * degree, dtype=raw.dtype)
    term_order = orientation.term_order

    for group in orientation.groups:
        if term_order is None:
            term_ids = tuple(range(group.term_start, group.term_stop))
        else:
            term_ids = tuple(
                int(term_id)
                for term_id in term_order[group.term_start : group.term_stop]
            )
        source_masks = (
            plan.term_output_masks if transpose else plan.term_input_masks
        )
        permutation_table = (
            plan.inverse_permutation_table
            if transpose
            else plan.permutation_table
        )
        if group.strategy in ("direct", "transpose_sum"):
            terms = []
            for local_index, term_id in enumerate(term_ids):
                source_mask = int(source_masks[term_id])
                selected = tensor[
                    (slice(None),) * len(batch)
                    + tuple(
                        _axis_slice(
                            source_mask,
                            axis,
                            plan.d_prime,
                            source_domains[axis],
                        )
                        for axis in range(degree)
                    )
                ]
                permutation_id = int(plan.term_permutation_ids[term_id])
                axes = tuple(int(axis) for axis in permutation_table[permutation_id])
                selected = xp.transpose(
                    selected,
                    tuple(range(len(batch)))
                    + tuple(len(batch) + axis for axis in axes),
                )
                sign = int(np.asarray(group.coefficient_signs).reshape(-1)[local_index])
                terms.append(selected if sign == 1 else -selected)
            value = terms[0]
            while len(terms) > 1:
                terms = [
                    terms[index] + terms[index + 1]
                    if index + 1 < len(terms)
                    else terms[index]
                    for index in range(0, len(terms), 2)
                ]
            value = terms[0]
        elif group.strategy == "flat_gather":
            indices = xp.asarray(group.flat_indices)
            values = xp.take(flat, indices, axis=-1)
            signs = xp.asarray(group.coefficient_signs, dtype=raw.dtype)
            value = (
                values * signs.reshape((1,) * len(batch) + (-1, 1))
            ).sum(axis=-2, dtype=raw.dtype).reshape(
                batch + group.output_radices
            )
        else:
            output_width = _product(group.output_radices)
            affine_dtype = np.asarray(group.affine_offsets).dtype
            coordinates = xp.arange(output_width, dtype=affine_dtype)
            divisors = xp.asarray(
                [
                    _product(group.output_radices[index + 1 :])
                    for index in range(degree)
                ],
                dtype=affine_dtype,
            )
            radices = xp.asarray(group.output_radices, dtype=affine_dtype)
            digits = (coordinates[:, None] // divisors[None, :]) % radices[None, :]
            value = _apply_affine_coefficient_chunks(
                xp,
                flat,
                group,
                digits,
            )
            value = value.reshape(batch + group.output_radices)
        target = (
            (slice(None),) * len(batch)
            + _output_slices(group.output_mask, degree, plan.d_prime)
        )
        if hasattr(result, "at"):
            result = result.at[target].set(value)
        else:
            result[target] = value
    return result.reshape(batch + (d**degree,))


__all__ = [
    "TotalMaskedPermutationExecutionGroup",
    "TotalMaskedPermutationPlan",
    "TotalShearPlanBuilder",
    "TotalShearPlanStore",
    "apply_total_masked_permutation_plan",
]
