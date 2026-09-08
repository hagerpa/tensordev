"""Placement-factored ordered-bidegree shear coordinates.

The numerical storage in this module is exactly ``BigradedTensor``:
one placement axis followed by packed prime and double-prime dense axes.  No
operation below constructs or traverses a dense total-degree level.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from types import MappingProxyType
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

from tensordev.core.capabilities import _WORDWISE_SIGNATURE_PROTOCOL
from tensordev.core.shear._compiled_gamma import (
    _ShearShuffleWorkspace,
    _shear_shuffle_group_count,
    _shear_shuffle_term_count,
    compile_shear_shuffle_support,
    shear_shuffle_colex_output_pair_indices,
    shear_shuffle_dense_axis_permutations,
)
from tensordev.core.shear._compiled_generator import (
    _generator_prime_term_count,
    compile_bigraded_shear_generator_support,
)
from tensordev.core.shear._compiled_support import (
    _MAX_MASK_DEGREE,
    _forward_group_count,
    _forward_term_count,
    _inverse_group_count,
    _inverse_term_count,
    compile_forward_shear_support,
    compile_inverse_shear_support,
    shear_colex_rank_pairs,
)
from tensordev.core.bigraded.jax_backend import _JaxBigradedBackend
from tensordev.core.bigraded.precompute import (
    BigradedPlanStore,
    _expected_plan_memory_bytes_by_category,
    colex_placements,
    colex_rank,
)
from tensordev.core.bigraded.shuffle import (
    _dense_axis_permutation,
    _pair_in_scope,
    _placement_pair_outer,
    _shuffle_scope,
)
from tensordev.core.bigraded.standard import StandardBigradedCore
from tensordev.core.bigraded.types import (
    Bidegree,
    BigradedSpec,
    _bidegree,
)
from tensordev.core.shear.bigraded_transport import BigradedShearCoordinateCore
from tensordev.core.shear.bigraded_jax_transport import (
    _bind_bigraded_shear_jax_methods,
)
from tensordev.core.shear.symbolic import (
    gamma_shuffle_pattern_support,
    invert_permutation,
    psi_inverse_support,
    psi_support,
    symbolic_plan_compilation_scope,
)
from tensordev.core.shuffle import (
    _normalize_precompute_shuffle,
    _precompute_shuffle_argument,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


_COMPILED_SHEAR_SHUFFLE_ENTRY_THRESHOLD = 100_000
_COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD = 100_000


def _packed_transform_permutation(n: int, permutation: tuple[int, ...]):
    """Packed prime/double-prime transpose for a symbolic row permutation."""
    inverse = invert_permutation(permutation)
    return tuple(range(n)) + tuple(n + axis for axis in inverse)


def _transform_rank_groups(grade: Bidegree, *, inverse: bool):
    n, m = grade
    placement_count = comb(n + m, n)
    support = psi_inverse_support(n, m) if inverse else psi_support(n, m)
    grouped: dict[tuple[int, ...], list[int]] = {}
    for term in support:
        output_rank = colex_rank(term.output_prime_positions)
        input_rank = colex_rank(term.input_prime_positions)
        permutation = _packed_transform_permutation(
            n, term.doubleprime_permutation
        )
        grouped.setdefault(permutation, []).append(
            output_rank * placement_count + input_rank
        )
    return placement_count, grouped


def _gamma_rank_groups(left_grade: Bidegree, right_grade: Bidegree):
    n1, m1 = left_grade
    n2, m2 = right_grade
    left_placements = colex_placements(n1 + m1, n1)
    right_placements = colex_placements(n2 + m2, n2)
    output_grade = n1 + n2, m1 + m2
    output_count = comb(sum(output_grade), output_grade[0])
    pair_count = len(left_placements) * len(right_placements)
    grouped: dict[tuple[int, ...], list[int]] = {}
    for left_rank, left_placement in enumerate(left_placements):
        for right_rank, right_placement in enumerate(right_placements):
            pair_index = left_rank * len(right_placements) + right_rank
            for term in gamma_shuffle_pattern_support(
                n1 + m1,
                n2 + m2,
                left_placement,
                right_placement,
            ):
                output_rank = colex_rank(term.output_prime_positions)
                permutation = _dense_axis_permutation(
                    left_grade,
                    right_grade,
                    term.prime_interleaving,
                    term.doubleprime_interleaving,
                )
                grouped.setdefault(permutation, []).append(
                    output_rank * pair_count + pair_index
                )
    return output_grade, output_count, pair_count, grouped


def _unbalanced_transform_group_size_counts(
    grade: Bidegree,
    *,
    inverse: bool,
) -> tuple[tuple[int, int], ...]:
    """Return ``(terms per group, group count)`` for a one-sided grade."""
    n, m = grade
    if n == 0 or m == 0:
        return ((1, 1),)
    if m == 1:
        term_count = 2 * n + 1 if inverse else comb(n + 2, 2)
        return ((term_count, 1),)
    if n != 1:
        raise ValueError("group-size shortcut requires n <= 1 or m <= 1")

    identity_terms = 2 * m + 1 if inverse else comb(m + 2, 2)
    if inverse:
        remaining_groups = (1 << m) - m - 1
        return (
            ((identity_terms, 1), (2, remaining_groups))
            if remaining_groups
            else ((identity_terms, 1),)
        )
    return ((identity_terms, 1),) + tuple(
        (group_terms, (1 << (m - group_terms)) - 1)
        for group_terms in range(1, m)
    )


def _forward_identity_group_size(grade: Bidegree) -> int:
    """Return the support size of the identity-permutation forward group."""
    n, m = grade
    degree = n + m
    return comb(degree, n) * comb(degree + 1, n) // (n + 1)


def _inverse_identity_group_size(grade: Bidegree) -> int:
    """Return the support size of the identity-permutation inverse group."""
    n, m = grade
    return sum(
        comb(n, shared) * comb(m, shared) * (1 << shared)
        for shared in range(min(n, m) + 1)
    )


def _transform_rank_matrix_group_count(
    grade: Bidegree,
    *,
    inverse: bool,
    placement_count: int,
    transform_itemsize: int,
) -> int:
    """Count groups whose retained dense rank matrix is size-competitive."""
    if placement_count > 64:
        return 0

    matrix_bytes = placement_count**2 * np.dtype(np.int8).itemsize
    required_terms = (matrix_bytes + 2 * transform_itemsize - 1) // (
        2 * transform_itemsize
    )
    n, m = grade
    if n <= 1 or m <= 1:
        return sum(
            group_count
            for group_terms, group_count in (
                _unbalanced_transform_group_size_counts(
                    grade,
                    inverse=inverse,
                )
            )
            if group_terms >= required_terms
        )

    # For a retained balanced grade, C(n + m, n) <= 64.  Consequently the
    # complete finite family has 2 <= n, m <= 9.  In that family the identity
    # permutation is the only forward group that can meet the exact matrix
    # threshold, while no inverse group can meet it.  Its forward size is the
    # plane-partition count below; the inverse identity (and maximum) is the
    # Delannoy number.  These finite exact bounds avoid enumerating
    # permutations merely to decide whether a derived execution matrix is
    # retained.
    if inverse:
        maximum_group_terms = _inverse_identity_group_size(grade)
        if maximum_group_terms >= required_terms:
            raise AssertionError(
                "balanced inverse rank-matrix bound disagrees with the "
                "retention threshold"
            )
        return 0

    identity_terms = _forward_identity_group_size(grade)
    return int(identity_terms >= required_terms)


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShearTransformKeyPlan:
    """Rank pairs sharing one packed dense-axis permutation."""

    encoded_rank_pairs: np.ndarray
    dense_axis_permutation: tuple[int, ...]
    rank_matrix: np.ndarray | None = None

    def rank_pairs(self, placement_count: int) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.encoded_rank_pairs
        return encoded // placement_count, encoded % placement_count

    def memory_bytes(self) -> int:
        return int(
            self.encoded_rank_pairs.nbytes
            + len(self.dense_axis_permutation) * np.dtype(np.intp).itemsize
            + (0 if self.rank_matrix is None else self.rank_matrix.nbytes)
        )


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShearTransformPlan:
    grade: Bidegree
    placement_count: int
    dense_shape: tuple[int, int, int]
    signed: bool
    placement_parities: np.ndarray
    key_plans: tuple[BigradedShearTransformKeyPlan, ...]

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "rank_pairs": sum(key.encoded_rank_pairs.nbytes for key in self.key_plans),
            "dense_permutations": sum(
                len(key.dense_axis_permutation) * np.dtype(np.intp).itemsize
                for key in self.key_plans
            ),
            "parities": int(self.placement_parities.nbytes),
            "rank_matrices": sum(
                0 if key.rank_matrix is None else key.rank_matrix.nbytes
                for key in self.key_plans
            ),
        }


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShearGeneratorKeyPlan:
    encoded_rank_pairs: np.ndarray
    dense_axis_permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShearGeneratorPlan:
    output_grade: Bidegree
    source_prime_grade: Bidegree | None
    doubleprime_source_ranks: np.ndarray
    doubleprime_output_ranks: np.ndarray
    prime_key_plans: tuple[BigradedShearGeneratorKeyPlan, ...]

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "rank_pairs": int(
                self.doubleprime_source_ranks.nbytes
                + self.doubleprime_output_ranks.nbytes
                + sum(key.encoded_rank_pairs.nbytes for key in self.prime_key_plans)
            ),
            "dense_permutations": int(
                sum(
                    len(key.dense_axis_permutation) * np.dtype(np.intp).itemsize
                    for key in self.prime_key_plans
                )
            ),
        }


class BigradedShearPlanStore:
    """Mandatory transform and native generator plans for one capacity."""

    @symbolic_plan_compilation_scope()
    def __init__(self, plan_store: BigradedPlanStore) -> None:
        if not isinstance(plan_store, BigradedPlanStore):
            raise TypeError("plan_store must be a BigradedPlanStore.")
        self.plan_store = plan_store
        self.dims = plan_store.dims
        self.max_truncation = plan_store.max_truncation
        self._forward = {}
        self._inverse = {}
        self._generators = {}
        generator_entry_count = sum(
            _generator_prime_term_count(grade)
            + (
                comb(sum(grade) - 1, grade[0])
                if grade[1] and grade != (0, 0)
                else 0
            )
            for grade in plan_store.grade_plans
            if grade != (0, 0)
        )
        self._use_compiled_generator_builder = (
            generator_entry_count
            >= _COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD
        )
        for grade in plan_store.grade_plans:
            meta = plan_store.grade_plan(grade)
            parities = _readonly(
                np.asarray(
                    [
                        -1 if sum(map(int, placement)) % 2 else 1
                        for placement in meta.placements
                    ],
                    dtype=np.int8,
                )
            )
            self._forward[grade] = self._build_transform(
                grade, inverse=False, parities=parities
            )
            self._inverse[grade] = self._build_transform(
                grade, inverse=True, parities=parities
            )
            if grade != (0, 0):
                self._generators[grade] = self._build_generator(grade)
        self.forward_plans = MappingProxyType(self._forward)
        self.inverse_plans = MappingProxyType(self._inverse)
        self.generator_plans = MappingProxyType(self._generators)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _build_transform(
        self,
        grade: Bidegree,
        *,
        inverse: bool,
        parities: np.ndarray,
    ) -> BigradedShearTransformPlan:
        n, m = grade
        meta = self.plan_store.grade_plan(grade)
        if n + m <= 64:
            support = (
                compile_inverse_shear_support(n, m)
                if inverse
                else compile_forward_shear_support(n, m)
            )
            placement_count = comb(n + m, n)
            encoded_pairs = shear_colex_rank_pairs(support)
            grouped_items = (
                (
                    _packed_transform_permutation(
                        n,
                        tuple(map(int, permutation)),
                    ),
                    encoded_pairs[support.group_slice(group)],
                )
                for group, permutation in enumerate(
                    support.doubleprime_permutations
                )
            )
        else:
            placement_count, grouped = _transform_rank_groups(
                grade, inverse=inverse
            )
            grouped_items = grouped.items()
        if placement_count != meta.placement_count:
            raise AssertionError("bidegree placement-count mismatch")
        maximum = max(meta.placement_count**2 - 1, 0)
        dtype = _unsigned_index_dtype(maximum)
        keys = []
        for permutation, pairs in grouped_items:
            encoded = _readonly(np.asarray(pairs, dtype=dtype))
            # Dense rank matrices are a derived execution form.  Retain one
            # only when its exact int8 payload is competitive with the paired
            # index list; sparse higher-grade fibres stay gather/scatter based.
            matrix_bytes = meta.placement_count**2 * np.dtype(np.int8).itemsize
            rank_matrix = None
            if meta.placement_count <= 64 and matrix_bytes <= 2 * encoded.nbytes:
                rank_matrix = np.zeros(
                    (meta.placement_count, meta.placement_count), dtype=np.int8
                )
                for encoded_pair in encoded:
                    output_rank = int(encoded_pair // meta.placement_count)
                    input_rank = int(encoded_pair % meta.placement_count)
                    coefficient = (
                        int(parities[output_rank]) * int(parities[input_rank])
                        if inverse
                        else 1
                    )
                    rank_matrix[output_rank, input_rank] += coefficient
                rank_matrix = _readonly(rank_matrix)
            keys.append(
                BigradedShearTransformKeyPlan(
                    encoded_rank_pairs=encoded,
                    dense_axis_permutation=permutation,
                    rank_matrix=rank_matrix,
                )
            )
        return BigradedShearTransformPlan(
            grade=grade,
            placement_count=meta.placement_count,
            dense_shape=meta.dense_shape,
            signed=inverse,
            placement_parities=parities,
            key_plans=tuple(keys),
        )

    def _build_generator(self, grade: Bidegree) -> BigradedShearGeneratorPlan:
        output_meta = self.plan_store.grade_plan(grade)
        support = compile_bigraded_shear_generator_support(
            output_meta.placements,
            grade,
            compiled=self._use_compiled_generator_builder,
        )
        if support.output_count != output_meta.placement_count:
            raise AssertionError("bidegree generator placement-count mismatch")
        prime_source_grade = (grade[0] - 1, grade[1]) if grade[0] else None
        if prime_source_grade is not None and support.prime_source_count != (
            self.plan_store.grade_plan(prime_source_grade).placement_count
        ):
            raise AssertionError(
                "bidegree generator source placement-count mismatch"
            )
        return BigradedShearGeneratorPlan(
            output_grade=grade,
            source_prime_grade=prime_source_grade,
            doubleprime_source_ranks=support.doubleprime_source_ranks,
            doubleprime_output_ranks=support.doubleprime_output_ranks,
            prime_key_plans=tuple(
                BigradedShearGeneratorKeyPlan(
                    encoded_rank_pairs=_readonly(
                        np.array(
                            support.prime_encoded_rank_pairs[
                                support.prime_group_slice(group)
                            ],
                            copy=True,
                        )
                    ),
                    dense_axis_permutation=tuple(
                        map(int, permutation)
                    ),
                )
                for group, permutation in enumerate(
                    support.prime_dense_permutations
                )
            ),
        )

    def transform_plan(
        self, grade: object, *, inverse: bool
    ) -> BigradedShearTransformPlan:
        grade = _bidegree(grade)
        return (self._inverse if inverse else self._forward)[grade]

    def generator_plan(self, grade: object) -> BigradedShearGeneratorPlan:
        return self._generators[_bidegree(grade)]

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        categories = {
            "transform_rank_pairs": 0,
            "transform_dense_permutations": 0,
            "transform_parities": 0,
            "transform_rank_matrices": 0,
            "generator_rank_pairs": 0,
            "generator_dense_permutations": 0,
        }
        for plan in tuple(self._forward.values()) + tuple(self._inverse.values()):
            memory = plan.memory_bytes_by_category()
            categories["transform_rank_pairs"] += memory["rank_pairs"]
            categories["transform_dense_permutations"] += memory["dense_permutations"]
            categories["transform_rank_matrices"] += memory["rank_matrices"]
        # Forward and inverse plans reference the same immutable parity vector
        # for each grade; count that authoritative payload exactly once.
        categories["transform_parities"] = sum(
            plan.placement_parities.nbytes for plan in self._forward.values()
        )
        for plan in self._generators.values():
            memory = plan.memory_bytes_by_category()
            categories["generator_rank_pairs"] += memory["rank_pairs"]
            categories["generator_dense_permutations"] += memory["dense_permutations"]
        return categories

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def plan_statistics(self) -> Mapping[str, object]:
        categories = self.memory_bytes_by_category()
        forward_terms = sum(
            key.encoded_rank_pairs.size
            for plan in self._forward.values()
            for key in plan.key_plans
        )
        inverse_terms = sum(
            key.encoded_rank_pairs.size
            for plan in self._inverse.values()
            for key in plan.key_plans
        )
        generator_terms = sum(
            plan.doubleprime_output_ranks.size
            + sum(key.encoded_rank_pairs.size for key in plan.prime_key_plans)
            for plan in self._generators.values()
        )
        forward_groups = sum(
            len(plan.key_plans) for plan in self._forward.values()
        )
        inverse_groups = sum(
            len(plan.key_plans) for plan in self._inverse.values()
        )
        generator_groups = sum(
            len(plan.prime_key_plans)
            + (plan.doubleprime_output_ranks.size > 0)
            for plan in self._generators.values()
        )
        rank_matrix_count = sum(
            key.rank_matrix is not None
            for plan in tuple(self._forward.values()) + tuple(self._inverse.values())
            for key in plan.key_plans
        )
        transform_group_count = forward_groups + inverse_groups
        derived_bytes = categories["transform_rank_matrices"]
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "plan_counts": {
                "forward": len(self._forward),
                "inverse": len(self._inverse),
                "generator": len(self._generators),
            },
            "term_counts": {
                "forward": int(forward_terms),
                "inverse": int(inverse_terms),
                "generator": int(generator_terms),
            },
            "permutation_group_counts": {
                "forward": forward_groups,
                "inverse": inverse_groups,
                "generator": generator_groups,
            },
            "strategy_counts": {
                "rank_matrix": rank_matrix_count,
                "rank_gather_scatter": transform_group_count - rank_matrix_count,
                "generator_rank_action": generator_groups,
            },
            "forward_grade_count": len(self._forward),
            "inverse_grade_count": len(self._inverse),
            "generator_grade_count": len(self._generators),
            "transform_key_count": transform_group_count,
            "rank_matrix_count": rank_matrix_count,
            "authoritative_memory_bytes": sum(categories.values()) - derived_bytes,
            "derived_execution_memory_bytes": derived_bytes,
            "bytes_by_category": categories,
            "memory_bytes": sum(categories.values()),
        }


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShearShuffleKeyPlan:
    encoded_output_pair_indices: np.ndarray
    dense_axis_permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShearShuffleBlockPlan:
    left_grade: Bidegree
    right_grade: Bidegree
    output_grade: Bidegree
    left_placement_count: int
    right_placement_count: int
    output_placement_count: int
    left_dense_shape: tuple[int, ...]
    right_dense_shape: tuple[int, ...]
    output_dense_shape: tuple[int, ...]
    key_plans: tuple[BigradedShearShuffleKeyPlan, ...]


class BigradedShearShufflePlanStore:
    """Optional shear shuffle plans sharing the base placement store."""

    @symbolic_plan_compilation_scope()
    def __init__(self, plan_store: BigradedPlanStore, scope: str = "full") -> None:
        if not isinstance(plan_store, BigradedPlanStore):
            raise TypeError("plan_store must be a BigradedPlanStore.")
        self.plan_store = plan_store
        self.scope = _shuffle_scope(scope)
        self.dims = plan_store.dims
        self.max_truncation = plan_store.max_truncation
        grades = tuple(plan_store.grade_plans)
        self._grade_order = {grade: index for index, grade in enumerate(grades)}
        self._block_plans = {}
        grade_pairs = []
        for left_index, left_grade in enumerate(grades):
            for right_grade in grades[: left_index + 1]:
                output = (
                    left_grade[0] + right_grade[0],
                    left_grade[1] + right_grade[1],
                )
                if (
                    _pair_in_scope(left_grade, right_grade, self.scope)
                    and output in plan_store.grade_plans
                ):
                    grade_pairs.append((left_grade, right_grade))
        candidate_entry_count = sum(
            _shear_shuffle_group_count(left_grade, right_grade)
            * comb(
                sum(left_grade) + sum(right_grade),
                left_grade[0] + right_grade[0],
            )
            for left_grade, right_grade in grade_pairs
        )
        self._use_compiled_plan_builder = (
            candidate_entry_count
            >= _COMPILED_SHEAR_SHUFFLE_ENTRY_THRESHOLD
        )
        workspace = _ShearShuffleWorkspace()
        for left_grade, right_grade in grade_pairs:
            self._block_plans[left_grade, right_grade] = self._build(
                left_grade,
                right_grade,
                workspace=workspace,
            )
        self.block_plans = MappingProxyType(self._block_plans)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _build(
        self,
        left_grade,
        right_grade,
        *,
        workspace: _ShearShuffleWorkspace,
    ) -> BigradedShearShuffleBlockPlan:
        n1, m1 = left_grade
        n2, m2 = right_grade
        left = self.plan_store.grade_plan(left_grade)
        right = self.plan_store.grade_plan(right_grade)
        output_grade = n1 + n2, m1 + m2
        if sum(output_grade) <= _MAX_MASK_DEGREE:
            support = compile_shear_shuffle_support(
                n1,
                m1,
                n2,
                m2,
                compiled=self._use_compiled_plan_builder,
                _workspace=workspace,
            )
            output_count = support.output_placement_count
            pair_count = support.pair_count
            encoded = shear_shuffle_colex_output_pair_indices(
                support,
                compiled=self._use_compiled_plan_builder,
                _workspace=workspace,
            )
            dense_permutations = shear_shuffle_dense_axis_permutations(
                support,
                compiled=self._use_compiled_plan_builder,
            )
            grouped_items = tuple(
                (
                    tuple(map(int, dense_permutations[group])),
                    encoded[support.group_slice(group)],
                )
                for group in range(support.group_count)
            )
        else:
            (
                counted_output_grade,
                output_count,
                pair_count,
                grouped,
            ) = _gamma_rank_groups(left_grade, right_grade)
            if counted_output_grade != output_grade:
                raise AssertionError("shear shuffle output-grade mismatch")
            grouped_items = tuple(grouped.items())
        output = self.plan_store.grade_plan(output_grade)
        if pair_count != left.placement_count * right.placement_count:
            raise AssertionError("shear shuffle pair-count mismatch")
        if output_count != output.placement_count:
            raise AssertionError("shear shuffle output placement-count mismatch")
        encoded_dtype = _unsigned_index_dtype(
            max(output.placement_count * pair_count - 1, 0)
        )
        dp, dd = self.dims
        return BigradedShearShuffleBlockPlan(
            left_grade=left_grade,
            right_grade=right_grade,
            output_grade=output_grade,
            left_placement_count=left.placement_count,
            right_placement_count=right.placement_count,
            output_placement_count=output.placement_count,
            left_dense_shape=(dp,) * n1 + (dd,) * m1,
            right_dense_shape=(dp,) * n2 + (dd,) * m2,
            output_dense_shape=(dp,) * (n1 + n2) + (dd,) * (m1 + m2),
            key_plans=tuple(
                BigradedShearShuffleKeyPlan(
                    encoded_output_pair_indices=_readonly(
                        np.asarray(values, dtype=encoded_dtype)
                    ),
                    dense_axis_permutation=permutation,
                )
                for permutation, values in grouped_items
            ),
        )

    def resolve_block_plan(self, left_grade, right_grade):
        left_grade = _bidegree(left_grade, name="left_grade")
        right_grade = _bidegree(right_grade, name="right_grade")
        swap = self._grade_order[left_grade] < self._grade_order[right_grade]
        key = (right_grade, left_grade) if swap else (left_grade, right_grade)
        try:
            return self._block_plans[key], swap
        except KeyError as error:
            if not _pair_in_scope(left_grade, right_grade, self.scope):
                raise KeyError(
                    f"shear shuffle pair {(left_grade, right_grade)} is unavailable "
                    f"in scope={self.scope!r}."
                ) from error
            raise KeyError(
                f"shear shuffle output exceeds capacity {self.max_truncation}."
            ) from error

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "rank_lists": int(
                sum(
                    key.encoded_output_pair_indices.nbytes
                    for plan in self._block_plans.values()
                    for key in plan.key_plans
                )
            ),
            "dense_permutations": int(
                sum(
                    len(key.dense_axis_permutation) * np.dtype(np.intp).itemsize
                    for plan in self._block_plans.values()
                    for key in plan.key_plans
                )
            ),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        categories = self.memory_bytes_by_category()
        term_count = sum(
            key.encoded_output_pair_indices.size
            for plan in self._block_plans.values()
            for key in plan.key_plans
        )
        group_count = sum(
            len(plan.key_plans) for plan in self._block_plans.values()
        )
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "scope": self.scope,
            "block_plan_count": len(self._block_plans),
            "term_count": int(term_count),
            "permutation_group_count": group_count,
            "strategy_counts": {"rank_gather_scatter": group_count},
            "key_plan_count": group_count,
            "authoritative_memory_bytes": sum(categories.values()),
            "derived_execution_memory_bytes": 0,
            "bytes_by_category": categories,
            "memory_bytes": sum(categories.values()),
            "memory_mb": sum(categories.values()) / 1024**2,
        }


def _expected_bigraded_shear_plan_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
) -> Mapping[str, int]:
    """Count mandatory bidegree-shear buffers without allocating arrays."""
    spec = BigradedSpec(*dims, max_truncation)
    categories = {
        "transform_rank_pairs": 0,
        "transform_dense_permutations": 0,
        "transform_parities": 0,
        "transform_rank_matrices": 0,
        "generator_rank_pairs": 0,
        "generator_dense_permutations": 0,
    }
    index_pointer_bytes = np.dtype(np.intp).itemsize
    for grade in spec.grades:
        n, m = grade
        degree = sum(grade)
        placement_count = comb(degree, n)
        categories["transform_parities"] += (
            placement_count * np.dtype(np.int8).itemsize
        )
        transform_itemsize = np.dtype(
            _unsigned_index_dtype(max(placement_count**2 - 1, 0))
        ).itemsize
        for inverse in (False, True):
            term_counter = (
                _inverse_term_count if inverse else _forward_term_count
            )
            group_counter = (
                _inverse_group_count if inverse else _forward_group_count
            )
            term_count = term_counter(n, m)
            permutation_count = group_counter(n, m)
            categories["transform_rank_pairs"] += (
                term_count * transform_itemsize
            )
            categories["transform_dense_permutations"] += (
                permutation_count * degree * index_pointer_bytes
            )
            if placement_count <= 64:
                matrix_bytes = (
                    placement_count**2 * np.dtype(np.int8).itemsize
                )
                qualifying_groups = _transform_rank_matrix_group_count(
                    grade,
                    inverse=inverse,
                    placement_count=placement_count,
                    transform_itemsize=transform_itemsize,
                )
                categories["transform_rank_matrices"] += (
                    qualifying_groups * matrix_bytes
                )

        if grade == (0, 0):
            continue
        output_count = placement_count
        prime_source_count = comb(degree - 1, n - 1) if n else 1
        doubleprime_term_count = comb(degree - 1, n) if m else 0
        prime_term_count = _generator_prime_term_count(grade)
        prime_permutation_count = (1 << m) - m if n else 0
        rank_itemsize = np.dtype(
            _unsigned_index_dtype(max(output_count - 1, 0))
        ).itemsize
        pair_itemsize = np.dtype(
            _unsigned_index_dtype(
                max(output_count * prime_source_count - 1, 0)
            )
        ).itemsize
        categories["generator_rank_pairs"] += (
            2 * doubleprime_term_count * rank_itemsize
            + prime_term_count * pair_itemsize
        )
        categories["generator_dense_permutations"] += (
            prime_permutation_count * degree * index_pointer_bytes
        )
    return MappingProxyType(categories)


def _expected_bigraded_gamma_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
    *,
    scope: str,
) -> Mapping[str, int]:
    """Count optional shear shuffle buffers without allocating arrays."""
    scope = _shuffle_scope(scope)
    spec = BigradedSpec(*dims, max_truncation)
    grades = spec.grades
    rank_lists = 0
    dense_permutations = 0
    for left_index, left_grade in enumerate(grades):
        for right_grade in grades[: left_index + 1]:
            output_grade = (
                left_grade[0] + right_grade[0],
                left_grade[1] + right_grade[1],
            )
            if (
                not _pair_in_scope(left_grade, right_grade, scope)
                or not spec.contains(output_grade)
            ):
                continue
            output_count = comb(sum(output_grade), output_grade[0])
            pair_count = (
                comb(sum(left_grade), left_grade[0])
                * comb(sum(right_grade), right_grade[0])
            )
            itemsize = np.dtype(
                _unsigned_index_dtype(max(output_count * pair_count - 1, 0))
            ).itemsize
            rank_lists += (
                _shear_shuffle_term_count(left_grade, right_grade)
                * itemsize
            )
            dense_permutations += (
                _shear_shuffle_group_count(left_grade, right_grade)
                * sum(output_grade)
                * np.dtype(np.intp).itemsize
            )
    return MappingProxyType(
        {
            "rank_lists": int(rank_lists),
            "dense_permutations": int(dense_permutations),
        }
    )


@symbolic_plan_compilation_scope()
def _expected_shear_bigraded_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
    *,
    precompute_shuffle: bool | Literal["generator"] = False,
) -> Mapping[str, int]:
    """Exact full bidegree-shear payload without constructing plan stores."""
    spec = BigradedSpec(*dims, max_truncation)
    scope = _normalize_precompute_shuffle(
        precompute_shuffle, allow_generator=True
    )
    categories = dict(
        _expected_plan_memory_bytes_by_category(spec.dims, spec.truncation)
    )
    categories.update(
        {
            f"shear_{name}": size
            for name, size in (
                _expected_bigraded_shear_plan_memory_bytes_by_category(
                    spec.dims, spec.truncation
                )
            ).items()
        }
    )
    if scope != "none":
        categories.update(
            {
                f"shuffle_{name}": size
                for name, size in (
                    _expected_bigraded_gamma_memory_bytes_by_category(
                        spec.dims,
                        spec.truncation,
                        scope=scope,
                    )
                ).items()
            }
        )
    return MappingProxyType(categories)


class ShearBigradedCore(BigradedShearCoordinateCore, StandardBigradedCore):
    """Backend-neutral ordered-bidegree shear algebra."""

    coordinates = "shear"

    def _validate_ordered_signature_pairing_block(
        self, block, grade, *, name: str
    ):
        grade = _bidegree(grade, name="grade")
        self._normalize_truncation(grade)
        block = self.xp.asarray(block)
        expected = self.plan_store.grade_plan(grade).block_width
        if block.ndim == 0 or block.shape[-1] != expected:
            width = None if block.ndim == 0 else block.shape[-1]
            raise ValueError(
                f"{name} block at bidegree {grade} has width {width}, "
                f"expected {expected}."
            )
        return block

    def _prepare_ordered_signature_pairing_operands(
        self,
        standard_words,
        ordered_standard_signature,
        *,
        words_first_on: bool,
        standard_first_on: bool,
    ):
        standard_words = self._validate_element_coordinates(
            standard_words,
            name="words",
            coordinates="standard",
        )
        ordered_standard_signature = self._validate_element_coordinates(
            ordered_standard_signature,
            name="standard_tensor",
            coordinates="standard",
        )
        del words_first_on
        if (
            standard_first_on
            == ordered_standard_signature.spec.include_scalar
        ):
            expected = "omit" if standard_first_on else "include"
            raise ValueError(
                f"standard_first_on={standard_first_on} requires its tensor "
                f"to {expected} the scalar block."
        )
        return standard_words, ordered_standard_signature

    @symbolic_plan_compilation_scope()
    def __init__(
        self,
        xp: Any,
        *,
        dims: Bidegree | None = None,
        max_trunc: Bidegree | None = None,
        default_trunc: Bidegree | None = None,
        plan_store: BigradedPlanStore | None = None,
        shear_plan_store: BigradedShearPlanStore | None = None,
        shuffle_plan_store: BigradedShearShufflePlanStore | None = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        scope = _normalize_precompute_shuffle(
            precompute_shuffle, allow_generator=True
        )
        StandardBigradedCore.__init__(
            self,
            xp,
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            plan_store=plan_store,
            precompute_shuffle=False,
        )
        if shear_plan_store is None:
            shear_plan_store = BigradedShearPlanStore(self.plan_store)
        else:
            if not isinstance(shear_plan_store, BigradedShearPlanStore):
                raise TypeError(
                    "shear_plan_store must be a BigradedShearPlanStore, got "
                    f"{type(shear_plan_store).__name__}."
                )
            if shear_plan_store.plan_store is not self.plan_store:
                raise ValueError(
                    "shear_plan_store must share this core's plan_store."
                )
        self.shear_plan_store = shear_plan_store
        if shuffle_plan_store is not None:
            if not isinstance(
                shuffle_plan_store,
                BigradedShearShufflePlanStore,
            ):
                raise TypeError(
                    "shuffle_plan_store must be a compatible bidegree shear "
                    f"shuffle plan store, got {type(shuffle_plan_store).__name__}."
                )
            if scope != "none":
                raise ValueError(
                    "precompute_shuffle and shuffle_plan_store are mutually exclusive."
                )
            if shuffle_plan_store.plan_store is not self.plan_store:
                raise ValueError(
                    "shuffle_plan_store must share this core's plan_store."
                )
            self.shuffle_plan_store = shuffle_plan_store
        elif scope != "none":
            self.shuffle_plan_store = BigradedShearShufflePlanStore(
                self.plan_store, scope=scope
            )
        else:
            self.shuffle_plan_store = None

    def __repr__(self) -> str:
        scope = (
            "none"
            if self.shuffle_plan_store is None
            else self.shuffle_plan_store.scope
        )
        precompute_shuffle = _precompute_shuffle_argument(scope)
        return (
            f"{type(self).__name__}(dims={self.dims}, coordinates='shear', "
            f"max_trunc={self.max_truncation}, "
            f"default_trunc={self.default_truncation}, "
            f"precompute_shuffle={precompute_shuffle!r})"
        )

    def at_truncation(self, trunc: Bidegree):
        active = self._normalize_truncation(trunc)
        cached = self._truncation_views.get(active)
        if cached is not None:
            return cached
        view = type(self)(
            plan_store=self.plan_store,
            shear_plan_store=self.shear_plan_store,
            shuffle_plan_store=self.shuffle_plan_store,
            default_trunc=active,
        )
        view._truncation_views = self._truncation_views
        self._truncation_views[active] = view
        return view

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        categories = dict(self.plan_store.memory_bytes_by_category())
        categories.update(
            {
                f"shear_{name}": value
                for name, value in (
                    self.shear_plan_store.memory_bytes_by_category().items()
                )
            }
        )
        if self.shuffle_plan_store is not None:
            categories.update(
                {
                    f"shuffle_{name}": value
                    for name, value in (
                        self.shuffle_plan_store.memory_bytes_by_category().items()
                    )
                }
            )
        return categories

    def plan_statistics(self) -> Mapping[str, object]:
        categories = self.memory_bytes_by_category()
        shear_statistics = self.shear_plan_store.plan_statistics()
        shuffle_statistics = (
            None
            if self.shuffle_plan_store is None
            else self.shuffle_plan_store.plan_statistics()
        )
        derived_bytes = categories.get("shear_transform_rank_matrices", 0)
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "default_truncation": self.default_truncation,
            "grading": self.grading,
            "coordinates": self.coordinates,
            "shuffle_enabled": self.shuffle_plan_store is not None,
            "shuffle_scope": (
                "none"
                if self.shuffle_plan_store is None
                else self.shuffle_plan_store.scope
            ),
            "block_plan_statistics": self.plan_store.plan_statistics(),
            "shear_plan_statistics": shear_statistics,
            "shuffle_plan_statistics": shuffle_statistics,
            "authoritative_memory_bytes": sum(categories.values()) - derived_bytes,
            "derived_execution_memory_bytes": derived_bytes,
            "bytes_by_category": categories,
            "memory_bytes": sum(categories.values()),
            "memory_mb": sum(categories.values()) / 1024**2,
        }

    def _apply_transform_block(self, block, grade, *, inverse, transpose):
        plan = self.shear_plan_store.transform_plan(grade, inverse=inverse)
        n, m = plan.grade
        batch = block.shape[:-1]
        expected = plan.placement_count * plan.dense_shape[1] * plan.dense_shape[2]
        if block.shape[-1] != expected:
            raise ValueError(
                f"block {grade} has width {block.shape[-1]}, expected {expected}."
            )
        source = block.reshape(
            batch + (plan.placement_count,) + (self.dims[0],) * n + (self.dims[1],) * m
        )
        output = self.xp.zeros(batch + plan.dense_shape, dtype=block.dtype)
        dense_degree = n + m
        parity = self.xp.asarray(plan.placement_parities)
        for key in plan.key_plans:
            output_ranks, input_ranks = key.rank_pairs(plan.placement_count)
            permutation = key.dense_axis_permutation
            if transpose:
                output_ranks, input_ranks = input_ranks, output_ranks
                permutation = invert_permutation(permutation)
            if key.rank_matrix is not None:
                values = source
                if permutation != tuple(range(dense_degree)):
                    prefix_ndim = len(batch) + 1
                    values = self.xp.transpose(
                        values,
                        tuple(range(prefix_ndim))
                        + tuple(prefix_ndim + axis for axis in permutation),
                    )
                values = values.reshape(
                    batch
                    + (
                        plan.placement_count,
                        plan.dense_shape[1] * plan.dense_shape[2],
                    )
                )
                matrix = self.xp.asarray(
                    key.rank_matrix.T if transpose else key.rank_matrix,
                    dtype=block.dtype,
                )
                values = self.xp.einsum("oi,...id->...od", matrix, values)
                output = output + values.reshape(batch + plan.dense_shape)
                continue
            values = self.xp.take(source, self.xp.asarray(input_ranks), axis=len(batch))
            if permutation != tuple(range(dense_degree)):
                prefix_ndim = len(batch) + 1
                values = self.xp.transpose(
                    values,
                    tuple(range(prefix_ndim))
                    + tuple(prefix_ndim + axis for axis in permutation),
                )
            values = values.reshape(
                batch + (len(input_ranks), plan.dense_shape[1], plan.dense_shape[2])
            )
            if plan.signed:
                signs = parity[self.xp.asarray(output_ranks)] * parity[
                    self.xp.asarray(input_ranks)
                ]
                values = values * signs.reshape((1,) * len(batch) + (-1, 1, 1))
            output = self._placement_scatter_add(
                output, self.xp.asarray(output_ranks), values
            )
        return output.reshape(batch + (expected,))

    def _coordinate_forward_block(self, block, grade):
        return self._apply_transform_block(
            block, grade, inverse=False, transpose=False
        )

    def _coordinate_inverse_block(self, block, grade):
        return self._apply_transform_block(
            block, grade, inverse=True, transpose=False
        )

    def _coordinate_forward_transpose_block(self, block, grade):
        return self._apply_transform_block(
            block, grade, inverse=False, transpose=True
        )

    def _coordinate_inverse_transpose_block(self, block, grade):
        return self._apply_transform_block(
            block, grade, inverse=True, transpose=True
        )

    def _right_multiply_generator_output_block(
        self,
        predecessor_blocks,
        generator_blocks,
        *,
        predecessor_grades,
        generator_grades,
        output_grade,
    ):
        predecessor_blocks, generator_blocks = self._validate_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )
        output_grade = _bidegree(output_grade, name="output_grade")
        supplied = {
            (_bidegree(sg), _bidegree(gg)): (source, generator)
            for source, generator, sg, gg in zip(
                predecessor_blocks,
                generator_blocks,
                predecessor_grades,
                generator_grades,
            )
        }
        if len(supplied) != len(predecessor_blocks):
            raise ValueError("generator action contains duplicate grade pairs.")
        n, m = output_grade
        expected = set()
        if m:
            expected.add(((n, m - 1), (0, 1)))
        if n:
            expected.add(((n - 1, m), (1, 0)))
        if set(supplied) != expected:
            raise ValueError(
                "generator inputs are not the predecessors of output_grade."
            )
        plan = self.shear_plan_store.generator_plan(output_grade)
        batch = self.xp.broadcast_shapes(
            *(array.shape[:-1] for pair in supplied.values() for array in pair)
        )
        output_meta = self.plan_store.grade_plan(output_grade)
        output = self.xp.zeros(
            batch + output_meta.dense_shape,
            dtype=self.xp.result_type(
                *(array.dtype for pair in supplied.values() for array in pair)
            ),
        )
        if m:
            source, generator = supplied[((n, m - 1), (0, 1))]
            source_meta = self.plan_store.grade_plan((n, m - 1))
            appended = self._append_doubleprime(
                self.xp.broadcast_to(source, batch + (source_meta.block_width,)),
                self.xp.broadcast_to(generator, batch + (self.dims[1],)),
                output_grade,
            )
            values = self.xp.take(
                appended,
                self.xp.asarray(plan.doubleprime_source_ranks),
                axis=-3,
            )
            output = self._placement_scatter_add(
                output,
                self.xp.asarray(plan.doubleprime_output_ranks),
                values,
            )
        if n:
            source, generator = supplied[((n - 1, m), (1, 0))]
            source_meta = self.plan_store.grade_plan((n - 1, m))
            source = self.xp.broadcast_to(
                source,
                batch + (source_meta.block_width,),
            ).reshape(
                batch
                + (source_meta.placement_count,)
                + (self.dims[0],) * (n - 1)
                + (self.dims[1],) * m
            )
            generator = self.xp.broadcast_to(generator, batch + (self.dims[0],))
            source_count = source_meta.placement_count
            for key in plan.prime_key_plans:
                encoded = key.encoded_rank_pairs
                output_ranks = encoded // source_count
                source_ranks = encoded % source_count
                selected = self.xp.take(
                    source, self.xp.asarray(source_ranks), axis=len(batch)
                )
                raw = selected[..., None] * generator.reshape(
                    batch + (1,) * (n + m) + (self.dims[0],)
                )
                prefix_ndim = len(batch) + 1
                values = self.xp.transpose(
                    raw,
                    tuple(range(prefix_ndim))
                    + tuple(
                        prefix_ndim + axis for axis in key.dense_axis_permutation
                    ),
                ).reshape(
                    batch
                    + (
                        len(source_ranks),
                        output_meta.dense_shape[1],
                        output_meta.dense_shape[2],
                    )
                )
                output = self._placement_scatter_add(
                    output, self.xp.asarray(output_ranks), values
                )
        return output.reshape(batch + (output_meta.block_width,))

    def _require_shuffle(self) -> BigradedShearShufflePlanStore:
        if self.shuffle_plan_store is None:
            raise RuntimeError(
                "This shear core has no shuffle plans; construct it with "
                "precompute_shuffle='generator' or True."
            )
        return self.shuffle_plan_store

    def _require_full_shuffle(self) -> BigradedShearShufflePlanStore:
        store = self._require_shuffle()
        if store.scope != "full":
            raise RuntimeError(
                "Arbitrary shear shuffle products require "
                "precompute_shuffle=True."
            )
        return store

    def _gamma_shuffle_block(self, left, right, left_grade, right_grade, output_grade):
        plan, swap = self._require_shuffle().resolve_block_plan(
            left_grade, right_grade
        )
        if plan.output_grade != output_grade:
            raise ValueError("shear shuffle plan/output grade mismatch.")
        if swap:
            left, right = right, left
        outer = _placement_pair_outer(self.xp, left, right, plan)
        degree = sum(output_grade)
        pair_count = plan.left_placement_count * plan.right_placement_count
        batch = self.xp.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        dp, dd = self.dims
        output = self.xp.zeros(
            batch
            + (
                plan.output_placement_count,
                dp ** output_grade[0],
                dd ** output_grade[1],
            ),
            dtype=self.xp.result_type(left, right),
        )
        for key in plan.key_plans:
            encoded = key.encoded_output_pair_indices
            output_ranks = encoded // pair_count
            pair_indices = encoded % pair_count
            values = self.xp.take(
                outer, self.xp.asarray(pair_indices), axis=-(degree + 1)
            )
            if key.dense_axis_permutation != tuple(range(degree)):
                prefix_ndim = len(batch) + 1
                values = self.xp.transpose(
                    values,
                    tuple(range(prefix_ndim))
                    + tuple(
                        prefix_ndim + axis for axis in key.dense_axis_permutation
                    ),
                )
            values = values.reshape(
                batch
                + (
                    len(pair_indices),
                    dp ** output_grade[0],
                    dd ** output_grade[1],
                )
            )
            output = self._placement_scatter_add(
                output, self.xp.asarray(output_ranks), values
            )
        return output.reshape(
            batch
            + (
                plan.output_placement_count
                * dp ** output_grade[0]
                * dd ** output_grade[1],
            )
        )

    def _shuffle_generator_output_block(
        self,
        predecessor_blocks,
        generator_blocks,
        *,
        predecessor_grades,
        generator_grades,
        output_grade,
    ):
        self._require_shuffle()
        predecessor_blocks, generator_blocks = self._validate_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )
        supplied = tuple(
            (_bidegree(source_grade), _bidegree(generator_grade))
            for source_grade, generator_grade in zip(
                predecessor_grades, generator_grades
            )
        )
        output_grade = _bidegree(output_grade)
        expected = {
            (source_grade, generator_grade)
            for source_grade, generator_grade in self.resolve_layout(
                output_grade, include_scalar=True
            ).product_splits(output_grade)
            if sum(generator_grade) == 1
        }
        if len(set(supplied)) != len(supplied) or set(supplied) != expected:
            raise ValueError(
                "shuffle-generator inputs are not the first-level splits of "
                f"output grade {output_grade}."
            )
        terms = tuple(
            self._gamma_shuffle_block(
                source,
                generator,
                _bidegree(source_grade),
                _bidegree(generator_grade),
                output_grade,
            )
            for source, generator, source_grade, generator_grade in zip(
                predecessor_blocks,
                generator_blocks,
                predecessor_grades,
                generator_grades,
            )
        )
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result


class JaxShearBigraded(_JaxBigradedBackend, ShearBigradedCore):
    """JAX ordered-bidegree shear core."""

    _wordwise_signature_protocol = _WORDWISE_SIGNATURE_PROTOCOL

    def __init__(
        self,
        *,
        dims: Bidegree | None = None,
        max_trunc: Bidegree | None = None,
        default_trunc: Bidegree | None = None,
        plan_store: BigradedPlanStore | None = None,
        shear_plan_store: BigradedShearPlanStore | None = None,
        shuffle_plan_store: BigradedShearShufflePlanStore | None = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        super().__init__(
            jnp,
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            plan_store=plan_store,
            shear_plan_store=shear_plan_store,
            shuffle_plan_store=shuffle_plan_store,
            precompute_shuffle=precompute_shuffle,
        )
        _bind_bigraded_shear_jax_methods(self)


__all__ = [
    "BigradedShearShuffleBlockPlan",
    "BigradedShearShuffleKeyPlan",
    "BigradedShearShufflePlanStore",
    "BigradedShearGeneratorPlan",
    "BigradedShearPlanStore",
    "BigradedShearTransformPlan",
    "JaxShearBigraded",
    "ShearBigradedCore",
]
