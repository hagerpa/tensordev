"""Partially symmetrized bidegree shear plans and numerical kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from types import MappingProxyType
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

from tensordev.core.capabilities import _WORDWISE_SIGNATURE_PROTOCOL
from tensordev.core.bigraded.jax_backend import (
    _JaxPartiallySymmetrizedBigradedBackend,
)
from tensordev.core.bigraded.symmetrized._compiled_generator import (
    compile_partially_symmetrized_prime_generator_support,
)
from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
)
from tensordev.core.bigraded.symmetrized.algebra import (
    PartiallySymmetrizedBigradedCore,
)
from tensordev.core.bigraded.symmetrized.bridge import (
    SymmetrizationBridgePlanStore,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
)
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
    _validated_capacity,
    apply_doubleprime_generator_prefix,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
    apply_partially_symmetrized_transform,
)
from tensordev.core.bigraded.types import Bidegree, _bidegree
from tensordev.core.shear.bigraded_transport import (
    BigradedShearCoordinateCore,
)
from tensordev.core.shear.bigraded_jax_transport import (
    _bind_bigraded_shear_jax_methods,
)
from tensordev.core.shuffle import _precompute_shuffle_argument
from tensordev.core.utils.precompute import _unsigned_index_dtype


_COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD = 25_000


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedPrimeGeneratorPlan:
    """Prime-generator source-rank ``L_(n,m)`` and coefficient arrays."""

    output_grade: Bidegree
    source_grade: Bidegree
    source_ranks: np.ndarray
    coefficients: np.ndarray
    source_rank_count: int
    output_rank_count: int
    source_dense_prime_width: int
    output_dense_prime_width: int
    d_prime: int

    def __post_init__(self) -> None:
        expected = (self.output_rank_count,)
        if self.source_ranks.shape != expected:
            raise ValueError(
                f"source_ranks has shape {self.source_ranks.shape}, "
                f"expected {expected}."
            )
        if self.coefficients.shape != expected:
            raise ValueError(
                f"coefficients has shape {self.coefficients.shape}, "
                f"expected {expected}."
            )
        if self.output_dense_prime_width != (
            self.source_dense_prime_width * self.d_prime
        ):
            raise ValueError("prime generator dense widths are inconsistent.")

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "source_ranks": int(self.source_ranks.nbytes),
            "coefficients": int(self.coefficients.nbytes),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())


def _expected_partially_symmetrized_shear_generator_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
) -> Mapping[str, int]:
    """Count partially symmetrized generator buffers without allocation."""

    normalized_dims, normalized_truncation, spec = _validated_capacity(
        dims,
        max_truncation,
    )
    del normalized_truncation
    d_doubleprime = normalized_dims[1]
    source_rank_bytes = 0
    coefficient_bytes = 0
    for n, m in spec.grades:
        if n == 0:
            continue
        output_rank_count = multiset_placement_count(
            d_doubleprime,
            (n, m),
        )
        source_rank_count = multiset_placement_count(
            d_doubleprime,
            (n - 1, m),
        )
        source_rank_bytes += output_rank_count * np.dtype(
            _unsigned_index_dtype(source_rank_count - 1)
        ).itemsize
        # The product of componentwise binomial coefficients is at most the
        # central binomial coefficient, and equality is attained by putting
        # all double-prime mass in one letter and splitting the last blocks
        # as evenly as possible.  Hence this is the exact retained dtype.
        maximum_coefficient = comb(m, m // 2)
        coefficient_bytes += output_rank_count * np.dtype(
            _coefficient_dtype(maximum_coefficient)
        ).itemsize
    return MappingProxyType(
        {
            "shear_generator_source_ranks": int(source_rank_bytes),
            "shear_generator_coefficients": int(coefficient_bytes),
        }
    )


def apply_partially_symmetrized_prime_generator_block(
    xp: Any,
    source,
    generator,
    plan: PartiallySymmetrizedPrimeGeneratorPlan,
):
    """Apply the native prime-generator contribution in output-rank order."""

    source = xp.asarray(source)
    generator = xp.asarray(generator)
    source_width = plan.source_rank_count * plan.source_dense_prime_width
    if source.ndim == 0 or source.shape[-1] != source_width:
        raise ValueError(
            f"source block has final width "
            f"{None if source.ndim == 0 else source.shape[-1]}, expected "
            f"{source_width} for grade {plan.source_grade}."
        )
    if generator.ndim == 0 or generator.shape[-1] != plan.d_prime:
        raise ValueError(
            f"prime generator has final width "
            f"{None if generator.ndim == 0 else generator.shape[-1]}, "
            f"expected {plan.d_prime}."
        )
    batch = xp.broadcast_shapes(source.shape[:-1], generator.shape[:-1])
    source = xp.broadcast_to(source, batch + (source_width,)).reshape(
        batch
        + (plan.source_rank_count, plan.source_dense_prime_width)
    )
    generator = xp.broadcast_to(generator, batch + (plan.d_prime,))
    selected = xp.take(
        source,
        xp.asarray(plan.source_ranks),
        axis=-2,
    )
    coefficients = xp.asarray(plan.coefficients).reshape(
        (1,) * len(batch) + (plan.output_rank_count, 1, 1)
    )
    values = (
        selected[..., :, :, None]
        * generator[..., None, None, :]
        * coefficients
    )
    return xp.reshape(
        values,
        batch
        + (plan.output_rank_count * plan.output_dense_prime_width,),
    )


class PartiallySymmetrizedShearGeneratorPlanStore:
    """Prime-generator plans sharing the partially symmetrized store."""

    def __init__(self, plan_store: PartiallySymmetrizedPlanStore) -> None:
        if not isinstance(plan_store, PartiallySymmetrizedPlanStore):
            raise TypeError(
                "plan_store must be a PartiallySymmetrizedPlanStore."
            )
        self.plan_store = plan_store
        self.dims = plan_store.dims
        self.max_truncation = plan_store.max_truncation
        rank_entry_count = sum(
            meta.rank_count
            for grade, meta in plan_store.grade_plans.items()
            if grade[0] > 0
        )
        self._use_compiled_plan_builder = (
            rank_entry_count
            >= _COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD
        )
        plans = {}
        for grade in plan_store.grade_plans:
            if grade[0] > 0:
                plans[grade] = self._build(grade)
        self.generator_plans = MappingProxyType(plans)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            f"max_truncation={self.max_truncation})"
        )

    def _build(
        self,
        output_grade: Bidegree,
    ) -> PartiallySymmetrizedPrimeGeneratorPlan:
        n, m = output_grade
        source_grade = n - 1, m
        output_meta = self.plan_store.grade_plan(output_grade)
        source_meta = self.plan_store.grade_plan(source_grade)
        support = compile_partially_symmetrized_prime_generator_support(
            output_meta.placements,
            output_grade,
            compiled=self._use_compiled_plan_builder,
        )
        d_prime = self.dims[0]
        return PartiallySymmetrizedPrimeGeneratorPlan(
            output_grade=output_grade,
            source_grade=source_grade,
            source_ranks=support.source_ranks,
            coefficients=support.coefficients,
            source_rank_count=source_meta.rank_count,
            output_rank_count=output_meta.rank_count,
            source_dense_prime_width=d_prime ** (n - 1),
            output_dense_prime_width=d_prime**n,
            d_prime=d_prime,
        )

    def generator_plan(
        self,
        output_grade: object,
    ) -> PartiallySymmetrizedPrimeGeneratorPlan:
        grade = _bidegree(output_grade, name="output_grade")
        try:
            return self.generator_plans[grade]
        except KeyError as exc:
            if grade in self.plan_store.grade_plans:
                raise KeyError(
                    f"output grade {grade} has no prime predecessor."
                ) from exc
            raise KeyError(
                f"output grade {grade} exceeds generator-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "shear_generator_source_ranks": int(
                sum(
                    plan.source_ranks.nbytes
                    for plan in self.generator_plans.values()
                )
            ),
            "shear_generator_coefficients": int(
                sum(
                    plan.coefficients.nbytes
                    for plan in self.generator_plans.values()
                )
            ),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        memory = self.memory_bytes_by_category()
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "generator_plan_count": len(self.generator_plans),
            "rank_entry_count": sum(
                plan.output_rank_count
                for plan in self.generator_plans.values()
            ),
            "bytes_by_category": memory,
            "memory_bytes": sum(memory.values()),
            "memory_mb": sum(memory.values()) / 1024**2,
        }


class PartiallySymmetrizedShearBigradedCore(
    BigradedShearCoordinateCore,
    PartiallySymmetrizedBigradedCore,
):
    """Partially symmetrized bidegree algebra in shear coordinates."""

    coordinates = "shear"

    def __init__(
        self,
        xp: Any,
        *,
        dims: Bidegree | None = None,
        max_trunc: Bidegree | None = None,
        default_trunc: Bidegree | None = None,
        plan_store: PartiallySymmetrizedPlanStore | None = None,
        bridge_plan_store: SymmetrizationBridgePlanStore | None = None,
        shear_plan_store: PartiallySymmetrizedShearPlanStore | None = None,
        generator_plan_store: (
            PartiallySymmetrizedShearGeneratorPlanStore | None
        ) = None,
        shuffle_plan_store: (
            PartiallySymmetrizedShearShufflePlanStore | None
        ) = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        if plan_store is None:
            if dims is None or max_trunc is None:
                raise TypeError(
                    "dims and max_trunc are required for a partially "
                    "symmetrized shear core."
                )
            plan_store = PartiallySymmetrizedPlanStore(dims, max_trunc)
        if shear_plan_store is None:
            shear_plan_store = PartiallySymmetrizedShearPlanStore(plan_store)
        elif not isinstance(
            shear_plan_store,
            PartiallySymmetrizedShearPlanStore,
        ):
            raise TypeError(
                "shear_plan_store must be a "
                "PartiallySymmetrizedShearPlanStore."
            )
        elif shear_plan_store.plan_store is not plan_store:
            raise ValueError(
                "shear_plan_store must share this core's plan_store."
            )

        PartiallySymmetrizedBigradedCore.__init__(
            self,
            xp,
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            plan_store=plan_store,
            bridge_plan_store=bridge_plan_store,
            shear_plan_store=shear_plan_store,
            shuffle_plan_store=shuffle_plan_store,
            precompute_shuffle=precompute_shuffle,
        )
        if generator_plan_store is None:
            generator_plan_store = PartiallySymmetrizedShearGeneratorPlanStore(
                self.plan_store
            )
        elif not isinstance(
            generator_plan_store,
            PartiallySymmetrizedShearGeneratorPlanStore,
        ):
            raise TypeError(
                "generator_plan_store must be a "
                "PartiallySymmetrizedShearGeneratorPlanStore."
            )
        elif generator_plan_store.plan_store is not self.plan_store:
            raise ValueError(
                "generator_plan_store must share this core's plan_store."
            )
        self.generator_plan_store = generator_plan_store

    def __repr__(self) -> str:
        scope = (
            "none"
            if self.shuffle_plan_store is None
            else self.shuffle_plan_store.scope
        )
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            "partially_symmetrized=True, coordinates='shear', "
            f"max_trunc={self.max_truncation}, "
            f"default_trunc={self.default_truncation}, "
            "precompute_shuffle="
            f"{_precompute_shuffle_argument(scope)!r})"
        )

    def at_truncation(self, trunc: Bidegree):
        active = self._normalize_truncation(trunc)
        cached = self._truncation_views.get(active)
        if cached is not None:
            return cached
        view = type(self)(
            plan_store=self.plan_store,
            bridge_plan_store=self.bridge_plan_store,
            shear_plan_store=self.shear_plan_store,
            generator_plan_store=self.generator_plan_store,
            shuffle_plan_store=self.shuffle_plan_store,
            default_trunc=active,
            precompute_shuffle=False,
        )
        view._truncation_views = self._truncation_views
        self._truncation_views[active] = view
        return view

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        categories = dict(
            PartiallySymmetrizedBigradedCore.memory_bytes_by_category(self)
        )
        categories.update(
            self.generator_plan_store.memory_bytes_by_category()
        )
        return categories

    def plan_statistics(self) -> Mapping[str, object]:
        statistics = dict(
            PartiallySymmetrizedBigradedCore.plan_statistics(self)
        )
        categories = self.memory_bytes_by_category()
        derived_bytes = categories.get("transform_rank_matrices", 0)
        statistics.update(
            {
                "generator_plan_statistics": (
                    self.generator_plan_store.plan_statistics()
                ),
                "authoritative_memory_bytes": (
                    sum(categories.values()) - derived_bytes
                ),
                "derived_execution_memory_bytes": derived_bytes,
                "bytes_by_category": categories,
                "memory_bytes": sum(categories.values()),
                "memory_mb": sum(categories.values()) / 1024**2,
            }
        )
        return statistics

    def _apply_transform_block(
        self,
        block,
        grade: Bidegree,
        *,
        orientation,
    ):
        return apply_partially_symmetrized_transform(
            self.xp,
            self.xp.asarray(block),
            self.shear_plan_store,
            grade,
            orientation=orientation,
            scatter_add=self._rank_scatter_add,
        )

    def _coordinate_forward_block(self, block, grade):
        return self._apply_transform_block(
            block,
            grade,
            orientation="forward",
        )

    def _coordinate_inverse_block(self, block, grade):
        return self._apply_transform_block(
            block,
            grade,
            orientation="inverse",
        )

    def _coordinate_forward_transpose_block(self, block, grade):
        return self._apply_transform_block(
            block,
            grade,
            orientation="forward_transpose",
        )

    def _coordinate_inverse_transpose_block(self, block, grade):
        return self._apply_transform_block(
            block,
            grade,
            orientation="inverse_transpose",
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
        supplied, batch, output_grade = self._resolve_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades=predecessor_grades,
            generator_grades=generator_grades,
            output_grade=output_grade,
        )
        if not supplied:
            raise ValueError("the scalar grade has no generator predecessors.")
        dtype = self.xp.result_type(
            *(array.dtype for pair in supplied.values() for array in pair)
        )
        n, m = output_grade
        output_meta = self.plan_store.grade_plan(output_grade)
        doubleprime = None
        if m > 0:
            source, generator = supplied[((n, m - 1), (0, 1))]
            source_meta = self.plan_store.grade_plan((n, m - 1))
            generator_plan = self.plan_store.doubleprime_generator_plan(
                output_grade
            )
            doubleprime = apply_doubleprime_generator_prefix(
                self.xp,
                self.xp.broadcast_to(
                    source,
                    batch + (source_meta.block_width,),
                ),
                self.xp.broadcast_to(
                    generator,
                    batch + (self.dims[1],),
                ),
                generator_plan,
                scatter_add=self._rank_scatter_add,
            ).reshape(
                batch
                + (
                    generator_plan.doubleprime_rank_count,
                    output_meta.dense_shape[1],
                )
            )
            doubleprime = doubleprime.astype(dtype)
        prime = None
        if n > 0:
            source, generator = supplied[((n - 1, m), (1, 0))]
            source_meta = self.plan_store.grade_plan((n - 1, m))
            prime = apply_partially_symmetrized_prime_generator_block(
                self.xp,
                self.xp.broadcast_to(
                    source,
                    batch + (source_meta.block_width,),
                ),
                self.xp.broadcast_to(
                    generator,
                    batch + (self.dims[0],),
                ),
                self.generator_plan_store.generator_plan(output_grade),
            ).reshape(batch + output_meta.dense_shape)
            prime = prime.astype(dtype)
        if doubleprime is None and prime is None:
            raise AssertionError(
                "a non-scalar generator output needs a predecessor."
            )
        if doubleprime is None:
            output = prime
        elif prime is None:
            output = doubleprime
        else:
            doubleprime_rank_count = doubleprime.shape[-2]
            output = self.xp.concatenate(
                (
                    prime[..., :doubleprime_rank_count, :] + doubleprime,
                    prime[..., doubleprime_rank_count:, :],
                ),
                axis=-2,
            )
        return output.reshape(batch + (output_meta.block_width,))

class JaxPartiallySymmetrizedShearBigraded(
    _JaxPartiallySymmetrizedBigradedBackend,
    PartiallySymmetrizedShearBigradedCore
):
    """JAX partially symmetrized bidegree core in shear coordinates."""

    _wordwise_signature_protocol = _WORDWISE_SIGNATURE_PROTOCOL

    def __init__(
        self,
        *,
        dims: Bidegree | None = None,
        max_trunc: Bidegree | None = None,
        default_trunc: Bidegree | None = None,
        plan_store: PartiallySymmetrizedPlanStore | None = None,
        bridge_plan_store: SymmetrizationBridgePlanStore | None = None,
        shear_plan_store: PartiallySymmetrizedShearPlanStore | None = None,
        generator_plan_store: (
            PartiallySymmetrizedShearGeneratorPlanStore | None
        ) = None,
        shuffle_plan_store: (
            PartiallySymmetrizedShearShufflePlanStore | None
        ) = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        super().__init__(
            jnp,
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            plan_store=plan_store,
            bridge_plan_store=bridge_plan_store,
            shear_plan_store=shear_plan_store,
            generator_plan_store=generator_plan_store,
            shuffle_plan_store=shuffle_plan_store,
            precompute_shuffle=precompute_shuffle,
        )
        _bind_bigraded_shear_jax_methods(self)


__all__ = [
    "JaxPartiallySymmetrizedShearBigraded",
    "PartiallySymmetrizedPrimeGeneratorPlan",
    "PartiallySymmetrizedShearBigradedCore",
    "PartiallySymmetrizedShearGeneratorPlanStore",
    "apply_partially_symmetrized_prime_generator_block",
]
