"""Concrete JAX cores for ordered shear coordinates."""

from __future__ import annotations

from numbers import Integral
import types
from typing import Literal

import jax.numpy as jnp

from tensordev.core.capabilities import _WORDWISE_SIGNATURE_PROTOCOL
from tensordev.core.einsum import Einsum
from tensordev.core.jax import Jax, _compiled_jittables
from tensordev.core.grading import GradedConvolutionSchedule
from tensordev.core.shear.algebra import ShearCoordinateCore
from tensordev.core.shear.total import (
    TotalShearPlanStore,
    _dims,
    _non_negative_int,
    apply_total_masked_permutation_plan,
)
from tensordev.core.shuffle import (
    _normalize_precompute_shuffle,
    _precompute_shuffle_argument,
)
from tensordev.core.utils.annotations import jit as dummy_jit


class JaxShearTotal(ShearCoordinateCore, Einsum):
    """Bounded dense total-degree JAX core in ordered shear coordinates."""

    _wordwise_signature_protocol = _WORDWISE_SIGNATURE_PROTOCOL
    grading = "total_degree"
    coordinates = "shear"
    _mapper = Jax._mapper
    _reducer = Jax._reducer
    _accumulator = Jax._accumulator

    def __init__(
        self,
        *,
        dims: tuple[int, int] | None = None,
        max_trunc: int | None = None,
        default_trunc: int | None = None,
        precompute_shuffle: bool | Literal["generator"] = False,
        plan_store: TotalShearPlanStore | None = None,
    ) -> None:
        if plan_store is None:
            if dims is None or max_trunc is None:
                raise TypeError(
                    "dims and max_trunc are required for a total shear core."
                )
            plan_store = TotalShearPlanStore(
                dims,
                max_trunc,
                precompute_shuffle=precompute_shuffle,
            )
        else:
            if not isinstance(plan_store, TotalShearPlanStore):
                raise TypeError(
                    "plan_store must be a TotalShearPlanStore, got "
                    f"{type(plan_store).__name__}."
                )
            shuffle_scope = _normalize_precompute_shuffle(
                precompute_shuffle,
                allow_generator=True,
            )
            if shuffle_scope != "none":
                raise ValueError(
                    "precompute_shuffle and plan_store are mutually exclusive."
                )
            if dims is not None and _dims(dims) != plan_store.dims:
                raise ValueError("dims disagree with the supplied plan_store.")
            if max_trunc is not None:
                normalized_max_trunc = _non_negative_int(
                    max_trunc,
                    name="max_trunc",
                )
                if normalized_max_trunc != plan_store.max_truncation:
                    raise ValueError(
                        "max_trunc disagrees with the supplied plan_store."
                    )

        self.shear_plan_store = plan_store
        self.dims = plan_store.dims
        Einsum.__init__(
            self,
            jnp,
            d=plan_store.d,
            max_trunc=plan_store.max_truncation,
            default_trunc=default_trunc,
            shuffle_plan_store=None,
        )
        # The generic capability protocol expects the native shuffle store at
        # this name.  It is never passed to the standard JAX constructor.
        self.shuffle_plan_store = (
            None if plan_store.shuffle_scope == "none" else plan_store
        )
        for name, function in _compiled_jittables(type(self)):
            setattr(self, name, types.MethodType(function, self))

    def __repr__(self) -> str:
        precompute_shuffle = _precompute_shuffle_argument(
            self.shear_plan_store.shuffle_scope
        )
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            f"max_trunc={self.max_truncation}, "
            f"default_trunc={self.default_truncation}, "
            f"precompute_shuffle={precompute_shuffle!r})"
        )

    @property
    def capabilities(self) -> frozenset[str]:
        capabilities = {
            "concatenation",
            "generator_action",
            "coordinate_conversion",
            "shear_pairing",
        }
        if self.shear_plan_store.shuffle_scope != "none":
            capabilities.add("shuffle")
        if self.shear_plan_store.shuffle_scope == "full":
            capabilities.add("shuffle_product")
        return frozenset(capabilities)

    def at_truncation(self, trunc: int):
        active = self.normalize_truncation(trunc)
        cached = self._truncation_views.get(active)
        if cached is not None:
            return cached
        view = type(self)(
            plan_store=self.shear_plan_store,
            default_trunc=active,
        )
        view._truncation_views = self._truncation_views
        self._truncation_views[active] = view
        return view

    def memory_bytes_by_category(self) -> dict[str, int]:
        return dict(self.shear_plan_store.memory_bytes_by_category())

    def plan_statistics(self) -> dict[str, object]:
        statistics = dict(self.shear_plan_store.plan_statistics())
        statistics["default_truncation"] = self.default_truncation
        statistics["memory_mb"] = self.memory_mb()
        return statistics

    def _validate_dense_block(self, block, grade: int, *, name: str):
        if isinstance(grade, bool) or not isinstance(grade, Integral):
            raise TypeError(f"{name} grade must be a non-negative integer.")
        grade = int(grade)
        self.normalize_truncation(grade)
        block = self.xp.asarray(block)
        if block.ndim == 0 or block.shape[-1] != self.d**grade:
            width = None if block.ndim == 0 else block.shape[-1]
            raise ValueError(
                f"{name} grade-{grade} block has width {width}, expected "
                f"{self.d**grade}."
            )
        return block

    def _validate_ordered_signature_pairing_block(
        self, block, grade, *, name: str
    ):
        return self._validate_dense_block(block, grade, name=name)

    def _prepare_ordered_signature_pairing_operands(
        self,
        standard_words,
        ordered_standard_signature,
        *,
        words_first_on: bool,
        standard_first_on: bool,
    ):
        standard_words = tuple(standard_words)
        ordered_standard_signature = tuple(
            self._validate_graded_element(
                ordered_standard_signature, name="standard_tensor"
            )
        )
        standard_start = 1 if standard_first_on else 0
        for index, block in enumerate(ordered_standard_signature):
            self._validate_ordered_signature_pairing_block(
                block,
                standard_start + index,
                name="standard_tensor",
            )

        del words_first_on
        return standard_words, ordered_standard_signature

    def _apply_coordinate_element(
        self,
        element,
        plans,
        *,
        trunc,
        first_on: bool,
        transpose: bool,
    ):
        element = tuple(self._validate_graded_element(element, name="element"))
        if not element:
            if trunc is not None:
                self.normalize_truncation(trunc)
            return tuple()
        start = 1 if first_on else 0
        natural = start + len(element) - 1
        active = self._effective_truncation(trunc, natural)
        blocks = []
        for grade in range(start, active + 1):
            block = self._validate_dense_block(
                element[grade - start], grade, name="element"
            )
            if grade <= 1:
                blocks.append(block)
            else:
                blocks.append(
                    apply_total_masked_permutation_plan(
                        self.xp,
                        block,
                        plans[grade],
                        transpose=transpose,
                    )
                )
        return tuple(blocks)

    def _coordinate_forward(self, element, *, trunc=None, first_on=False):
        return self._apply_coordinate_element(
            element,
            self.shear_plan_store.forward_plans,
            trunc=trunc,
            first_on=first_on,
            transpose=False,
        )

    def _coordinate_inverse(self, element, *, trunc=None, first_on=False):
        return self._apply_coordinate_element(
            element,
            self.shear_plan_store.inverse_plans,
            trunc=trunc,
            first_on=first_on,
            transpose=False,
        )

    def _coordinate_forward_transpose(
        self, element, *, trunc=None, first_on=False
    ):
        return self._apply_coordinate_element(
            element,
            self.shear_plan_store.forward_plans,
            trunc=trunc,
            first_on=first_on,
            transpose=True,
        )

    def _coordinate_inverse_transpose(
        self, element, *, trunc=None, first_on=False
    ):
        return self._apply_coordinate_element(
            element,
            self.shear_plan_store.inverse_plans,
            trunc=trunc,
            first_on=first_on,
            transpose=True,
        )

    def _coordinate_forward_block(self, block, grade):
        block = self._validate_dense_block(block, grade, name="block")
        if grade <= 1:
            return block
        return apply_total_masked_permutation_plan(
            self.xp, block, self.shear_plan_store.forward_plans[grade]
        )

    def _coordinate_inverse_block(self, block, grade):
        block = self._validate_dense_block(block, grade, name="block")
        if grade <= 1:
            return block
        return apply_total_masked_permutation_plan(
            self.xp, block, self.shear_plan_store.inverse_plans[grade]
        )

    def _coordinate_forward_transpose_block(self, block, grade):
        block = self._validate_dense_block(block, grade, name="block")
        if grade <= 1:
            return block
        return apply_total_masked_permutation_plan(
            self.xp,
            block,
            self.shear_plan_store.forward_plans[grade],
            transpose=True,
        )

    def _coordinate_inverse_transpose_block(self, block, grade):
        block = self._validate_dense_block(block, grade, name="block")
        if grade <= 1:
            return block
        return apply_total_masked_permutation_plan(
            self.xp,
            block,
            self.shear_plan_store.inverse_plans[grade],
            transpose=True,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("left_grade", "right_grade"),
        dynamic_batch=("Ai", "Bj"),
    )
    def tensor_product_homogeneous(
        self,
        Ai,
        Bj,
        *,
        left_grade: int | None = None,
        right_grade: int | None = None,
    ):
        if left_grade is None or right_grade is None:
            raise TypeError(
                "total shear homogeneous products require left_grade= and "
                "right_grade=."
            )
        left_grade = self.normalize_truncation(left_grade)
        right_grade = self.normalize_truncation(right_grade)
        output_grade = left_grade + right_grade
        self.normalize_truncation(output_grade)
        return self._product_output_block(
            ((Ai, Bj, left_grade, right_grade),),
            output_grade,
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
        if len(predecessor_blocks) != 1:
            raise ValueError("a total-degree generator action has one predecessor")
        source_grade = self.normalize_truncation(
            tuple(predecessor_grades)[0]
        )
        generator_grade = self.normalize_truncation(
            tuple(generator_grades)[0]
        )
        output_grade = self.normalize_truncation(output_grade)
        if generator_grade != 1 or source_grade + 1 != output_grade:
            raise ValueError("generator grades do not add to output_grade")
        source = self._validate_dense_block(
            predecessor_blocks[0], source_grade, name="predecessor"
        )
        generator = self._validate_dense_block(
            generator_blocks[0], 1, name="generator"
        )
        batch = self.xp.broadcast_shapes(source.shape[:-1], generator.shape[:-1])
        dtype = self.xp.result_type(source, generator)
        source = self.xp.broadcast_to(
            source, batch + (self.d**source_grade,)
        ).astype(dtype)
        generator = self.xp.broadcast_to(generator, batch + (self.d,)).astype(dtype)

        z_prime = generator[..., : self.dims[0]]
        z_doubleprime = generator[..., self.dims[0] :]
        prime_raw = (
            source[..., :, None] * z_prime[..., None, :]
        ).reshape(batch + (self.d**source_grade * self.dims[0],))
        prime = apply_total_masked_permutation_plan(
            self.xp,
            prime_raw,
            self.shear_plan_store.generator_plans[output_grade],
        )

        source_tensor = source.reshape(batch + (self.d,) * source_grade)
        doubleprime_values = source_tensor[..., None] * z_doubleprime.reshape(
            batch + (1,) * source_grade + (self.dims[1],)
        )
        doubleprime = self.xp.zeros(
            batch + (self.d,) * output_grade, dtype=dtype
        )
        target = (
            (slice(None),) * len(batch)
            + (slice(None),) * source_grade
            + (slice(self.dims[0], None),)
        )
        doubleprime = doubleprime.at[target].set(doubleprime_values)
        return prime + doubleprime.reshape(batch + (self.d**output_grade,))

    def _require_shuffle(self) -> TotalShearPlanStore:
        if self.shear_plan_store.shuffle_scope == "none":
            raise RuntimeError(
                "This shear core was constructed with precompute_shuffle=False."
            )
        return self.shear_plan_store

    def _require_full_shuffle(self) -> TotalShearPlanStore:
        store = self._require_shuffle()
        if store.shuffle_scope != "full":
            raise RuntimeError(
                "Arbitrary shear shuffle products require precompute_shuffle=True; "
                "this core contains generator-only plans."
            )
        return store

    def _shuffle_schedule(
        self,
        A,
        B,
        trunc,
        *,
        a_first_on,
        b_first_on,
        first_on_out,
    ) -> GradedConvolutionSchedule:
        self._require_full_shuffle()
        return self._standard_product_schedule(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )

    def _shuffle_block(
        self, left, right, left_grade, right_grade, output_grade
    ):
        left_grade = self.normalize_truncation(left_grade)
        right_grade = self.normalize_truncation(right_grade)
        output_grade = self.normalize_truncation(output_grade)
        if left_grade + right_grade != output_grade:
            raise ValueError("shear shuffle plan/output grade mismatch")
        plan, swapped = self.shear_plan_store.shuffle_plan(
            left_grade, right_grade
        )
        if swapped:
            left, right = right, left
            left_grade, right_grade = right_grade, left_grade
        left = self._validate_dense_block(left, int(left_grade), name="left")
        right = self._validate_dense_block(right, int(right_grade), name="right")
        batch = self.xp.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        dtype = self.xp.result_type(left, right)
        left = self.xp.broadcast_to(
            left, batch + (self.d ** int(left_grade),)
        ).astype(dtype)
        right = self.xp.broadcast_to(
            right, batch + (self.d ** int(right_grade),)
        ).astype(dtype)
        raw = (left[..., :, None] * right[..., None, :]).reshape(
            batch + (self.d ** int(output_grade),)
        )
        return apply_total_masked_permutation_plan(self.xp, raw, plan)

    @dummy_jit(static_argnums=(0, 3), dynamic_batch=("Ai", "v"))
    def tensor_shuffle_vector_homogeneous(self, Ai, v, i: int):
        self._require_shuffle()
        input_grade = self.normalize_truncation(i)
        output_grade = self.normalize_truncation(input_grade + 1)
        return self._shuffle_block(Ai, v, input_grade, 1, output_grade)

    @dummy_jit(
        static_argnums=0,
        static_argnames=("multiplier_grade", "output_grade"),
        dynamic_batch=("Ai", "Yni"),
    )
    def tensor_adjoint_left_homogeneous(
        self,
        Ai,
        Yni,
        *,
        multiplier_grade: int | None = None,
        output_grade: int | None = None,
    ):
        if multiplier_grade is None or output_grade is None:
            raise TypeError(
                "total shear homogeneous adjoints require multiplier_grade= "
                "and output_grade=."
            )
        multiplier_grade = self.normalize_truncation(multiplier_grade)
        output_grade = self.normalize_truncation(output_grade)
        target_grade = self.normalize_truncation(
            multiplier_grade + output_grade
        )
        return self._adjoint_left_block(
            Ai,
            Yni,
            multiplier_grade,
            target_grade,
            output_grade,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("multiplier_grade", "output_grade"),
        dynamic_batch=("Bj", "Ynj"),
    )
    def tensor_adjoint_right_homogeneous(
        self,
        Bj,
        Ynj,
        *,
        multiplier_grade: int | None = None,
        output_grade: int | None = None,
    ):
        if multiplier_grade is None or output_grade is None:
            raise TypeError(
                "total shear homogeneous adjoints require multiplier_grade= "
                "and output_grade=."
            )
        multiplier_grade = self.normalize_truncation(multiplier_grade)
        output_grade = self.normalize_truncation(output_grade)
        target_grade = self.normalize_truncation(
            output_grade + multiplier_grade
        )
        return self._adjoint_right_block(
            Bj,
            Ynj,
            multiplier_grade,
            target_grade,
            output_grade,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("left_grade", "right_grade", "row_axis", "col_axis"),
        dynamic_batch=("A", "B"),
    )
    def tensor_matrix_product_homogeneous(
        self,
        A,
        B,
        row_axis: int = -3,
        col_axis: int = -2,
        *,
        left_grade: int | None = None,
        right_grade: int | None = None,
    ):
        if left_grade is None or right_grade is None:
            raise TypeError(
                "total shear homogeneous matrix products require left_grade= "
                "and right_grade=."
            )
        left_grade = self.normalize_truncation(left_grade)
        right_grade = self.normalize_truncation(right_grade)
        output_grade = self.normalize_truncation(left_grade + right_grade)
        return self._matrix_product_block(
            A,
            B,
            left_grade,
            right_grade,
            output_grade,
            row_axis=row_axis,
            col_axis=col_axis,
        )


__all__ = ["JaxShearTotal"]
