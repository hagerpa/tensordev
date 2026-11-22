"""Standard-coordinate algebra on partially symmetrized bidegree blocks."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np

from tensordev.core.bigraded.standard import StandardBigradedCore
from tensordev.core.bigraded.symmetrized.bridge import (
    SymmetrizationBridgePlanStore,
    _validate_block_width,
    _lift_partially_symmetrized_block,
    pair_partially_symmetrized_with_ordered_block,
    partially_symmetrize_block,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
    apply_doubleprime_generator_prefix,
)
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
    apply_partially_symmetrized_shear_shuffle_block,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
    apply_partially_symmetrized_transform,
)
from tensordev.core.bigraded.types import (
    Bidegree,
    BigradedSpec,
    BigradedTensor,
    _bidegree,
)
from tensordev.core.utils.annotations import jit as dummy_jit
from tensordev.core.utils.segmented import (
    SegmentedRankPlan,
    apply_segmented_rank_plan,
)
from tensordev.core.shuffle import _normalize_precompute_shuffle


class PartiallySymmetrizedBigradedCore(StandardBigradedCore):
    """Shared graded drivers with partially symmetrized block kernels."""

    partially_symmetrized = True
    coordinates = "standard"

    @property
    def capabilities(self) -> frozenset[str]:
        capabilities = {
            "concatenation",
            "generator_action",
            "coordinate_conversion",
            "partial_symmetrization",
            "shear_pairing",
        }
        if self.shuffle_plan_store is not None:
            capabilities.add("shuffle")
            if self.shuffle_plan_store.scope == "full":
                capabilities.add("shuffle_product")
        return frozenset(capabilities)

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
        shuffle_plan_store: PartiallySymmetrizedShearShufflePlanStore | None = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        shuffle_scope = _normalize_precompute_shuffle(
            precompute_shuffle,
            allow_generator=True,
        )
        if plan_store is None:
            if dims is None or max_trunc is None:
                raise TypeError(
                    "dims and max_trunc are required for a partially "
                    "symmetrized bidegree core."
                )
            plan_store = PartiallySymmetrizedPlanStore(dims, max_trunc)
        elif not isinstance(plan_store, PartiallySymmetrizedPlanStore):
            raise TypeError(
                "plan_store must be a PartiallySymmetrizedPlanStore, got "
                f"{type(plan_store).__name__}."
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
        if bridge_plan_store is None:
            bridge_plan_store = SymmetrizationBridgePlanStore(
                self.dims,
                self.max_truncation,
            )
        elif not isinstance(
            bridge_plan_store,
            SymmetrizationBridgePlanStore,
        ):
            raise TypeError(
                "bridge_plan_store must be a SymmetrizationBridgePlanStore, "
                f"got {type(bridge_plan_store).__name__}."
            )
        if bridge_plan_store.dims != self.dims:
            raise ValueError("bridge_plan_store dimensions disagree with plan_store.")
        if bridge_plan_store.max_truncation != self.max_truncation:
            raise ValueError(
                "bridge_plan_store capacity disagrees with plan_store."
            )
        self.bridge_plan_store = bridge_plan_store
        if shear_plan_store is not None:
            if not isinstance(
                shear_plan_store,
                PartiallySymmetrizedShearPlanStore,
            ):
                raise TypeError(
                    "shear_plan_store must be a "
                    "PartiallySymmetrizedShearPlanStore, got "
                    f"{type(shear_plan_store).__name__}."
                )
            if shear_plan_store.plan_store is not self.plan_store:
                raise ValueError("shear_plan_store must share this core's plan_store.")

        if shuffle_plan_store is not None:
            if not isinstance(
                shuffle_plan_store,
                PartiallySymmetrizedShearShufflePlanStore,
            ):
                raise TypeError(
                    "shuffle_plan_store must be a compatible partially "
                    "symmetrized shear-shuffle plan store, got "
                    f"{type(shuffle_plan_store).__name__}."
                )
            if shuffle_scope != "none":
                raise ValueError(
                    "precompute_shuffle and shuffle_plan_store are mutually "
                    "exclusive."
                )
            if shuffle_plan_store.plan_store is not self.plan_store:
                raise ValueError(
                    "shuffle_plan_store must share this core's plan_store."
                )
            shuffle_scope = shuffle_plan_store.scope
            if shuffle_scope == "none":
                shuffle_plan_store = None
        elif shuffle_scope != "none":
            shuffle_plan_store = PartiallySymmetrizedShearShufflePlanStore(
                self.plan_store,
                scope=shuffle_scope,
            )

        if shuffle_scope != "none" and shear_plan_store is None:
            shear_plan_store = PartiallySymmetrizedShearPlanStore(
                self.plan_store
            )
        self.shear_plan_store = shear_plan_store
        self.shuffle_plan_store = shuffle_plan_store

    def __repr__(self) -> str:
        scope = (
            "none"
            if self.shuffle_plan_store is None
            else self.shuffle_plan_store.scope
        )
        shuffle = {"none": False, "generator": "generator", "full": True}[scope]
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            "partially_symmetrized=True, "
            f"max_trunc={self.max_truncation}, "
            f"default_trunc={self.default_truncation}, "
            f"precompute_shuffle={shuffle!r})"
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
            shuffle_plan_store=self.shuffle_plan_store,
            default_trunc=active,
            precompute_shuffle=False,
        )
        view._truncation_views = self._truncation_views
        self._truncation_views[active] = view
        return view

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        categories = dict(self.plan_store.memory_bytes_by_category())
        categories.update(self.bridge_plan_store.memory_bytes_by_category())
        if self.shear_plan_store is not None:
            categories.update(self.shear_plan_store.memory_bytes_by_category())
        if self.shuffle_plan_store is not None:
            categories.update(self.shuffle_plan_store.memory_bytes_by_category())
        return categories

    def plan_statistics(self) -> Mapping[str, object]:
        categories = self.memory_bytes_by_category()
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "default_truncation": self.default_truncation,
            "partially_symmetrized": self.partially_symmetrized,
            "coordinates": self.coordinates,
            "block_plan_statistics": self.plan_store.plan_statistics(),
            "bridge_plan_statistics": self.bridge_plan_store.plan_statistics(),
            "bytes_by_category": categories,
            "memory_bytes": sum(categories.values()),
            "memory_mb": sum(categories.values()) / 1024**2,
            "shear_plan_statistics": (
                None
                if self.shear_plan_store is None
                else self.shear_plan_store.plan_statistics()
            ),
            "shuffle_plan_statistics": (
                None
                if self.shuffle_plan_store is None
                else self.shuffle_plan_store.plan_statistics()
            ),
            "shuffle_enabled": self.shuffle_plan_store is not None,
            "shuffle_scope": (
                "none"
                if self.shuffle_plan_store is None
                else self.shuffle_plan_store.scope
            ),
        }

    # ------------------------------------------------------------------
    # Shared rank executor boundary
    # ------------------------------------------------------------------

    def _rank_scatter(self, output, target_ranks, values):
        raise NotImplementedError("The backend must implement functional rank scatter.")

    def _rank_scatter_add(self, output, target_ranks, values):
        raise NotImplementedError(
            "The backend must implement functional rank scatter-add."
        )

    def _expanded_block_shape(self, grade: Bidegree) -> tuple[int, ...]:
        n, _ = grade
        plan = self.plan_store.grade_plan(grade)
        return (plan.rank_count,) + (self.dims[0],) * n

    def _quotient_concat_values(self, coordinate_pairs, plan):
        """Arrange flat coordinate pairs as rank-pair/dense-prime rows."""
        left_meta = self.plan_store.grade_plan(plan.left_grade)
        right_meta = self.plan_store.grade_plan(plan.right_grade)
        prefix = coordinate_pairs.shape[:-2]
        raw = coordinate_pairs.reshape(
            prefix
            + (
                left_meta.rank_count,
                left_meta.dense_shape[1],
                right_meta.rank_count,
                right_meta.dense_shape[1],
            )
        )
        prefix_ndim = len(prefix)
        raw = self.xp.transpose(
            raw,
            tuple(range(prefix_ndim))
            + (
                prefix_ndim,
                prefix_ndim + 2,
                prefix_ndim + 1,
                prefix_ndim + 3,
            ),
        )
        return raw.reshape(
            prefix
            + (
                plan.rank_plan.source_count,
                self.plan_store.grade_plan(plan.output_grade).dense_shape[1],
            )
        )

    def _quotient_concat_scatter(self, coordinate_pairs, plan):
        values = self._quotient_concat_values(coordinate_pairs, plan)
        output = apply_segmented_rank_plan(
            self.xp,
            values,
            plan.rank_plan,
            scatter_add=self._rank_scatter_add,
        )
        return output.reshape(
            coordinate_pairs.shape[:-2]
            + (self.plan_store.grade_plan(plan.output_grade).block_width,)
        )

    def _quotient_concat_pull(self, target, plan):
        left_meta = self.plan_store.grade_plan(plan.left_grade)
        right_meta = self.plan_store.grade_plan(plan.right_grade)
        output_meta = self.plan_store.grade_plan(plan.output_grade)
        batch = target.shape[:-1]
        target = target.reshape(batch + output_meta.dense_shape)
        selected = self.xp.take(
            target,
            self.xp.asarray(plan.rank_plan.target_ranks),
            axis=-2,
        ).reshape(
            batch
            + (
                left_meta.rank_count,
                right_meta.rank_count,
                left_meta.dense_shape[1],
                right_meta.dense_shape[1],
            )
        )
        batch_ndim = len(batch)
        selected = self.xp.transpose(
            selected,
            tuple(range(batch_ndim))
            + (
                batch_ndim,
                batch_ndim + 2,
                batch_ndim + 1,
                batch_ndim + 3,
            ),
        )
        return selected.reshape(
            batch + (left_meta.block_width, right_meta.block_width)
        )

    # ------------------------------------------------------------------
    # Standard partially symmetrized concatenation and transposes
    # ------------------------------------------------------------------

    def _standard_product_block(
        self,
        left,
        right,
        left_grade: Bidegree,
        right_grade: Bidegree,
        output_grade: Bidegree,
    ):
        plan = self.plan_store.concat_plan(left_grade, right_grade)
        if plan.output_grade != output_grade:
            raise ValueError("product plan/output grade mismatch")
        left_meta = self.plan_store.grade_plan(left_grade)
        right_meta = self.plan_store.grade_plan(right_grade)
        batch = self.xp.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        left = self.xp.broadcast_to(left, batch + (left_meta.block_width,))
        right = self.xp.broadcast_to(right, batch + (right_meta.block_width,))
        return self._quotient_concat_scatter(
            left[..., :, None] * right[..., None, :],
            plan,
        )

    def _standard_product_output_block(self, contributions, output_grade: Bidegree):
        output_meta = self.plan_store.grade_plan(output_grade)
        batch = self.xp.broadcast_shapes(
            *(block.shape[:-1] for pair in contributions for block in pair[:2])
        )
        values = []
        targets = []
        for left, right, left_grade, right_grade in contributions:
            plan = self.plan_store.concat_plan(left_grade, right_grade)
            if plan.output_grade != output_grade:
                raise ValueError("product plan/output grade mismatch")
            left_meta = self.plan_store.grade_plan(left_grade)
            right_meta = self.plan_store.grade_plan(right_grade)
            left = self.xp.broadcast_to(left, batch + (left_meta.block_width,))
            right = self.xp.broadcast_to(right, batch + (right_meta.block_width,))
            values.append(
                self._quotient_concat_values(
                    left[..., :, None] * right[..., None, :],
                    plan,
                )
            )
            targets.append(plan.rank_plan.target_ranks)

        fused_values = (
            values[0]
            if len(values) == 1
            else self.xp.concatenate(tuple(values), axis=-2)
        )
        fused_targets = (
            targets[0]
            if len(targets) == 1
            else np.concatenate(tuple(targets))
        )
        fused_plan = SegmentedRankPlan.from_targets(
            fused_targets,
            output_rank_count=output_meta.rank_count,
        )
        output = apply_segmented_rank_plan(
            self.xp,
            fused_values,
            fused_plan,
            scatter_add=self._rank_scatter_add,
        )
        return output.reshape(batch + (output_meta.block_width,))

    def _standard_adjoint_left_block(
        self,
        multiplier,
        target,
        multiplier_grade: Bidegree,
        target_grade: Bidegree,
        output_grade: Bidegree,
    ):
        expected = (
            multiplier_grade[0] + output_grade[0],
            multiplier_grade[1] + output_grade[1],
        )
        if target_grade != expected:
            raise ValueError("left-adjoint grade mismatch")
        plan = self.plan_store.concat_plan(multiplier_grade, output_grade)
        multiplier_width = self.plan_store.grade_plan(multiplier_grade).block_width
        target_width = self.plan_store.grade_plan(target_grade).block_width
        batch = self.xp.broadcast_shapes(multiplier.shape[:-1], target.shape[:-1])
        multiplier = self.xp.broadcast_to(
            multiplier, batch + (multiplier_width,)
        )
        target = self.xp.broadcast_to(target, batch + (target_width,))
        selected = self._quotient_concat_pull(target, plan)
        return (multiplier[..., :, None] * selected).sum(axis=-2)

    def _standard_adjoint_right_block(
        self,
        multiplier,
        target,
        multiplier_grade: Bidegree,
        target_grade: Bidegree,
        output_grade: Bidegree,
    ):
        expected = (
            output_grade[0] + multiplier_grade[0],
            output_grade[1] + multiplier_grade[1],
        )
        if target_grade != expected:
            raise ValueError("right-adjoint grade mismatch")
        plan = self.plan_store.concat_plan(output_grade, multiplier_grade)
        multiplier_width = self.plan_store.grade_plan(multiplier_grade).block_width
        target_width = self.plan_store.grade_plan(target_grade).block_width
        batch = self.xp.broadcast_shapes(multiplier.shape[:-1], target.shape[:-1])
        multiplier = self.xp.broadcast_to(
            multiplier, batch + (multiplier_width,)
        )
        target = self.xp.broadcast_to(target, batch + (target_width,))
        selected = self._quotient_concat_pull(target, plan)
        return (selected * multiplier[..., None, :]).sum(axis=-1)

    def _standard_matrix_product_block(
        self,
        left,
        right,
        left_grade: Bidegree,
        right_grade: Bidegree,
        output_grade: Bidegree,
        *,
        row_axis: int,
        col_axis: int,
    ):
        left, original_row, original_col = self._canonicalize_matrix_axes(
            left, row_axis, col_axis
        )
        right, _, _ = self._canonicalize_matrix_axes(
            right, row_axis, col_axis
        )
        pairs = self.xp.einsum("...rki,...klj->...rlij", left, right)
        plan = self.plan_store.concat_plan(left_grade, right_grade)
        if plan.output_grade != output_grade:
            raise ValueError("matrix product plan/output grade mismatch")
        output = self._quotient_concat_scatter(pairs, plan)
        return self._restore_matrix_axes(output, original_row, original_col)

    # ------------------------------------------------------------------
    # Standard partially symmetrized first-level Horner action
    # ------------------------------------------------------------------

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
        n, m = output_grade
        output_meta = self.plan_store.grade_plan(output_grade)
        dtype = self.xp.result_type(
            *(array.dtype for pair in supplied.values() for array in pair)
        )
        pieces = []
        if m > 0:
            source, generator = supplied[((n, m - 1), (0, 1))]
            source_meta = self.plan_store.grade_plan((n, m - 1))
            generator_plan = self.plan_store.doubleprime_generator_plan(
                output_grade
            )
            doubleprime = apply_doubleprime_generator_prefix(
                self.xp,
                self.xp.broadcast_to(source, batch + (source_meta.block_width,)),
                self.xp.broadcast_to(generator, batch + (self.dims[1],)),
                generator_plan,
                scatter_add=self._rank_scatter_add,
            ).reshape(
                batch
                + (
                    generator_plan.doubleprime_rank_count,
                    output_meta.dense_shape[1],
                )
            )
            pieces.append(doubleprime.astype(dtype))
        if n > 0:
            source, generator = supplied[((n - 1, m), (1, 0))]
            source_meta = self.plan_store.grade_plan((n - 1, m))
            source = self.xp.broadcast_to(
                source, batch + (source_meta.block_width,)
            ).reshape(batch + source_meta.dense_shape)
            generator = self.xp.broadcast_to(
                generator, batch + (self.dims[0],)
            )
            values = (source[..., None] * generator[..., None, None, :]).reshape(
                batch + (source_meta.rank_count, output_meta.dense_shape[1])
            )
            pieces.append(values.astype(dtype))
        if not pieces:
            raise ValueError("the scalar grade has no generator predecessors.")
        output = (
            pieces[0]
            if len(pieces) == 1
            else self.xp.concatenate(tuple(pieces), axis=-2)
        )
        if output.shape[-2] != output_meta.rank_count:
            raise AssertionError("generator rank pieces do not fill the output")
        return output.reshape(batch + (output_meta.block_width,))

    # ------------------------------------------------------------------
    # Partially symmetrized shuffle through shear-coordinate transport
    # ------------------------------------------------------------------

    def _doubleprime_shuffle_generator_block(self, generator):
        """Place a raw double-prime vector in symmetrized rank order."""
        generator = self.xp.asarray(generator)
        if generator.ndim == 0 or generator.shape[-1] != self.dims[1]:
            raise ValueError("double-prime generator has the wrong width.")
        plan = self.plan_store.doubleprime_generator_plan((0, 1))
        source = self.xp.ones(
            generator.shape[:-1] + (1,),
            dtype=generator.dtype,
        )
        return apply_doubleprime_generator_prefix(
            self.xp,
            source,
            generator,
            plan,
            scatter_add=self._rank_scatter_add,
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
        generator_blocks = tuple(
            self._doubleprime_shuffle_generator_block(generator)
            if _bidegree(grade, name="generator_grade") == (0, 1)
            else generator
            for generator, grade in zip(generator_blocks, generator_grades)
        )
        return super()._shuffle_generator_output_block(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades=predecessor_grades,
            generator_grades=generator_grades,
            output_grade=output_grade,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("input_grade", "generator_part"),
        dynamic_batch=("Ai", "v"),
    )
    def tensor_shuffle_vector_homogeneous(
        self,
        Ai,
        v,
        *,
        input_grade=None,
        generator_part=None,
    ):
        if generator_part == "doubleprime":
            v = self.xp.asarray(v)
            if v.ndim > 0 and v.shape[-1] == sum(self.dims):
                prime, doubleprime = self._split_generator(v)
                v = self.xp.concat(
                    (
                        prime,
                        self._doubleprime_shuffle_generator_block(
                            doubleprime
                        ),
                    ),
                    axis=-1,
                )
            elif v.ndim > 0 and v.shape[-1] == self.dims[1]:
                v = self._doubleprime_shuffle_generator_block(v)
        return super().tensor_shuffle_vector_homogeneous(
            Ai,
            v,
            input_grade=input_grade,
            generator_part=generator_part,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "a_first_on"),
        dynamic_batch=("A", "v"),
    )
    def tensor_shuffle_vector(
        self,
        A,
        v,
        *,
        trunc=None,
        a_first_on=False,
    ):
        v = self.xp.asarray(v)
        if v.ndim > 0 and v.shape[-1] == sum(self.dims):
            prime, doubleprime = self._split_generator(v)
            v = self.xp.concat(
                (
                    prime,
                    self._doubleprime_shuffle_generator_block(doubleprime),
                ),
                axis=-1,
            )
        return super().tensor_shuffle_vector(
            A,
            v,
            trunc=trunc,
            a_first_on=a_first_on,
        )

    def _apply_quotient_transform(self, block, grade, *, orientation: str):
        if self.shear_plan_store is None:
            raise RuntimeError(
                "partially symmetrized shuffle coordinate data was not "
                "precomputed."
            )
        return apply_partially_symmetrized_transform(
            self.xp,
            block,
            self.shear_plan_store,
            grade,
            orientation=orientation,
            scatter_add=self._rank_scatter_add,
        )

    def _gamma_shuffle_block(
        self,
        left,
        right,
        left_grade: Bidegree,
        right_grade: Bidegree,
        output_grade: Bidegree,
    ):
        store = self._require_shuffle()
        plan, swap = store.resolve_block_plan(left_grade, right_grade)
        if plan.output_grade != output_grade:
            raise ValueError("shuffle plan/output grade mismatch")
        if swap:
            left, right = right, left
        return apply_partially_symmetrized_shear_shuffle_block(
            self.xp,
            left,
            right,
            plan,
            scatter_add=self._rank_scatter_add,
        )

    def _shuffle_block(
        self,
        left,
        right,
        left_grade: Bidegree,
        right_grade: Bidegree,
        output_grade: Bidegree,
    ):
        expected = (
            left_grade[0] + right_grade[0],
            left_grade[1] + right_grade[1],
        )
        if output_grade != expected:
            raise ValueError("shuffle plan/output grade mismatch")
        self._require_shuffle()
        shear_left = self._apply_quotient_transform(
            left,
            left_grade,
            orientation="inverse_transpose",
        )
        shear_right = self._apply_quotient_transform(
            right,
            right_grade,
            orientation="inverse_transpose",
        )
        shear_output = self._gamma_shuffle_block(
            shear_left,
            shear_right,
            left_grade,
            right_grade,
            output_grade,
        )
        return self._apply_quotient_transform(
            shear_output,
            output_grade,
            orientation="forward_transpose",
        )

    # ------------------------------------------------------------------
    # Partial-symmetrization bridge
    # ------------------------------------------------------------------

    def _validate_ordered_bridge_element(self, element, *, name: str):
        if not isinstance(element, BigradedTensor):
            raise TypeError(
                f"{name} must be a BigradedTensor, got {type(element).__name__}."
            )
        if element.spec.dims != self.dims:
            raise ValueError(
                f"{name} uses alphabet dimensions {element.spec.dims}, "
                f"expected {self.dims}."
            )
        if element.spec.coordinates != self.coordinates:
            raise ValueError(
                f"{name} uses coordinates {element.spec.coordinates!r}, "
                f"expected {self.coordinates!r}."
            )
        if element.spec.partially_symmetrized:
            raise ValueError(
                f"{name} is partially symmetrized; expected an ordered tensor."
            )
        return element

    def _validate_standard_pairing_tensor(
        self,
        element,
        *,
        name: str,
        expected_partially_symmetrized: bool,
    ):
        """Validate the symmetrization state of a standard-coordinate target."""
        if not isinstance(element, BigradedTensor):
            raise TypeError(
                f"{name} must be a BigradedTensor, got "
                f"{type(element).__name__}."
            )
        if element.spec.dims != self.dims:
            raise ValueError(
                f"{name} uses alphabet dimensions {element.spec.dims}, "
                f"expected {self.dims}."
            )
        if element.spec.coordinates != "standard":
            raise ValueError(
                f"{name} uses coordinates {element.spec.coordinates!r}, "
                "expected 'standard'."
            )
        if (
            element.spec.partially_symmetrized
            != expected_partially_symmetrized
        ):
            expected = (
                "a partially symmetrized tensor"
                if expected_partially_symmetrized
                else "an ordered tensor"
            )
            raise ValueError(
                f"{name} has partially_symmetrized="
                f"{element.spec.partially_symmetrized!r}; expected {expected}."
            )
        return element

    @dummy_jit(
        static_argnums=0,
        static_argnames=("grade",),
        dynamic_batch=("block",),
    )
    def tensor_partially_symmetrize_homogeneous(
        self,
        block,
        *,
        grade: Bidegree | None = None,
    ):
        if grade is None:
            raise TypeError(
                "tensor_partially_symmetrize_homogeneous requires grade=."
            )
        grade = _bidegree(grade, name="grade")
        self.plan_store.grade_plan(grade)
        return partially_symmetrize_block(
            self.xp,
            self.xp.asarray(block),
            self.bridge_plan_store.grade_plan(grade),
            scatter_add=self._rank_scatter_add,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "first_on"),
        dynamic_batch=("element",),
    )
    def tensor_partially_symmetrize(
        self,
        element,
        *,
        trunc: Bidegree | None = None,
        first_on: bool = False,
    ) -> BigradedTensor:
        element = self._validate_ordered_bridge_element(element, name="element")
        if first_on == element.spec.include_scalar:
            expected = "omit" if first_on else "include"
            raise ValueError(
                f"first_on={first_on} requires the element to {expected} "
                "the scalar block."
            )
        requested = (
            self.max_truncation
            if trunc is None
            else self._normalize_truncation(trunc)
        )
        active = (
            min(requested[0], element.truncation[0]),
            min(requested[1], element.truncation[1]),
        )
        layout = self.resolve_layout(
            active,
            include_scalar=element.spec.include_scalar,
        )
        return BigradedTensor(
            tuple(
                partially_symmetrize_block(
                    self.xp,
                    element[grade],
                    self.bridge_plan_store.grade_plan(grade),
                    scatter_add=self._rank_scatter_add,
                )
                for grade in layout.grades
            ),
            layout.spec,
        )

    def _lift_partially_symmetrized_block(self, block, grade: Bidegree):
        grade = _bidegree(grade, name="grade")
        return _lift_partially_symmetrized_block(
            self.xp,
            block,
            self.bridge_plan_store.grade_plan(grade),
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "first_on"),
        dynamic_batch=("element",),
    )
    def tensor_to_ordered(
        self,
        element,
        *,
        trunc: Bidegree | None = None,
        first_on: bool = False,
    ) -> BigradedTensor:
        """Expand a partially symmetrized word tensor in ordered form.

        Each compact coefficient is copied to every ordered word it
        represents.  This is adjoint to ``tensor_partially_symmetrize``, not
        its inverse; grading, coordinates, and scalar policy are preserved.
        """
        return self._lift_partially_symmetrized(
            element,
            trunc=trunc,
            first_on=first_on,
        )

    def _lift_partially_symmetrized(
        self,
        element,
        *,
        trunc: Bidegree | None = None,
        first_on: bool = False,
    ) -> BigradedTensor:
        element = self._validate_element(element, name="element")
        if first_on == element.spec.include_scalar:
            raise ValueError("first_on disagrees with the element scalar policy.")
        requested = (
            element.truncation
            if trunc is None
            else self._normalize_truncation(trunc)
        )
        active = (
            min(requested[0], element.truncation[0]),
            min(requested[1], element.truncation[1]),
        )
        spec = BigradedSpec(
            *self.dims,
            active,
            coordinates=self.coordinates,
            include_scalar=element.spec.include_scalar,
            partially_symmetrized=False,
        )
        return BigradedTensor(
            tuple(
                self._lift_partially_symmetrized_block(element[grade], grade)
                for grade in spec.grades
            ),
            spec,
        )

    def _pair_standard_block_with_ordered_signature(
        self,
        words,
        signature,
        *,
        grade: Bidegree,
    ):
        return pair_partially_symmetrized_with_ordered_block(
            self.xp,
            words,
            signature,
            self.bridge_plan_store.grade_plan(grade),
        )

    def _pair_standard_block_with_standard_tensor(
        self,
        words,
        standard_tensor,
        *,
        grade: Bidegree,
        standard_partially_symmetrized: bool,
    ):
        if not standard_partially_symmetrized:
            return self._pair_standard_block_with_ordered_signature(
                words,
                standard_tensor,
                grade=grade,
            )
        plan = self.bridge_plan_store.grade_plan(grade)
        _validate_block_width(
            words,
            plan.quotient_block_width,
            name="partially symmetrized word block",
            grade=grade,
        )
        _validate_block_width(
            standard_tensor,
            plan.quotient_block_width,
            name="partially symmetrized standard block",
            grade=grade,
        )
        return self.tensor_inner_product_homogeneous(words, standard_tensor)

    def _prepare_standard_pairing_operands(
        self,
        standard_words,
        standard_tensor,
        *,
        words_first_on: bool = False,
        standard_first_on: bool = False,
        expected_partially_symmetrized: bool,
    ):
        standard_words = self._validate_element_coordinates(
            standard_words,
            name="words",
            coordinates="standard",
        )
        standard_tensor = self._validate_standard_pairing_tensor(
            standard_tensor,
            name="standard_tensor",
            expected_partially_symmetrized=expected_partially_symmetrized,
        )
        for name, first_on, include_scalar in (
            (
                "words_first_on",
                words_first_on,
                standard_words.spec.include_scalar,
            ),
            (
                "standard_first_on",
                standard_first_on,
                standard_tensor.spec.include_scalar,
            ),
        ):
            if first_on == include_scalar:
                expected = "omit" if first_on else "include"
                raise ValueError(
                    f"{name}={first_on} requires its tensor to {expected} "
                    "the scalar block."
                )
        return standard_words, standard_tensor

    def _pair_prepared_standard_words(
        self,
        words,
        standard_tensor,
        *,
        standard_partially_symmetrized: bool,
    ):
        common = tuple(
            grade
            for grade in words.grades
            if standard_tensor.spec.contains(grade)
        )
        if not common:
            return self._standard_pairing_zero(words, standard_tensor)
        terms = tuple(
            self._pair_standard_block_with_standard_tensor(
                words[grade],
                standard_tensor[grade],
                grade=grade,
                standard_partially_symmetrized=(
                    standard_partially_symmetrized
                ),
            )
            for grade in common
        )
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    def _pair_standard_words_with_ordered_signature(
        self,
        standard_words,
        ordered_standard_signature,
        *,
        words_first_on: bool,
        standard_first_on: bool,
    ):
        words, signature = self._prepare_standard_pairing_operands(
            standard_words,
            ordered_standard_signature,
            words_first_on=words_first_on,
            standard_first_on=standard_first_on,
            expected_partially_symmetrized=False,
        )
        return self._pair_prepared_standard_words(
            words,
            signature,
            standard_partially_symmetrized=False,
        )

    def _pair_standard_words_with_standard_tensor(
        self,
        standard_words,
        standard_tensor,
        *,
        words_first_on: bool,
        standard_first_on: bool,
    ):
        partially_symmetrized = getattr(
            getattr(standard_tensor, "spec", None),
            "partially_symmetrized",
            None,
        )
        if partially_symmetrized is False:
            return self._pair_standard_words_with_ordered_signature(
                standard_words,
                standard_tensor,
                words_first_on=words_first_on,
                standard_first_on=standard_first_on,
            )
        words, signature = self._prepare_standard_pairing_operands(
            standard_words,
            standard_tensor,
            words_first_on=words_first_on,
            standard_first_on=standard_first_on,
            expected_partially_symmetrized=True,
        )
        return self._pair_prepared_standard_words(
            words,
            signature,
            standard_partially_symmetrized=True,
        )

    def tensor_to_total(self, A):
        del A
        raise RuntimeError(
            "tensor_to_total is unavailable for partially symmetrized "
            "tensors. To pair partially symmetrized words with an "
            "ordered or partially symmetrized standard tensor, use "
            "tensor_shear_pairing."
        )

    def tensor_from_total(self, levels, *, trunc=None, include_scalar=True):
        del levels, trunc, include_scalar
        raise RuntimeError(
            "tensor_from_total is unavailable for partially symmetrized "
            "tensors; construct an ordered bidegree tensor and apply "
            "tensor_partially_symmetrize."
        )


__all__ = ["PartiallySymmetrizedBigradedCore"]
