"""Ordered-coordinate bidegree tensor algebra.

This module owns the grading-specific block primitives.  Outer grade
convolutions, contractions, and formal series are delegated to the same
drivers used by the total-degree core.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

import numpy as np

from tensordev.core.bigraded.layout import BigradedLayout
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.bigraded.shuffle import BigradedShufflePlanStore
from tensordev.core.bigraded.types import Bidegree, BigradedTensor, _bidegree
from tensordev.core.grading import (
    GradedContractionSchedule,
    GradedConvolutionSchedule,
    GradedInnerProductSchedule,
    GradedMapSchedule,
    GradedSummationSchedule,
    graded_horner_first_level,
)
from tensordev.core.shuffle import _normalize_precompute_shuffle
from tensordev.core.universal import Universal
from tensordev.core.utils.annotations import jit as dummy_jit
from tensordev.core.utils.pytrees import tree_first_leaf


class StandardBigradedCore(Universal):
    """Backend-neutral orchestration for a bounded ordered bidegree core.

    Concrete array backends implement only the functional scatter hooks.  The
    JAX implementation is provided by :class:`JaxBigraded`.
    """

    grading = "bidegree"
    representation = "ordered"
    coordinates = "standard"

    @property
    def capabilities(self) -> frozenset[str]:
        capabilities = {
            "concatenation",
            "generator_action",
            "coordinate_conversion",
            "signature_pairing",
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
        plan_store: BigradedPlanStore | None = None,
        shuffle_plan_store: BigradedShufflePlanStore | None = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        super().__init__(xp)
        shuffle_scope = _normalize_precompute_shuffle(
            precompute_shuffle,
            allow_generator=True,
        )
        if plan_store is None:
            if dims is None or max_trunc is None:
                raise TypeError("dims and max_trunc are required for a bidegree core.")
            plan_store = BigradedPlanStore(dims, max_trunc)
        else:
            if dims is not None and _bidegree(dims, name="dims") != plan_store.dims:
                raise ValueError("dims disagree with the supplied plan_store.")
            if max_trunc is not None and _bidegree(
                max_trunc, name="max_trunc"
            ) != plan_store.max_truncation:
                raise ValueError("max_trunc disagrees with the supplied plan_store.")

        self.plan_store = plan_store
        self.dims = plan_store.dims
        self.max_truncation = plan_store.max_truncation
        self.default_truncation = self._normalize_truncation(
            self.max_truncation if default_trunc is None else default_trunc
        )
        if shuffle_plan_store is not None:
            if shuffle_scope != "none":
                raise ValueError(
                    "precompute_shuffle and shuffle_plan_store are mutually "
                    "exclusive."
                )
            if shuffle_plan_store.plan_store is not plan_store:
                raise ValueError("shuffle_plan_store must reference this core's plan_store.")
            self.shuffle_plan_store = shuffle_plan_store
        elif shuffle_scope != "none":
            self.shuffle_plan_store = BigradedShufflePlanStore(
                plan_store,
                scope=shuffle_scope,
            )
        else:
            self.shuffle_plan_store = None
        self._truncation_views = {self.default_truncation: self}

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            f"max_trunc={self.max_truncation}, "
            f"default_trunc={self.default_truncation})"
        )

    # ------------------------------------------------------------------
    # Layout, validation, and construction
    # ------------------------------------------------------------------

    def _normalize_truncation(self, trunc: object | None) -> Bidegree:
        if trunc is None:
            return self.default_truncation
        normalized = _bidegree(trunc, name="trunc")
        if (
            normalized[0] > self.max_truncation[0]
            or normalized[1] > self.max_truncation[1]
        ):
            raise ValueError(
                f"active truncation {normalized} exceeds core capacity "
                f"{self.max_truncation}."
            )
        return normalized

    def normalize_truncation(self, trunc: object | None) -> Bidegree:
        """Public protocol hook used by developments and signatures."""
        return self._normalize_truncation(trunc)

    def prepare_development_input(
        self,
        X,
        *,
        trunc: Bidegree,
        increment_input: bool,
        axis: int,
    ):
        """Prepare a first-level path while preserving the combined alphabet."""
        X = tuple(X)
        if len(X) != 1:
            raise ValueError(
                "bidegree free developments accept exactly one "
                "combined first-level path array."
            )
        level = self.xp.asarray(X[0])
        if level.shape[-1] != sum(self.dims):
            raise ValueError(
                f"path width {level.shape[-1]} does not match split dimensions {self.dims}."
            )
        increment = level if increment_input else self.xp.diff(level, axis=axis)
        return (increment,)

    def resolve_layout(
        self,
        trunc: object | None = None,
        *,
        include_scalar: bool = True,
    ) -> BigradedLayout:
        return self._resolve_layout_coordinates(
            trunc,
            include_scalar=include_scalar,
            coordinates=self.coordinates,
        )

    def _resolve_layout_coordinates(
        self,
        trunc: object | None,
        *,
        include_scalar: bool,
        coordinates: str,
    ) -> BigradedLayout:
        return self.plan_store.resolve(
            self._normalize_truncation(trunc),
            include_scalar=include_scalar,
            coordinates=coordinates,
        )

    def _coordinate_identity(
        self,
        element: BigradedTensor,
        *,
        trunc: Bidegree | None = None,
        first_on: bool = False,
    ) -> BigradedTensor:
        """Validated identity map for ordered bidegree coordinates."""
        element = self._validate_element(element, name="element")
        if first_on == element.spec.include_scalar:
            expected = "omit" if first_on else "include"
            raise ValueError(
                f"first_on={first_on} requires the element to {expected} the scalar block."
            )
        requested = (
            element.truncation
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
        if element.spec == layout.spec:
            return element
        return BigradedTensor(
            tuple(element[grade] for grade in layout.grades),
            layout.spec,
        )

    def _validate_alphabet_dim(self, alphabet_dim: int) -> int:
        alphabet_dim = super()._validate_alphabet_dim(alphabet_dim)
        expected = sum(self.dims)
        if alphabet_dim != expected:
            raise ValueError(
                f"alphabet dimension {alphabet_dim} does not match bidegree "
                f"dimensions {self.dims} (total {expected})."
            )
        return alphabet_dim

    def _block_width_for_layout(
            self,
            layout: BigradedLayout,
            grade: Bidegree,
            *,
            alphabet_dim: int,
    ) -> int:
        self._validate_alphabet_dim(alphabet_dim)
        return layout.block_width(grade)

    def _element_block(
            self,
            element: BigradedTensor,
            grade: Bidegree,
            *,
            layout: BigradedLayout,
    ):
        element = self._validate_element(element, name="element")
        if element.spec != layout.spec:
            raise ValueError(
                "element layout does not match the resolved active layout."
            )
        return element[grade]

    def _assemble_element(self, blocks, layout: BigradedLayout) -> BigradedTensor:
        blocks = tuple(blocks)
        if len(blocks) != len(layout.grades):
            raise ValueError(
                f"received {len(blocks)} blocks, expected {len(layout.grades)} "
                "for the resolved layout."
            )
        return BigradedTensor(blocks, layout.spec)

    def at_truncation(self, trunc: Bidegree):
        """Return a cheap core view sharing this core's capacity plans."""
        active = self._normalize_truncation(trunc)
        cached = self._truncation_views.get(active)
        if cached is not None:
            return cached
        view = type(self)(
            plan_store=self.plan_store,
            shuffle_plan_store=self.shuffle_plan_store,
            default_trunc=active,
            precompute_shuffle=False,
        )
        view._truncation_views = self._truncation_views
        self._truncation_views[active] = view
        return view

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        """All eager base and optional shuffle plan payloads by category."""
        categories = dict(self.plan_store.memory_bytes_by_category())
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

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        statistics = dict(self.plan_store.plan_statistics())
        categories = self.memory_bytes_by_category()
        statistics["base_memory_bytes"] = self.plan_store.memory_bytes()
        statistics["shuffle_memory_bytes"] = (
            0
            if self.shuffle_plan_store is None
            else self.shuffle_plan_store.memory_bytes()
        )
        statistics["bytes_by_category"] = categories
        statistics["memory_bytes"] = sum(categories.values())
        statistics["memory_mb"] = sum(categories.values()) / 1024**2
        statistics["shuffle_enabled"] = self.shuffle_plan_store is not None
        statistics["shuffle_scope"] = (
            "none"
            if self.shuffle_plan_store is None
            else self.shuffle_plan_store.scope
        )
        return statistics

    def _validate_element(self, tensor: Any, *, name: str) -> BigradedTensor:
        return self._validate_element_coordinates(
            tensor,
            name=name,
            coordinates=self.coordinates,
        )

    def _validate_element_coordinates(
        self,
        tensor: Any,
        *,
        name: str,
        coordinates: str,
    ) -> BigradedTensor:
        if not isinstance(tensor, BigradedTensor):
            raise TypeError(f"{name} must be a BigradedTensor, got {type(tensor).__name__}.")
        if tensor.spec.dims != self.dims:
            raise ValueError(
                f"{name} uses alphabet dimensions {tensor.spec.dims}, expected {self.dims}."
            )
        if tensor.spec.coordinates != coordinates:
            raise ValueError(
                f"{name} uses coordinates {tensor.spec.coordinates!r}, expected {coordinates!r}."
            )
        if tensor.spec.representation != self.representation:
            raise ValueError(
                f"{name} uses representation {tensor.spec.representation!r}, "
                f"expected {self.representation!r}."
            )
        self._normalize_truncation(tensor.truncation)
        return tensor

    def _validate_graded_element(self, tensor: Any, *, name: str) -> BigradedTensor:
        return self._validate_element(tensor, name=name)

    def _result_dtype(self, *tensors: BigradedTensor):
        dtypes = [tree_first_leaf(tensor).dtype for tensor in tensors if tensor.blocks]
        return self.xp.result_type(*dtypes) if dtypes else self.xp.asarray(0.0).dtype

    def _result_batch(self, *tensors: BigradedTensor) -> tuple[int, ...]:
        shapes = [tensor.batch_shape for tensor in tensors if tensor.blocks]
        return self.xp.broadcast_shapes(*shapes) if shapes else ()

    def _zero_block(
        self,
        layout: BigradedLayout,
        grade: Bidegree,
        *,
        batch: tuple[int, ...],
        dtype: Any,
    ):
        return self.xp.zeros(batch + (layout.block_width(grade),), dtype=dtype)

    def _constant_element(
        self,
        layout: BigradedLayout,
        *,
        batch: tuple[int, ...],
        dtype: Any,
        scalar: float = 0.0,
    ) -> BigradedTensor:
        blocks = []
        for grade in layout.grades:
            if grade == (0, 0):
                block = self.xp.full(
                    batch + (layout.block_width(grade),),
                    scalar,
                    dtype=dtype,
                )
            else:
                block = self._zero_block(
                    layout, grade, batch=batch, dtype=dtype
                )
            blocks.append(block)
        return BigradedTensor(tuple(blocks), layout.spec)

    def _zero_block_for_layout(
            self,
            layout: BigradedLayout,
            grade: Bidegree,
            *,
            batch_shape: tuple[int, ...],
            dtype: Any,
            alphabet_dim: int,
    ):
        self._validate_alphabet_dim(alphabet_dim)
        return self._zero_block(
            layout, grade, batch=tuple(batch_shape), dtype=dtype
        )

    def _constant_element_for_layout(
            self,
            layout: BigradedLayout,
            *,
            batch_shape: tuple[int, ...],
            dtype: Any,
            alphabet_dim: int,
            scalar: float = 0.0,
    ) -> BigradedTensor:
        self._validate_alphabet_dim(alphabet_dim)
        return self._constant_element(
            layout,
            batch=tuple(batch_shape),
            dtype=dtype,
            scalar=scalar,
        )

    def _reframe(
        self,
        tensor: BigradedTensor,
        layout: BigradedLayout,
        *,
        batch: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        scalar_if_missing: float = 0.0,
    ) -> BigradedTensor:
        tensor = self._validate_element_coordinates(
            tensor,
            name="tensor",
            coordinates=layout.coordinates,
        )
        batch = tensor.batch_shape if batch is None else batch
        dtype = tensor.dtype if dtype is None else dtype
        blocks = []
        for grade in layout.grades:
            if tensor.spec.contains(grade):
                block = self.xp.broadcast_to(
                    tensor[grade], batch + (layout.block_width(grade),)
                ).astype(dtype)
            else:
                block = self._zero_block(layout, grade, batch=batch, dtype=dtype)
                if grade == (0, 0):
                    block = block + self.xp.asarray(scalar_if_missing, dtype=dtype)
            blocks.append(block)
        return BigradedTensor(tuple(blocks), layout.spec)

    # ------------------------------------------------------------------
    # Backend scatter hooks
    # ------------------------------------------------------------------

    def _placement_scatter(self, output, target_ranks, values):
        raise NotImplementedError("The concrete backend must implement functional scatter.")

    def _placement_scatter_add(self, output, target_ranks, values):
        raise NotImplementedError("The concrete backend must implement functional scatter-add.")

    def _coordinate_scatter(self, output, target_indices, values):
        raise NotImplementedError("The concrete backend must implement functional scatter.")

    # ------------------------------------------------------------------
    # Componentwise operations
    # ------------------------------------------------------------------

    def _bigraded_summation_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree] = None,
        *,
        coordinates: str,
    ) -> GradedSummationSchedule:
        A = self._validate_element_coordinates(A, name="A", coordinates=coordinates)
        B = self._validate_element_coordinates(B, name="B", coordinates=coordinates)
        include_scalar = A.spec.include_scalar or B.spec.include_scalar
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=include_scalar,
            coordinates=coordinates,
        )
        batch = self._result_batch(A, B)
        dtype = self._result_dtype(A, B)
        broadcast = lambda block, grade: self.xp.broadcast_to(
            block, batch + (layout.block_width(grade),)
        ).astype(dtype)
        return GradedSummationSchedule(
            grades=layout.grades,
            left_contains=A.spec.contains,
            right_contains=B.spec.contains,
            left_block=A.__getitem__,
            right_block=B.__getitem__,
            assemble=lambda blocks: BigradedTensor(blocks, layout.spec),
            add=lambda left, right, grade: broadcast(left, grade)
            + broadcast(right, grade),
            left_only=broadcast,
            right_only=broadcast,
            zero_block=lambda grade: self._zero_block(
                layout, grade, batch=batch, dtype=dtype
            ),
        )

    def _standard_summation_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree] = None,
    ) -> GradedSummationSchedule:
        return self._bigraded_summation_schedule(
            A, B, trunc, coordinates="standard"
        )

    def _summation_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree] = None,
    ) -> GradedSummationSchedule:
        return self._bigraded_summation_schedule(
            A, B, trunc, coordinates=self.coordinates
        )

    def _dilation_schedule(self, A: BigradedTensor) -> GradedMapSchedule:
        A = self._validate_element(A, name="A")
        return GradedMapSchedule(
            grades=A.grades,
            block=A.__getitem__,
            assemble=lambda blocks: BigradedTensor(blocks, A.spec),
        )

    def _grade_total_degree(self, grade: Bidegree) -> int:
        return grade[0] + grade[1]

    @dummy_jit(
        static_argnums=0,
        dynamic_batch=("A",),
        full_dynamic=("c_prime", "c_doubleprime"),
    )
    def tensor_bidilation(
        self,
        A: BigradedTensor,
        c_prime: Any,
        c_doubleprime: Any,
    ) -> BigradedTensor:
        A = self._validate_element(A, name="A")
        if not A.blocks:
            return A
        cp = self.xp.asarray(c_prime, dtype=A.blocks[0].dtype)
        cd = self.xp.asarray(c_doubleprime, dtype=A.blocks[0].dtype)
        blocks = tuple(
            block
            * self.xp.expand_dims(cp**grade[0] * cd**grade[1], axis=-1)
            for grade, block in zip(A.grades, A.blocks)
        )
        return BigradedTensor(blocks, A.spec)

    def _bigraded_inner_product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        *,
        coordinates: str,
        induced_pairing: bool = False,
    ) -> GradedInnerProductSchedule:
        A = self._validate_element_coordinates(
            A, name="A", coordinates=coordinates
        )
        B = self._validate_element_coordinates(
            B, name="B", coordinates=coordinates
        )
        common = tuple(grade for grade in A.grades if B.spec.contains(grade))
        return GradedInnerProductSchedule(
            grades=common,
            left_block=A.__getitem__,
            right_block=B.__getitem__,
            zero=(
                (lambda: self._standard_pairing_zero(A, B))
                if induced_pairing
                else (
                    lambda: self.xp.asarray(
                        0.0, dtype=self._result_dtype(A, B)
                    )
                )
            ),
        )

    def _standard_inner_product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
    ) -> GradedInnerProductSchedule:
        return self._bigraded_inner_product_schedule(
            A, B, coordinates="standard"
        )

    def _standard_pairing_inner_product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        *,
        a_first_on: bool,
        b_first_on: bool,
    ) -> GradedInnerProductSchedule:
        del a_first_on, b_first_on
        return self._bigraded_inner_product_schedule(
            A,
            B,
            coordinates="standard",
            induced_pairing=True,
        )

    def _inner_product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
    ) -> GradedInnerProductSchedule:
        return self._bigraded_inner_product_schedule(
            A, B, coordinates=self.coordinates
        )

    # ------------------------------------------------------------------
    # Concatenation and its adjoints
    # ------------------------------------------------------------------

    def _expanded_block_shape(self, grade: Bidegree) -> tuple[int, ...]:
        meta = self.plan_store.grade_plan(grade)
        dp, dd = self.dims
        return (meta.placement_count,) + (dp,) * grade[0] + (dd,) * grade[1]

    def _factored_concat_values(self, coordinate_pairs, plan):
        """Arrange coordinate pairs as updates to an output placement axis."""
        output_meta = self.plan_store.grade_plan(plan.output_grade)
        prefix = coordinate_pairs.shape[:-2]
        left_shape = self._expanded_block_shape(plan.left_grade)
        right_shape = self._expanded_block_shape(plan.right_grade)
        raw = coordinate_pairs.reshape(prefix + left_shape + right_shape)
        prefix_ndim = len(prefix)
        permutation = tuple(range(prefix_ndim)) + tuple(
            prefix_ndim + axis for axis in plan.outer_axis_permutation
        )
        return self.xp.transpose(raw, permutation).reshape(
            prefix
            + (
                plan.left_placement_count * plan.right_placement_count,
                output_meta.dense_shape[1],
                output_meta.dense_shape[2],
            )
        )

    def _factored_concat_scatter(self, coordinate_pairs, plan):
        """Assemble coordinate pairs with the factored concatenation plan."""
        output_meta = self.plan_store.grade_plan(plan.output_grade)
        prefix = coordinate_pairs.shape[:-2]
        values = self._factored_concat_values(coordinate_pairs, plan)
        output = self.xp.zeros(
            prefix + output_meta.dense_shape, dtype=coordinate_pairs.dtype
        )
        target_ranks = self.xp.asarray(plan.target_ranks().reshape(-1))
        output = self._placement_scatter(output, target_ranks, values)
        return output.reshape(prefix + (output_meta.block_width,))

    def _factored_concat_pull(self, target, plan):
        """Pull an output block back to left/right coordinate pairs."""
        left_meta = self.plan_store.grade_plan(plan.left_grade)
        right_meta = self.plan_store.grade_plan(plan.right_grade)
        output_meta = self.plan_store.grade_plan(plan.output_grade)
        batch = target.shape[:-1]
        dense = target.reshape(batch + output_meta.dense_shape)
        target_ranks = self.xp.asarray(plan.target_ranks())
        selected = self.xp.take(dense, target_ranks, axis=-3)

        left_shape = self._expanded_block_shape(plan.left_grade)
        right_shape = self._expanded_block_shape(plan.right_grade)
        raw_shape = left_shape + right_shape
        permuted_shape = tuple(raw_shape[axis] for axis in plan.outer_axis_permutation)
        selected = selected.reshape(batch + permuted_shape)
        inverse = [0] * len(plan.outer_axis_permutation)
        for output_axis, input_axis in enumerate(plan.outer_axis_permutation):
            inverse[input_axis] = output_axis
        batch_ndim = len(batch)
        permutation = tuple(range(batch_ndim)) + tuple(
            batch_ndim + axis for axis in inverse
        )
        return self.xp.transpose(selected, permutation).reshape(
            batch + (left_meta.block_width, right_meta.block_width)
        )

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
        coordinate_pairs = left[..., :, None] * right[..., None, :]
        return self._factored_concat_scatter(coordinate_pairs, plan)

    def _product_block(
        self,
        left,
        right,
        left_grade: Bidegree,
        right_grade: Bidegree,
        output_grade: Bidegree,
    ):
        return self._standard_product_block(
            left, right, left_grade, right_grade, output_grade
        )

    def _standard_product_output_block(self, contributions, output_grade: Bidegree):
        """Fuse all Chen-product splits into one output placement scatter.

        A single bidegree split only occupies selected output placements.  The
        homogeneous block primitive therefore has to materialize a zero-filled
        output.  A full graded product has several such splits, so assembling
        their raw placement updates together avoids one full zero tensor and
        one full block addition per split.
        """
        output_meta = self.plan_store.grade_plan(output_grade)
        batch = self.xp.broadcast_shapes(
            *(block.shape[:-1] for pair in contributions for block in pair[:2])
        )
        dtype = self.xp.result_type(
            *(block.dtype for pair in contributions for block in pair[:2])
        )
        values = []
        target_ranks = []
        for left, right, left_grade, right_grade in contributions:
            plan = self.plan_store.concat_plan(left_grade, right_grade)
            if plan.output_grade != output_grade:
                raise ValueError("product plan/output grade mismatch")
            left_meta = self.plan_store.grade_plan(left_grade)
            right_meta = self.plan_store.grade_plan(right_grade)
            left = self.xp.broadcast_to(left, batch + (left_meta.block_width,))
            right = self.xp.broadcast_to(right, batch + (right_meta.block_width,))
            coordinate_pairs = left[..., :, None] * right[..., None, :]
            values.append(self._factored_concat_values(coordinate_pairs, plan))
            target_ranks.append(plan.target_ranks().reshape(-1))

        if len(values) == 1:
            updates = values[0]
            ranks = self.xp.asarray(target_ranks[0])
            output = self.xp.zeros(
                batch + output_meta.dense_shape, dtype=dtype
            )
            output = self._placement_scatter(output, ranks, updates)
        else:
            updates = self.xp.concatenate(tuple(values), axis=-3)
            ranks = self.xp.asarray(np.concatenate(target_ranks))
            output = self.xp.zeros(
                batch + output_meta.dense_shape, dtype=dtype
            )
            output = self._placement_scatter_add(output, ranks, updates)
        return output.reshape(batch + (output_meta.block_width,))

    def _product_grade(self, contributions, output_grade: Bidegree):
        """Forward to the fused standard output-grade kernel."""
        return self._standard_product_output_block(contributions, output_grade)

    def _product_output_block(self, contributions, output_grade: Bidegree):
        return self._standard_product_output_block(contributions, output_grade)

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
        left_grade: Bidegree | None = None,
        right_grade: Bidegree | None = None,
    ):
        if left_grade is None or right_grade is None:
            raise TypeError(
                "bidegree homogeneous products require left_grade= and right_grade=."
            )
        left_grade = _bidegree(left_grade, name="left_grade")
        right_grade = _bidegree(right_grade, name="right_grade")
        output = left_grade[0] + right_grade[0], left_grade[1] + right_grade[1]
        return self._standard_product_block(Ai, Bj, left_grade, right_grade, output)

    def _standard_product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree] = None,
        *,
        a_first_on: bool = False,
        b_first_on: bool = False,
        first_on_out: bool = False,
    ) -> GradedConvolutionSchedule:
        A = self._validate_element_coordinates(A, name="A", coordinates="standard")
        B = self._validate_element_coordinates(B, name="B", coordinates="standard")
        if a_first_on and A.spec.include_scalar:
            raise ValueError("a_first_on=True conflicts with A including the scalar block.")
        if b_first_on and B.spec.include_scalar:
            raise ValueError("b_first_on=True conflicts with B including the scalar block.")
        include_scalar = (
            not first_on_out
            and A.spec.include_scalar
            and B.spec.include_scalar
        )
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=include_scalar,
            coordinates="standard",
        )
        batch = self._result_batch(A, B)
        dtype = self._result_dtype(A, B)
        return GradedConvolutionSchedule(
            grades=layout.grades,
            splits=layout.product_splits,
            left_contains=A.spec.contains,
            right_contains=B.spec.contains,
            left_block=A.__getitem__,
            right_block=B.__getitem__,
            assemble=lambda blocks: BigradedTensor(blocks, layout.spec),
            zero_block=lambda grade: self._zero_block(
                layout, grade, batch=batch, dtype=dtype
            ),
            product_grade=self._standard_product_output_block,
        )

    def _product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree] = None,
        *,
        a_first_on: bool = False,
        b_first_on: bool = False,
        first_on_out: bool = False,
    ) -> GradedConvolutionSchedule:
        return self._standard_product_schedule(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )

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
        mult_width = self.plan_store.grade_plan(multiplier_grade).block_width
        target_width = self.plan_store.grade_plan(target_grade).block_width
        batch = self.xp.broadcast_shapes(multiplier.shape[:-1], target.shape[:-1])
        multiplier = self.xp.broadcast_to(multiplier, batch + (mult_width,))
        target = self.xp.broadcast_to(target, batch + (target_width,))
        selected = self._factored_concat_pull(target, plan)
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
        mult_width = self.plan_store.grade_plan(multiplier_grade).block_width
        target_width = self.plan_store.grade_plan(target_grade).block_width
        batch = self.xp.broadcast_shapes(multiplier.shape[:-1], target.shape[:-1])
        multiplier = self.xp.broadcast_to(multiplier, batch + (mult_width,))
        target = self.xp.broadcast_to(target, batch + (target_width,))
        selected = self._factored_concat_pull(target, plan)
        return (selected * multiplier[..., None, :]).sum(axis=-1)

    def _adjoint_left_block(
        self,
        multiplier,
        target,
        multiplier_grade: Bidegree,
        target_grade: Bidegree,
        output_grade: Bidegree,
    ):
        return self._standard_adjoint_left_block(
            multiplier,
            target,
            multiplier_grade,
            target_grade,
            output_grade,
        )

    def _adjoint_right_block(
        self,
        multiplier,
        target,
        multiplier_grade: Bidegree,
        target_grade: Bidegree,
        output_grade: Bidegree,
    ):
        return self._standard_adjoint_right_block(
            multiplier,
            target,
            multiplier_grade,
            target_grade,
            output_grade,
        )

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
        multiplier_grade: Bidegree | None = None,
        output_grade: Bidegree | None = None,
    ):
        if multiplier_grade is None or output_grade is None:
            raise TypeError(
                "bidegree homogeneous adjoints require multiplier_grade= and output_grade=."
            )
        multiplier_grade = _bidegree(multiplier_grade, name="multiplier_grade")
        output_grade = _bidegree(output_grade, name="output_grade")
        target_grade = (
            multiplier_grade[0] + output_grade[0],
            multiplier_grade[1] + output_grade[1],
        )
        return self._standard_adjoint_left_block(
            Ai, Yni, multiplier_grade, target_grade, output_grade
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
        multiplier_grade: Bidegree | None = None,
        output_grade: Bidegree | None = None,
    ):
        if multiplier_grade is None or output_grade is None:
            raise TypeError(
                "bidegree homogeneous adjoints require multiplier_grade= and output_grade=."
            )
        multiplier_grade = _bidegree(multiplier_grade, name="multiplier_grade")
        output_grade = _bidegree(output_grade, name="output_grade")
        target_grade = (
            output_grade[0] + multiplier_grade[0],
            output_grade[1] + multiplier_grade[1],
        )
        return self._standard_adjoint_right_block(
            Bj, Ynj, multiplier_grade, target_grade, output_grade
        )

    def _standard_adjoint_schedule(
        self,
        W: BigradedTensor,
        Y: BigradedTensor,
        trunc: Optional[Bidegree] = None,
        *,
        contract,
        w_first_on: bool = False,
        y_first_on: bool = False,
        first_on_out: bool = False,
    ) -> GradedContractionSchedule:
        W = self._validate_element_coordinates(W, name="W", coordinates="standard")
        Y = self._validate_element_coordinates(Y, name="Y", coordinates="standard")
        if w_first_on and W.spec.include_scalar:
            raise ValueError("w_first_on=True conflicts with W including scalar.")
        if y_first_on and Y.spec.include_scalar:
            raise ValueError("y_first_on=True conflicts with Y including scalar.")
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=not first_on_out,
            coordinates="standard",
        )
        batch = self._result_batch(W, Y)
        dtype = self._result_dtype(W, Y)

        def pairs(output_grade: Bidegree):
            return tuple(
                (
                    multiplier_grade,
                    (
                        output_grade[0] + multiplier_grade[0],
                        output_grade[1] + multiplier_grade[1],
                    ),
                )
                for multiplier_grade in W.grades
                if Y.spec.contains(
                    (
                        output_grade[0] + multiplier_grade[0],
                        output_grade[1] + multiplier_grade[1],
                    )
                )
            )

        return GradedContractionSchedule(
            grades=layout.grades,
            pairs=pairs,
            multiplier_block=W.__getitem__,
            target_block=Y.__getitem__,
            assemble=lambda blocks: BigradedTensor(blocks, layout.spec),
            zero_block=lambda grade: self._zero_block(
                layout, grade, batch=batch, dtype=dtype
            ),
            contract_block=contract,
        )

    def _adjoint_schedule(
        self,
        W: BigradedTensor,
        Y: BigradedTensor,
        trunc: Optional[Bidegree] = None,
        *,
        contract,
        w_first_on: bool = False,
        y_first_on: bool = False,
        first_on_out: bool = False,
    ) -> GradedContractionSchedule:
        return self._standard_adjoint_schedule(
            W,
            Y,
            trunc,
            contract=contract,
            w_first_on=w_first_on,
            y_first_on=y_first_on,
            first_on_out=first_on_out,
        )

    # ------------------------------------------------------------------
    # Formal series and the pruned first-level Horner action
    # ------------------------------------------------------------------

    def _split_generator(self, z):
        z = self.xp.asarray(z)
        expected = self.dims[0] + self.dims[1]
        if z.shape[-1] != expected:
            raise ValueError(
                f"first-level generator has width {z.shape[-1]}, expected {expected}."
            )
        return z[..., : self.dims[0]], z[..., self.dims[0] :]

    def _generator_blocks(self, z, *, layout: BigradedLayout):
        self._normalize_truncation(layout.truncation)
        prime, doubleprime = self._split_generator(z)
        return (((1, 0), prime), ((0, 1), doubleprime))

    def _append_doubleprime(self, block, z_doubleprime, grade: Bidegree):
        n, m = grade
        source = self.plan_store.grade_plan((n, m - 1))
        batch = self.xp.broadcast_shapes(block.shape[:-1], z_doubleprime.shape[:-1])
        block = self.xp.broadcast_to(block, batch + (source.block_width,)).reshape(
            batch + source.dense_shape
        )
        z = self.xp.broadcast_to(z_doubleprime, batch + (self.dims[1],))
        value = block[..., None] * z[..., None, None, None, :]
        return value.reshape(
            batch
            + (
                source.placement_count,
                self.dims[0] ** n,
                self.dims[1] ** m,
            )
        )

    def _append_prime(self, block, z_prime, grade: Bidegree):
        n, m = grade
        source = self.plan_store.grade_plan((n - 1, m))
        batch = self.xp.broadcast_shapes(block.shape[:-1], z_prime.shape[:-1])
        block = self.xp.broadcast_to(block, batch + (source.block_width,)).reshape(
            batch + source.dense_shape
        )
        z = self.xp.broadcast_to(z_prime, batch + (self.dims[0],))
        raw = block[..., None] * z[..., None, None, None, :]
        # raw: batch, placement, prime_old, double, prime_new
        batch_ndim = len(batch)
        raw = self.xp.transpose(
            raw,
            tuple(range(batch_ndim))
            + (
                batch_ndim,
                batch_ndim + 1,
                batch_ndim + 3,
                batch_ndim + 2,
            ),
        )
        return raw.reshape(
            batch
            + (
                source.placement_count,
                self.dims[0] ** n,
                self.dims[1] ** m,
            )
        )

    def _resolve_generator_action_inputs(
        self,
        predecessor_blocks,
        generator_blocks,
        *,
        predecessor_grades,
        generator_grades,
        output_grade,
    ):
        """Validate and canonicalize the two bidegree generator splits.

        This is representation-neutral orchestration.  Ordered and partially
        symmetrized cores differ only in how the two resolved contributions
        are written into the output block.
        """
        predecessor_blocks, generator_blocks = self._validate_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )
        output_grade = _bidegree(output_grade, name="output_grade")
        predecessor_grades = tuple(
            _bidegree(grade, name="predecessor_grade")
            for grade in predecessor_grades
        )
        generator_grades = tuple(
            _bidegree(grade, name="generator_grade")
            for grade in generator_grades
        )
        supplied = {
            (source_grade, generator_grade): (source, generator)
            for source_grade, generator_grade, source, generator in zip(
                predecessor_grades,
                generator_grades,
                predecessor_blocks,
                generator_blocks,
            )
        }
        if len(supplied) != len(predecessor_blocks):
            raise ValueError("generator action contains duplicate grade pairs.")

        n, m = output_grade
        expected = []
        if m > 0:
            expected.append(((n, m - 1), (0, 1)))
        if n > 0:
            expected.append(((n - 1, m), (1, 0)))
        if set(supplied) != set(expected):
            raise ValueError(
                "supplied generator pairs are not the first-level splits of "
                f"output grade {output_grade}."
            )

        batch_shapes = [
            array.shape[:-1]
            for pair in supplied.values()
            for array in pair
        ]
        batch = self.xp.broadcast_shapes(*batch_shapes)
        return supplied, batch, output_grade

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
        pieces = []
        if m > 0:
            source, generator = supplied[((n, m - 1), (0, 1))]
            source_width = self.plan_store.grade_plan((n, m - 1)).block_width
            pieces.append(
                self._append_doubleprime(
                    self.xp.broadcast_to(source, batch + (source_width,)),
                    self.xp.broadcast_to(generator, batch + (self.dims[1],)),
                    output_grade,
                )
            )
        if n > 0:
            source, generator = supplied[((n - 1, m), (1, 0))]
            source_width = self.plan_store.grade_plan((n - 1, m)).block_width
            pieces.append(
                self._append_prime(
                    self.xp.broadcast_to(source, batch + (source_width,)),
                    self.xp.broadcast_to(generator, batch + (self.dims[0],)),
                    output_grade,
                )
            )
        if not pieces:
            raise ValueError("the scalar grade has no generator predecessors.")
        packed = pieces[0] if len(pieces) == 1 else self.xp.concat(pieces, axis=-3)
        width = self.plan_store.grade_plan(output_grade).block_width
        return packed.reshape(batch + (width,))

    def _fmexp_first_level(
        self,
        g: BigradedTensor,
        z,
        *,
        trunc: Bidegree,
    ) -> BigradedTensor:
        if not g.spec.include_scalar:
            raise ValueError("g must include the scalar block.")
        z = self.xp.asarray(z)
        layout = self.resolve_layout(trunc, include_scalar=True)
        batch = self.xp.broadcast_shapes(g.batch_shape, z.shape[:-1])
        dtype = self.xp.result_type(g.blocks[0], z)

        def base_block(grade: Bidegree, _like):
            if g.spec.contains(grade):
                return self.xp.broadcast_to(
                    g[grade], batch + (layout.block_width(grade),)
                ).astype(dtype)
            return self._zero_block(layout, grade, batch=batch, dtype=dtype)

        return graded_horner_first_level(
            layout,
            max_order=trunc[0] + trunc[1],
            base_block=base_block,
            generator_blocks=self._generator_blocks(z, layout=layout),
            right_generator_output_block=(
                self._right_multiply_generator_output_block
            ),
            assemble=lambda blocks: BigradedTensor(blocks, layout.spec),
        )

    def _series_argument(self, X) -> tuple[str, Any]:
        if isinstance(X, (tuple, list)):
            if len(X) == 0:
                return "empty", X
            if len(X) != 1:
                raise TypeError(
                    "higher-level bidegree series must be supplied as a BigradedTensor."
                )
            return "first", X[0]
        if isinstance(X, BigradedTensor):
            return ("empty", X) if not X.blocks else ("general", X)
        return "first", X

    def _series_order(self, trunc: Bidegree) -> int:
        return trunc[0] + trunc[1]

    def _validate_series_element(
        self,
        tensor: BigradedTensor,
        *,
        name: str,
        coordinates: str,
        include_scalar: bool,
    ) -> BigradedTensor:
        tensor = self._validate_element_coordinates(
            tensor, name=name, coordinates=coordinates
        )
        if tensor.spec.include_scalar != include_scalar:
            policy = "include" if include_scalar else "omit"
            raise ValueError(f"{name} must {policy} the scalar block.")
        return tensor

    def _series_validate_left_factor(self, g: BigradedTensor) -> BigradedTensor:
        return self._validate_series_element(
            g,
            name="g",
            coordinates=self.coordinates,
            include_scalar=True,
        )

    def _validate_series_exponential_argument(
        self,
        X,
        *,
        coordinates: str,
    ):
        if isinstance(X, BigradedTensor):
            return self._validate_series_element(
                X,
                name="X",
                coordinates=coordinates,
                include_scalar=False,
            )
        return X

    def _series_validate_exponential_argument(self, X):
        return self._validate_series_exponential_argument(
            X,
            coordinates=self.coordinates,
        )

    def _standard_series_validate_left_factor(
        self, g: BigradedTensor
    ) -> BigradedTensor:
        return self._validate_series_element(
            g,
            name="g",
            coordinates="standard",
            include_scalar=True,
        )

    def _standard_series_validate_exponential_argument(self, X):
        return self._validate_series_exponential_argument(
            X,
            coordinates="standard",
        )

    def _truncate_series_left_factor(
        self,
        g: BigradedTensor,
        trunc: Bidegree,
        *,
        coordinates: str,
    ) -> BigradedTensor:
        g = self._validate_element_coordinates(
            g, name="g", coordinates=coordinates
        )
        return self._reframe(
            g,
            self._resolve_layout_coordinates(
                trunc,
                include_scalar=True,
                coordinates=coordinates,
            ),
        )

    def _series_truncate_left_factor(
        self,
        g: BigradedTensor,
        trunc: Bidegree,
    ) -> BigradedTensor:
        return self._truncate_series_left_factor(
            g, trunc, coordinates=self.coordinates
        )

    def _standard_series_truncate_left_factor(
        self,
        g: BigradedTensor,
        trunc: Bidegree,
    ) -> BigradedTensor:
        return self._truncate_series_left_factor(
            g, trunc, coordinates="standard"
        )

    def _series_exponential_data_coordinates(
        self,
        X: BigradedTensor,
        *,
        trunc: Bidegree,
        left_factor: BigradedTensor,
        coordinates: str,
    ) -> tuple[BigradedTensor, BigradedTensor, int]:
        X = self._validate_element_coordinates(
            X, name="X", coordinates=coordinates
        )
        if X.spec.include_scalar:
            raise ValueError("X must omit the scalar block.")
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=True,
            coordinates=coordinates,
        )
        batch = self._result_batch(left_factor, X)
        dtype = self._result_dtype(left_factor, X)
        H = self._reframe(
            X, layout, batch=batch, dtype=dtype, scalar_if_missing=0.0
        )
        identity = self._constant_element(
            layout, batch=batch, dtype=dtype, scalar=1.0
        )
        return H, identity, self._series_order(trunc)

    def _series_exponential_data(
        self,
        X: BigradedTensor,
        *,
        trunc: Bidegree,
        left_factor: BigradedTensor,
    ) -> tuple[BigradedTensor, BigradedTensor, int]:
        return self._series_exponential_data_coordinates(
            X,
            trunc=trunc,
            left_factor=left_factor,
            coordinates=self.coordinates,
        )

    def _standard_series_exponential_data(
        self,
        X: BigradedTensor,
        *,
        trunc: Bidegree,
        left_factor: BigradedTensor,
    ) -> tuple[BigradedTensor, BigradedTensor, int]:
        return self._series_exponential_data_coordinates(
            X,
            trunc=trunc,
            left_factor=left_factor,
            coordinates="standard",
        )

    def _series_identity_for_argument(
        self,
        X,
        *,
        trunc: Bidegree,
    ) -> BigradedTensor:
        kind, argument = self._series_argument(X)
        if kind == "first":
            layout = self.resolve_layout(trunc, include_scalar=True)
            z = self.xp.asarray(argument)
            return self._constant_element(
                layout, batch=z.shape[:-1], dtype=z.dtype, scalar=1.0
            )
        argument = self._validate_element_coordinates(
            argument, name="X", coordinates=argument.spec.coordinates
        )
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=True,
            coordinates=argument.spec.coordinates,
        )
        dtype = (
            argument.blocks[0].dtype
            if argument.blocks
            else self.xp.asarray(1.0).dtype
        )
        return self._constant_element(
            layout, batch=argument.batch_shape, dtype=dtype, scalar=1.0
        )

    def _series_exponential_zero(
        self,
        *,
        trunc: Bidegree,
        output_zero_level: bool,
    ) -> BigradedTensor:
        layout = self.resolve_layout(trunc, include_scalar=output_zero_level)
        return self._constant_element(
            layout,
            batch=(),
            dtype=self.xp.asarray(1.0).dtype,
            scalar=1.0,
        )

    def _series_validate_log_argument(self, X) -> BigradedTensor:
        return self._validate_series_element(
            X,
            name="X",
            coordinates=self.coordinates,
            include_scalar=False,
        )

    def _standard_series_validate_log_argument(self, X) -> BigradedTensor:
        return self._validate_series_element(
            X,
            name="X",
            coordinates="standard",
            include_scalar=False,
        )

    def _series_logarithm_data_coordinates(
        self,
        X: BigradedTensor,
        *,
        trunc: Bidegree,
        coordinates: str,
    ) -> tuple[BigradedTensor, BigradedTensor, int]:
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=True,
            coordinates=coordinates,
        )
        dtype = X.blocks[0].dtype if X.blocks else self.xp.asarray(0.0).dtype
        H = self._reframe(X, layout, dtype=dtype, scalar_if_missing=0.0)
        zero = self._constant_element(
            layout, batch=X.batch_shape, dtype=dtype, scalar=0.0
        )
        return H, zero, self._series_order(trunc)

    def _series_logarithm_data(
        self,
        X: BigradedTensor,
        *,
        trunc: Bidegree,
    ) -> tuple[BigradedTensor, BigradedTensor, int]:
        return self._series_logarithm_data_coordinates(
            X, trunc=trunc, coordinates=self.coordinates
        )

    def _standard_series_logarithm_data(
        self,
        X: BigradedTensor,
        *,
        trunc: Bidegree,
    ) -> tuple[BigradedTensor, BigradedTensor, int]:
        return self._series_logarithm_data_coordinates(
            X, trunc=trunc, coordinates="standard"
        )

    def _series_logarithm_zero(
        self,
        *,
        trunc: Bidegree,
        output_zero_level: bool,
    ) -> BigradedTensor:
        layout = self.resolve_layout(trunc, include_scalar=output_zero_level)
        return self._constant_element(
            layout,
            batch=(),
            dtype=self.xp.asarray(0.0).dtype,
            scalar=0.0,
        )

    def _standard_series_logarithm_zero(
        self,
        *,
        trunc: Bidegree,
        output_zero_level: bool,
    ) -> BigradedTensor:
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=output_zero_level,
            coordinates="standard",
        )
        return self._constant_element(
            layout,
            batch=(),
            dtype=self.xp.asarray(0.0).dtype,
            scalar=0.0,
        )

    def _series_output(
        self,
        result: BigradedTensor,
        *,
        trunc: Bidegree,
        output_zero_level: bool,
    ) -> BigradedTensor:
        if output_zero_level:
            return result
        return self._reframe(
            result,
            self._resolve_layout_coordinates(
                trunc,
                include_scalar=False,
                coordinates=result.spec.coordinates,
            ),
        )

    def _standard_series_output(
        self,
        result: BigradedTensor,
        *,
        trunc: Bidegree,
        output_zero_level: bool,
    ) -> BigradedTensor:
        if output_zero_level:
            return result
        return self._reframe(
            result,
            self._resolve_layout_coordinates(
                trunc,
                include_scalar=False,
                coordinates="standard",
            ),
        )

    # ------------------------------------------------------------------
    # Packing and conversion to the dense total-degree representation
    # ------------------------------------------------------------------

    @dummy_jit(
        static_argnums=0,
        static_argnames=("start_at_level_one",),
        dynamic_batch=("levels",),
    )
    def tensor_to_flat(
        self,
        levels: BigradedTensor,
        *,
        start_at_level_one: bool = False,
    ):
        levels = self._validate_element(levels, name="levels")
        blocks = levels.blocks
        scalar_reference = blocks[0] if blocks else None
        if start_at_level_one and levels.spec.include_scalar:
            blocks = blocks[1:]
        if not blocks:
            if scalar_reference is not None:
                return scalar_reference[..., :0]
            return self.xp.asarray([], dtype=levels.dtype or float)
        return self.xp.concat(blocks, axis=-1)

    @dummy_jit(
        static_argnums=0,
        static_argnames=("dim", "insert_zero_level", "trunc", "include_scalar"),
        dynamic_batch=("flat",),
    )
    def tensor_from_flat(
        self,
        flat,
        dim: int | None = None,
        insert_zero_level: Optional[float | bool] = None,
        *,
        trunc: Optional[Bidegree] = None,
        include_scalar: bool = True,
    ) -> BigradedTensor:
        flat = self.xp.asarray(flat)
        if dim is not None and dim != sum(self.dims):
            raise ValueError(f"dim={dim} disagrees with split dimensions {self.dims}.")
        layout = self.resolve_layout(trunc, include_scalar=include_scalar)
        expected = sum(layout.block_width(grade) for grade in layout.grades)
        if flat.shape[-1] != expected:
            raise ValueError(
                f"flat width {flat.shape[-1]} does not match bidegree layout width {expected}."
            )
        blocks = []
        offset = 0
        for grade in layout.grades:
            width = layout.block_width(grade)
            block = flat[..., offset : offset + width]
            if grade == (0, 0) and insert_zero_level is not None:
                if insert_zero_level is True:
                    block = self.xp.ones_like(block)
                elif insert_zero_level is False:
                    block = self.xp.zeros_like(block)
                else:
                    block = self.xp.ones_like(block) * float(insert_zero_level)
            blocks.append(block)
            offset += width
        return BigradedTensor(tuple(blocks), layout.spec)

    def tensor_densify(
        self,
        levels: BigradedTensor | Mapping[Bidegree, Any],
        *,
        trunc: Optional[Bidegree] = None,
        include_scalar: bool = True,
    ) -> BigradedTensor:
        layout = self.resolve_layout(trunc, include_scalar=include_scalar)
        if isinstance(levels, BigradedTensor):
            outside = tuple(
                grade for grade in levels.grades if not layout.contains(grade)
            )
            if outside:
                raise ValueError(
                    f"input contains grades outside the requested layout: {outside}."
                )
            return self._reframe(levels, layout)
        if not isinstance(levels, Mapping):
            raise TypeError("levels must be a BigradedTensor or grade-to-array mapping.")
        present = { _bidegree(grade): block for grade, block in levels.items() if block is not None }
        outside = tuple(grade for grade in present if not layout.contains(grade))
        if outside:
            raise ValueError(
                f"input contains grades outside the requested layout: {outside}."
            )
        if not present:
            return self._constant_element(
                layout, batch=(), dtype=self.xp.asarray(0.0).dtype, scalar=0.0
            )
        reference = next(iter(present.values()))
        batch = tuple(reference.shape[:-1])
        dtype = reference.dtype
        blocks = []
        for grade in layout.grades:
            if grade in present:
                block = present[grade]
                if block.shape != batch + (layout.block_width(grade),):
                    raise ValueError(f"malformed block at grade {grade}.")
                blocks.append(block)
            else:
                blocks.append(self._zero_block(layout, grade, batch=batch, dtype=dtype))
        return BigradedTensor(tuple(blocks), layout.spec)

    @dummy_jit(static_argnums=0, dynamic_batch=("A",))
    def tensor_to_total(self, A: BigradedTensor):
        """Merge bidegree blocks into dense total levels without rebasing.

        This changes grading/layout, not coordinates.  In particular, calling
        it on a shear bidegree core returns dense total levels in shear
        coordinates; use a standard-coordinate core after an explicit
        coordinate conversion when standard dense levels are required.
        """
        A = self._validate_element(A, name="A")
        max_total = sum(A.truncation)
        dimension = sum(self.dims)
        batch = A.batch_shape
        dtype = A.blocks[0].dtype if A.blocks else self.xp.asarray(0.0).dtype
        levels = []
        for total in range(max_total + 1):
            level = self.xp.zeros(batch + (dimension**total,), dtype=dtype)
            for grade in A.grades:
                if sum(grade) != total:
                    continue
                indices = self.xp.asarray(
                    self.plan_store.grade_plan(grade).block_to_total_indices
                )
                level = self._coordinate_scatter(level, indices, A[grade])
            levels.append(level)
        return tuple(levels)

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "include_scalar"),
        dynamic_batch=("levels",),
    )
    def tensor_from_total(
        self,
        levels,
        *,
        trunc: Optional[Bidegree] = None,
        include_scalar: bool = True,
    ) -> BigradedTensor:
        """Split dense total levels into this core's bidegree coordinates.

        The input levels are interpreted in the core's current coordinate
        system.  The returned ``BigradedTensor`` is tagged ``"standard"`` or
        ``"shear"`` accordingly; this is a layout conversion, not the
        standard/shear coordinate map.
        """
        levels = tuple(levels)
        layout = self.resolve_layout(trunc, include_scalar=include_scalar)
        dimension = sum(self.dims)
        blocks = []
        for grade in layout.grades:
            total = sum(grade)
            if total >= len(levels):
                raise ValueError(f"ordinary tensor is missing total degree {total}.")
            level = levels[total]
            if level.shape[-1] != dimension**total:
                raise ValueError(
                    f"ordinary degree {total} has width {level.shape[-1]}, "
                    f"expected {dimension**total}."
                )
            indices = self.xp.asarray(
                self.plan_store.grade_plan(grade).block_to_total_indices
            )
            blocks.append(self.xp.take(level, indices, axis=-1))
        return BigradedTensor(tuple(blocks), layout.spec)

    # ------------------------------------------------------------------
    # Matrix-valued bidegree operations
    # ------------------------------------------------------------------

    def _matrix_map_schedule(
        self,
        A: BigradedTensor,
        trunc: Optional[Bidegree] = None,
    ) -> GradedMapSchedule:
        A = self._validate_element(A, name="A")
        layout = self.resolve_layout(trunc, include_scalar=A.spec.include_scalar)
        projected = self._reframe(A, layout)
        return GradedMapSchedule(
            grades=layout.grades,
            block=projected.__getitem__,
            assemble=lambda blocks: BigradedTensor(blocks, layout.spec),
        )

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
        right, _, _ = self._canonicalize_matrix_axes(right, row_axis, col_axis)
        pair = self.xp.einsum("...rki,...klj->...rlij", left, right)
        plan = self.plan_store.concat_plan(left_grade, right_grade)
        if plan.output_grade != output_grade:
            raise ValueError("matrix product plan/output grade mismatch")
        output = self._factored_concat_scatter(pair, plan)
        return self._restore_matrix_axes(output, original_row, original_col)

    def _matrix_product_block(
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
        return self._standard_matrix_product_block(
            left,
            right,
            left_grade,
            right_grade,
            output_grade,
            row_axis=row_axis,
            col_axis=col_axis,
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
        left_grade: Bidegree | None = None,
        right_grade: Bidegree | None = None,
    ):
        if left_grade is None or right_grade is None:
            raise TypeError(
                "bidegree homogeneous matrix products require left_grade= and right_grade=."
            )
        left_grade = _bidegree(left_grade, name="left_grade")
        right_grade = _bidegree(right_grade, name="right_grade")
        output_grade = (
            left_grade[0] + right_grade[0],
            left_grade[1] + right_grade[1],
        )
        return self._standard_matrix_product_block(
            A,
            B,
            left_grade,
            right_grade,
            output_grade,
            row_axis=row_axis,
            col_axis=col_axis,
        )

    def _standard_matrix_product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree] = None,
        *,
        row_axis: int = -3,
        col_axis: int = -2,
    ) -> GradedConvolutionSchedule:
        A = self._validate_element_coordinates(A, name="A", coordinates="standard")
        B = self._validate_element_coordinates(B, name="B", coordinates="standard")
        if not A.spec.include_scalar or not B.spec.include_scalar:
            raise ValueError("matrix tensor products require scalar-including elements.")
        layout = self._resolve_layout_coordinates(
            trunc,
            include_scalar=True,
            coordinates="standard",
        )

        def zero(grade):
            a0, _, _ = self._canonicalize_matrix_axes(
                A.blocks[0], row_axis, col_axis
            )
            b0, _, _ = self._canonicalize_matrix_axes(
                B.blocks[0], row_axis, col_axis
            )
            batch = self.xp.broadcast_shapes(a0.shape[:-3], b0.shape[:-3])
            canonical = self.xp.zeros(
                batch
                + (a0.shape[-3], b0.shape[-2], layout.block_width(grade)),
                dtype=self.xp.result_type(a0, b0),
            )
            return self._restore_matrix_axes(
                canonical, row_axis % A.blocks[0].ndim, col_axis % A.blocks[0].ndim
            )

        return GradedConvolutionSchedule(
            grades=layout.grades,
            splits=layout.product_splits,
            left_contains=A.spec.contains,
            right_contains=B.spec.contains,
            left_block=A.__getitem__,
            right_block=B.__getitem__,
            assemble=lambda blocks: BigradedTensor(blocks, layout.spec),
            zero_block=zero,
        )

    def _matrix_product_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree] = None,
        *,
        row_axis: int = -3,
        col_axis: int = -2,
    ) -> GradedConvolutionSchedule:
        return self._standard_matrix_product_schedule(
            A,
            B,
            trunc,
            row_axis=row_axis,
            col_axis=col_axis,
        )

    # ------------------------------------------------------------------
    # Standard shuffle
    # ------------------------------------------------------------------

    def _require_shuffle(self) -> BigradedShufflePlanStore:
        if self.shuffle_plan_store is None:
            raise RuntimeError(
                "This core was constructed with precompute_shuffle=False; "
                "construct a shuffle-enabled core to use shuffle operations."
            )
        return self.shuffle_plan_store

    def _require_full_shuffle(self) -> BigradedShufflePlanStore:
        store = self._require_shuffle()
        if store.scope != "full":
            raise RuntimeError(
                "Arbitrary shuffle products require precompute_shuffle=True; "
                "this core contains generator-only shuffle plans."
            )
        return store

    def _shuffle_schedule(
        self,
        A: BigradedTensor,
        B: BigradedTensor,
        trunc: Optional[Bidegree],
        *,
        a_first_on: bool,
        b_first_on: bool,
        first_on_out: bool,
    ) -> GradedConvolutionSchedule:
        """Reuse the bidegree convolution layout for a full shuffle."""
        self._require_full_shuffle()
        schedule = self._product_schedule(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )
        return GradedConvolutionSchedule(
            grades=schedule.grades,
            splits=schedule.splits,
            left_contains=schedule.left_contains,
            right_contains=schedule.right_contains,
            left_block=schedule.left_block,
            right_block=schedule.right_block,
            assemble=schedule.assemble,
            zero_block=schedule.zero_block,
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
        return self._require_shuffle().tensor_shuffle_product_homogeneous(
            self.xp, left, right, left_grade, right_grade
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
        predecessor_blocks, generator_blocks = self._validate_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )
        output_grade = _bidegree(output_grade, name="output_grade")
        predecessor_grades = tuple(
            _bidegree(grade, name="predecessor_grade")
            for grade in predecessor_grades
        )
        generator_grades = tuple(
            _bidegree(grade, name="generator_grade")
            for grade in generator_grades
        )
        expected = {
            (source_grade, generator_grade)
            for source_grade, generator_grade in self.resolve_layout(
                output_grade, include_scalar=True
            ).product_splits(output_grade)
            if sum(generator_grade) == 1
        }
        supplied = tuple(zip(predecessor_grades, generator_grades))
        if len(set(supplied)) != len(supplied) or set(supplied) != expected:
            raise ValueError(
                "supplied generator pairs are not the first-level splits of "
                f"output grade {output_grade}."
            )
        terms = tuple(
            self._shuffle_block(
                source,
                generator,
                source_grade,
                generator_grade,
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

    @dummy_jit(
        static_argnums=0,
        static_argnames=("left_grade", "right_grade"),
        dynamic_batch=("Ai", "Bj"),
    )
    def tensor_shuffle_product_homogeneous(
        self,
        Ai,
        Bj,
        *,
        left_grade: Bidegree | None = None,
        right_grade: Bidegree | None = None,
    ):
        self._require_full_shuffle()
        if left_grade is None or right_grade is None:
            raise TypeError(
                "bidegree homogeneous shuffles require left_grade= and right_grade=."
            )
        left_grade = _bidegree(left_grade, name="left_grade")
        right_grade = _bidegree(right_grade, name="right_grade")
        output_grade = (
            left_grade[0] + right_grade[0],
            left_grade[1] + right_grade[1],
        )
        return self._shuffle_block(
            Ai, Bj, left_grade, right_grade, output_grade
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
        input_grade: Bidegree | None = None,
        generator_part: Literal["prime", "doubleprime"] | None = None,
    ):
        if input_grade is None or generator_part is None:
            raise TypeError(
                "bidegree vector shuffles require input_grade= and "
                "generator_part='prime' or 'doubleprime'."
            )
        input_grade = _bidegree(input_grade, name="input_grade")
        if generator_part == "prime":
            generator_grade = (1, 0)
            if v.shape[-1] == sum(self.dims):
                v = v[..., : self.dims[0]]
            elif v.shape[-1] != self.dims[0]:
                raise ValueError("prime generator has the wrong width.")
        elif generator_part == "doubleprime":
            generator_grade = (0, 1)
            if v.shape[-1] == sum(self.dims):
                v = v[..., self.dims[0] :]
            elif v.shape[-1] != self.dims[1]:
                raise ValueError("double-prime generator has the wrong width.")
        else:
            raise ValueError("generator_part must be 'prime' or 'doubleprime'.")
        output_grade = (
            input_grade[0] + generator_grade[0],
            input_grade[1] + generator_grade[1],
        )
        return self._shuffle_block(
            Ai, v, input_grade, generator_grade, output_grade
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "a_first_on"),
        dynamic_batch=("A", "v"),
    )
    def tensor_shuffle_vector(
        self,
        A: BigradedTensor,
        v,
        *,
        trunc: Optional[Bidegree] = None,
        a_first_on: bool = False,
    ) -> BigradedTensor:
        self._require_shuffle()
        A = self._validate_element(A, name="A")
        if a_first_on and A.spec.include_scalar:
            raise ValueError("a_first_on=True conflicts with A including scalar.")
        v = self.xp.asarray(v)
        prime, doubleprime = self._split_generator(v)
        layout = self.resolve_layout(trunc, include_scalar=False)
        batch = self.xp.broadcast_shapes(A.batch_shape, v.shape[:-1])
        dtype = self.xp.result_type(A.blocks[0], v) if A.blocks else v.dtype
        blocks = []
        for output_grade in layout.grades:
            n, m = output_grade
            terms = []
            if n > 0 and A.spec.contains((n - 1, m)):
                terms.append(
                    self._shuffle_block(
                        A[n - 1, m],
                        prime,
                        (n - 1, m),
                        (1, 0),
                        output_grade,
                    )
                )
            if m > 0 and A.spec.contains((n, m - 1)):
                terms.append(
                    self._shuffle_block(
                        A[n, m - 1],
                        doubleprime,
                        (n, m - 1),
                        (0, 1),
                        output_grade,
                    )
                )
            if not terms:
                blocks.append(
                    self._zero_block(layout, output_grade, batch=batch, dtype=dtype)
                )
            else:
                term = terms[0]
                for addition in terms[1:]:
                    term = term + addition
                blocks.append(term)
        return BigradedTensor(tuple(blocks), layout.spec)


__all__ = ["StandardBigradedCore"]
