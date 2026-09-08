"""Coordinate transport shared by bidegree shear cores."""

from __future__ import annotations

from tensordev.core.bigraded.types import BigradedTensor, _bidegree
from tensordev.core.shear.algebra import ShearCoordinateCore
from tensordev.core.utils.annotations import jit as dummy_jit


class BigradedShearCoordinateCore(ShearCoordinateCore):
    """Bidegree-shaped drivers shared by shear cores.

    Concrete cores provide the four coordinate block actions and the standard
    block-algebra hooks.  This layer only transports those hooks and assembles
    ``BigradedTensor`` elements without changing partial symmetrization.
    """

    def _coordinate_element(
        self,
        element,
        *,
        trunc,
        first_on,
        source_coordinates,
        target_coordinates,
        transform,
    ):
        element = self._validate_element_coordinates(
            element, name="element", coordinates=source_coordinates
        )
        if first_on == element.spec.include_scalar:
            raise ValueError("first_on disagrees with the element's scalar policy.")
        requested = (
            element.truncation
            if trunc is None
            else self._normalize_truncation(trunc)
        )
        active = (
            min(requested[0], element.truncation[0]),
            min(requested[1], element.truncation[1]),
        )
        layout = self._resolve_layout_coordinates(
            active,
            include_scalar=element.spec.include_scalar,
            coordinates=target_coordinates,
        )
        return BigradedTensor(
            tuple(transform(element[grade], grade) for grade in layout.grades),
            layout.spec,
        )

    def _coordinate_forward(self, element, *, trunc=None, first_on=False):
        return self._coordinate_element(
            element,
            trunc=trunc,
            first_on=first_on,
            source_coordinates="standard",
            target_coordinates="shear",
            transform=self._coordinate_forward_block,
        )

    def _coordinate_inverse(self, element, *, trunc=None, first_on=False):
        return self._coordinate_element(
            element,
            trunc=trunc,
            first_on=first_on,
            source_coordinates="shear",
            target_coordinates="standard",
            transform=self._coordinate_inverse_block,
        )

    def _coordinate_forward_transpose(
        self, element, *, trunc=None, first_on=False
    ):
        return self._coordinate_element(
            element,
            trunc=trunc,
            first_on=first_on,
            source_coordinates="shear",
            target_coordinates="standard",
            transform=self._coordinate_forward_transpose_block,
        )

    def _coordinate_inverse_transpose(
        self, element, *, trunc=None, first_on=False
    ):
        return self._coordinate_element(
            element,
            trunc=trunc,
            first_on=first_on,
            source_coordinates="standard",
            target_coordinates="shear",
            transform=self._coordinate_inverse_transpose_block,
        )

    def _shuffle_schedule(
        self,
        A,
        B,
        trunc,
        *,
        a_first_on,
        b_first_on,
        first_on_out,
    ):
        """Resolve native-coordinate bidegree blocks for a full shuffle."""
        self._require_full_shuffle()
        A = self._validate_element(A, name="A")
        B = self._validate_element(B, name="B")
        if a_first_on and A.spec.include_scalar:
            raise ValueError("a_first_on conflicts with A's scalar block.")
        if b_first_on and B.spec.include_scalar:
            raise ValueError("b_first_on conflicts with B's scalar block.")
        include_scalar = (
            not first_on_out
            and A.spec.include_scalar
            and B.spec.include_scalar
        )
        active = self._effective_product_truncation(A, B, trunc)
        layout = self.resolve_layout(active, include_scalar=include_scalar)
        batch = self._result_batch(A, B)
        dtype = self._result_dtype(A, B)

        from tensordev.core.grading import GradedConvolutionSchedule

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
        )

    def _shuffle_block(
        self,
        left,
        right,
        left_grade,
        right_grade,
        output_grade,
    ):
        """Apply the core's native shear shuffle."""
        return self._gamma_shuffle_block(
            left,
            right,
            left_grade,
            right_grade,
            output_grade,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("left_grade", "right_grade"),
        dynamic_batch=("Ai", "Bj"),
    )
    def tensor_product_homogeneous(
        self, Ai, Bj, *, left_grade=None, right_grade=None
    ):
        if left_grade is None or right_grade is None:
            raise TypeError("shear homogeneous products require explicit bidegrees.")
        left_grade = _bidegree(left_grade, name="left_grade")
        right_grade = _bidegree(right_grade, name="right_grade")
        output_grade = (
            left_grade[0] + right_grade[0],
            left_grade[1] + right_grade[1],
        )
        return self._product_output_block(
            ((Ai, Bj, left_grade, right_grade),), output_grade
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("multiplier_grade", "output_grade"),
        dynamic_batch=("Ai", "Yni"),
    )
    def tensor_adjoint_left_homogeneous(
        self, Ai, Yni, *, multiplier_grade=None, output_grade=None
    ):
        if multiplier_grade is None or output_grade is None:
            raise TypeError("shear homogeneous adjoints require explicit bidegrees.")
        multiplier_grade = _bidegree(multiplier_grade, name="multiplier_grade")
        output_grade = _bidegree(output_grade, name="output_grade")
        target_grade = (
            multiplier_grade[0] + output_grade[0],
            multiplier_grade[1] + output_grade[1],
        )
        return self._adjoint_left_block(
            Ai, Yni, multiplier_grade, target_grade, output_grade
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("multiplier_grade", "output_grade"),
        dynamic_batch=("Bj", "Ynj"),
    )
    def tensor_adjoint_right_homogeneous(
        self, Bj, Ynj, *, multiplier_grade=None, output_grade=None
    ):
        if multiplier_grade is None or output_grade is None:
            raise TypeError("shear homogeneous adjoints require explicit bidegrees.")
        multiplier_grade = _bidegree(multiplier_grade, name="multiplier_grade")
        output_grade = _bidegree(output_grade, name="output_grade")
        target_grade = (
            output_grade[0] + multiplier_grade[0],
            output_grade[1] + multiplier_grade[1],
        )
        return self._adjoint_right_block(
            Bj, Ynj, multiplier_grade, target_grade, output_grade
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
        row_axis=-3,
        col_axis=-2,
        *,
        left_grade=None,
        right_grade=None,
    ):
        if left_grade is None or right_grade is None:
            raise TypeError(
                "shear homogeneous matrix products require explicit bidegrees."
            )
        left_grade = _bidegree(left_grade, name="left_grade")
        right_grade = _bidegree(right_grade, name="right_grade")
        output_grade = (
            left_grade[0] + right_grade[0],
            left_grade[1] + right_grade[1],
        )
        return self._matrix_product_block(
            A,
            B,
            left_grade,
            right_grade,
            output_grade,
            row_axis=row_axis,
            col_axis=col_axis,
        )


__all__ = ["BigradedShearCoordinateCore"]
