"""Coordinate-law orchestration shared by total and bidegree shear cores."""

from __future__ import annotations

from typing import Literal

from tensordev.core.utils.annotations import jit as dummy_jit


class ShearCoordinateCore:
    """Mixin implementing coordinate-transported shear operations.

    Concrete subclasses provide the four coordinate actions
    and their homogeneous block variants.  The ordinary algebra drivers live
    on :class:`~tensordev.core.universal.Universal` and its block
    specializations, so neither shear core carries a second implementation of
    graded products, contractions, matrix products, or formal series.
    """

    coordinates = "shear"

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "a_first_on", "b_first_on"),
        dynamic_batch=("A", "B"),
    )
    def tensor_product(
            self,
            A,
            B,
            trunc=None,
            *,
            a_first_on: bool = False,
            b_first_on: bool = False,
    ):
        """Compute the truncated tensor product in shear coordinates."""
        active = self.normalize_truncation(trunc)
        left = self._coordinate_inverse(
            A, trunc=active, first_on=a_first_on
        )
        right = self._coordinate_inverse(
            B, trunc=active, first_on=b_first_on
        )
        first_on_out = a_first_on or b_first_on
        standard = self._standard_tensor_product(
            left,
            right,
            active,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )
        return self._coordinate_forward(
            standard, trunc=active, first_on=first_on_out
        )

    def _product_output_block(self, contributions, output_grade):
        """Coordinate-native fused product hook used by Volterra."""
        standard_contributions = tuple(
            (
                self._coordinate_inverse_block(left, left_grade),
                self._coordinate_inverse_block(right, right_grade),
                left_grade,
                right_grade,
            )
            for left, right, left_grade, right_grade in contributions
        )
        standard = self._standard_product_output_block(
            standard_contributions, output_grade
        )
        return self._coordinate_forward_block(standard, output_grade)

    def _adjoint_left_block(
            self,
            multiplier,
            target,
            multiplier_grade,
            target_grade,
            output_grade,
    ):
        standard = self._standard_adjoint_left_block(
            self._coordinate_inverse_block(multiplier, multiplier_grade),
            self._coordinate_forward_transpose_block(target, target_grade),
            multiplier_grade,
            target_grade,
            output_grade,
        )
        return self._coordinate_inverse_transpose_block(standard, output_grade)

    def _adjoint_right_block(
            self,
            multiplier,
            target,
            multiplier_grade,
            target_grade,
            output_grade,
    ):
        standard = self._standard_adjoint_right_block(
            self._coordinate_inverse_block(multiplier, multiplier_grade),
            self._coordinate_forward_transpose_block(target, target_grade),
            multiplier_grade,
            target_grade,
            output_grade,
        )
        return self._coordinate_inverse_transpose_block(standard, output_grade)

    def tensor_adjoint_product(
            self,
            W,
            Y,
            trunc=None,
            side: Literal["left", "right"] = "left",
            *,
            w_first_on: bool = False,
            y_first_on: bool = False,
            first_on_out: bool = False,
    ):
        """Apply the explicit transpose-transport adjoint identity."""
        active = self.normalize_truncation(trunc)
        multiplier_truncation = self._coordinate_natural_truncation(
            W, first_on=w_first_on
        )
        target_truncation = self._coordinate_natural_truncation(
            Y, first_on=y_first_on
        )
        multiplier = self._coordinate_inverse(
            W, trunc=multiplier_truncation, first_on=w_first_on
        )
        target = self._coordinate_forward_transpose(
            Y, trunc=target_truncation, first_on=y_first_on
        )
        standard = self._standard_tensor_adjoint_product(
            multiplier,
            target,
            active,
            side=side,
            w_first_on=w_first_on,
            y_first_on=y_first_on,
            first_on_out=first_on_out,
        )
        return self._coordinate_inverse_transpose(
            standard, trunc=active, first_on=first_on_out
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "output_zero_level"),
        dynamic_batch=("g", "X"),
    )
    def tensor_fmexp(
            self,
            g,
            X,
            *,
            trunc=None,
            output_zero_level: bool = True,
    ):
        """Compute a shear formal exponential with one transport each way."""
        active = self.normalize_truncation(trunc)
        g = self._series_validate_left_factor(g)
        X = self._series_validate_exponential_argument(X)
        kind, argument = self._series_argument(X)
        if self._series_order(active) == 0 or kind == "empty":
            result = self._series_truncate_left_factor(g, active)
        elif kind == "first":
            result = self._fmexp_first_level(g, argument, trunc=active)
        else:
            standard_g = self._coordinate_inverse(
                g, trunc=active, first_on=False
            )
            standard_X = self._coordinate_inverse(
                argument, trunc=active, first_on=True
            )
            standard = self._standard_tensor_fmexp(
                standard_g,
                standard_X,
                trunc=active,
                output_zero_level=True,
            )
            result = self._coordinate_forward(
                standard, trunc=active, first_on=False
            )
        return self._series_output(
            result,
            trunc=active,
            output_zero_level=output_zero_level,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "output_zero_level"),
        dynamic_batch=("X",),
    )
    def tensor_logarithm(
            self,
            X,
            *,
            trunc=None,
            output_zero_level: bool = True,
    ):
        """Compute the shear logarithm through one inverse/forward transport."""
        active = self.normalize_truncation(trunc)
        X = self._series_validate_log_argument(X)
        kind, _ = self._series_argument(X)
        if kind == "empty":
            return self._series_logarithm_zero(
                trunc=active,
                output_zero_level=output_zero_level,
            )
        standard_X = self._coordinate_inverse(
            X, trunc=active, first_on=True
        )
        standard = self._standard_tensor_logarithm(
            standard_X,
            trunc=active,
            output_zero_level=True,
        )
        result = self._coordinate_forward(
            standard, trunc=active, first_on=False
        )
        return self._series_output(
            result,
            trunc=active,
            output_zero_level=output_zero_level,
        )

    def _matrix_product_block(
            self,
            left,
            right,
            left_grade,
            right_grade,
            output_grade,
            *,
            row_axis: int,
            col_axis: int,
    ):
        standard = self._standard_matrix_product_block(
            self._coordinate_inverse_block(left, left_grade),
            self._coordinate_inverse_block(right, right_grade),
            left_grade,
            right_grade,
            output_grade,
            row_axis=row_axis,
            col_axis=col_axis,
        )
        return self._coordinate_forward_block(standard, output_grade)

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "row_axis", "col_axis"),
        dynamic_batch=("A", "B"),
    )
    def tensor_matrix_product(
            self,
            A,
            B,
            trunc=None,
            row_axis: int = -3,
            col_axis: int = -2,
    ):
        """Transport a complete matrix-valued graded product once."""
        active = self.normalize_truncation(trunc)
        left = self._coordinate_inverse(A, trunc=active, first_on=False)
        right = self._coordinate_inverse(B, trunc=active, first_on=False)
        standard = self._standard_tensor_matrix_product(
            left,
            right,
            active,
            row_axis=row_axis,
            col_axis=col_axis,
        )
        return self._coordinate_forward(
            standard, trunc=active, first_on=False
        )


__all__ = ["ShearCoordinateCore"]
