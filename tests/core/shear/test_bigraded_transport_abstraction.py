"""Ownership contracts for representation-neutral bidegree shear transport."""

from __future__ import annotations

import tensordev.core.shear.bigraded as ordered_bigraded_module

from tensordev.core.jax import _compiled_jittables
from tensordev.core.shear.algebra import ShearCoordinateCore
from tensordev.core.shear.bigraded import (
    JaxShearBigraded,
    ShearBigradedCore,
)
from tensordev.core.shear.bigraded_transport import (
    BigradedShearCoordinateCore,
)
from tensordev.core.shear.bigraded_jax_transport import (
    _COMPILED_BIGRADED_SHEAR_ADJOINT_PRODUCT,
    _bind_bigraded_shear_jax_methods,
)
from tensordev.core.shear.jax import JaxShearTotal
from tensordev.core.utils.annotations import get_jit_kwargs, is_jittable


ELEMENT_DRIVERS = (
    "_coordinate_element",
    "_coordinate_forward",
    "_coordinate_inverse",
    "_coordinate_forward_transpose",
    "_coordinate_inverse_transpose",
    "_shuffle_schedule",
)

HOMOGENEOUS_DRIVERS = (
    "tensor_product_homogeneous",
    "tensor_adjoint_left_homogeneous",
    "tensor_adjoint_right_homogeneous",
    "tensor_matrix_product_homogeneous",
)


def test_ordered_bidegree_shear_inherits_one_transport_implementation():
    assert BigradedShearCoordinateCore.__bases__ == (ShearCoordinateCore,)
    assert ShearBigradedCore.__mro__[1] is BigradedShearCoordinateCore

    for name in ELEMENT_DRIVERS + HOMOGENEOUS_DRIVERS:
        assert name not in ShearBigradedCore.__dict__
        assert (
            getattr(ShearBigradedCore, name)
            is BigradedShearCoordinateCore.__dict__[name]
        )

    for name in (
        "tensor_product",
        "_product_output_block",
        "tensor_adjoint_product",
        "_adjoint_left_block",
        "_adjoint_right_block",
        "tensor_matrix_product",
        "_matrix_product_block",
    ):
        assert getattr(ShearBigradedCore, name) is ShearCoordinateCore.__dict__[name]


def test_representation_specific_coordinate_kernels_stay_on_ordered_core():
    for name in (
        "_coordinate_forward_block",
        "_coordinate_inverse_block",
        "_coordinate_forward_transpose_block",
        "_coordinate_inverse_transpose_block",
    ):
        assert name not in BigradedShearCoordinateCore.__dict__
        assert name in ShearBigradedCore.__dict__


def test_homogeneous_drivers_retain_jit_metadata_and_binding():
    expected_static_names = {
        "tensor_product_homogeneous": ("left_grade", "right_grade"),
        "tensor_adjoint_left_homogeneous": (
            "multiplier_grade",
            "output_grade",
        ),
        "tensor_adjoint_right_homogeneous": (
            "multiplier_grade",
            "output_grade",
        ),
        "tensor_matrix_product_homogeneous": (
            "left_grade",
            "right_grade",
            "row_axis",
            "col_axis",
        ),
    }
    compiled_names = {
        name for name, _ in _compiled_jittables(JaxShearBigraded)
    }

    for name, static_names in expected_static_names.items():
        function = getattr(BigradedShearCoordinateCore, name)
        assert is_jittable(function)
        assert get_jit_kwargs(function)["static_argnums"] == 0
        assert get_jit_kwargs(function)["static_argnames"] == static_names
        assert name in compiled_names


def test_whole_adjoint_jax_binding_is_shared_outside_ordered_core_module():
    assert not hasattr(
        ordered_bigraded_module,
        "_COMPILED_SHEAR_ADJOINT_PRODUCT",
    )
    core = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
    assert (
        core.tensor_adjoint_product.__func__
        is _COMPILED_BIGRADED_SHEAR_ADJOINT_PRODUCT
    )
    assert callable(_bind_bigraded_shear_jax_methods)


def test_total_degree_shear_keeps_its_existing_transport_shape_boundary():
    assert BigradedShearCoordinateCore not in JaxShearTotal.__mro__
    assert JaxShearTotal.__mro__[1] is ShearCoordinateCore
    for name in HOMOGENEOUS_DRIVERS:
        assert name in JaxShearTotal.__dict__
