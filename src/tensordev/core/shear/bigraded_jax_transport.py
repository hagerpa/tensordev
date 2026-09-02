"""Shared JAX method binding for bidegree shear coordinate cores."""

from __future__ import annotations

import jax

from tensordev.core.bigraded.jax_bindings import _bind_bigraded_jax_methods
from tensordev.core.shear.algebra import ShearCoordinateCore


_COMPILED_BIGRADED_SHEAR_ADJOINT_PRODUCT = jax.jit(
    ShearCoordinateCore.tensor_adjoint_product,
    static_argnums=0,
    static_argnames=(
        "trunc",
        "side",
        "w_first_on",
        "y_first_on",
        "first_on_out",
    ),
)


def _bind_bigraded_shear_jax_methods(core) -> None:
    """Bind all generic JAX wrappers required by a bidegree shear core."""
    _bind_bigraded_jax_methods(
        core,
        adjoint_product=_COMPILED_BIGRADED_SHEAR_ADJOINT_PRODUCT,
    )


__all__ = ["_bind_bigraded_shear_jax_methods"]
