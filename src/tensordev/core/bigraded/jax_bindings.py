"""Shared JAX bindings for standard-coordinate bidegree representations."""

from __future__ import annotations

import types

import jax

from tensordev.core.jax import _compiled_jittables
from tensordev.core.universal import Universal


_COMPILED_BIGRADED_ADJOINT_PRODUCT = jax.jit(
    Universal.tensor_adjoint_product,
    static_argnums=0,
    static_argnames=(
        "trunc",
        "side",
        "w_first_on",
        "y_first_on",
        "first_on_out",
    ),
)


def _bind_bigraded_jax_methods(
    core,
    *,
    adjoint_product=_COMPILED_BIGRADED_ADJOINT_PRODUCT,
) -> None:
    """Bind decorated methods and the explicit whole-adjoint boundary."""
    for name, function in _compiled_jittables(type(core)):
        setattr(core, name, types.MethodType(function, core))
    setattr(
        core,
        "tensor_adjoint_product",
        types.MethodType(adjoint_product, core),
    )


__all__ = []
