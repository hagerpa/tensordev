"""Construct partially symmetrized cores from ordered JAX bidegree cores."""

from __future__ import annotations

from typing import Any, Literal

from tensordev.core._factory import _resolved_precompute_shuffle
from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.symmetrized import (
    JaxPartiallySymmetrizedShearBigraded,
)


def symmetrized_core(
    core: Any,
    *,
    precompute_shuffle: bool | Literal["generator"] | None = None,
) -> (
    JaxPartiallySymmetrizedBigraded
    | JaxPartiallySymmetrizedShearBigraded
):
    """Construct the matching partially symmetrized bidegree core.

    The source core is authoritative for dimensions, capacity, active default
    truncation, backend, and coordinates.  By default its shuffle scope is
    inherited; pass ``False``, ``"generator"``, or ``True`` to override it.
    """
    shuffle = _resolved_precompute_shuffle(core, precompute_shuffle)

    if isinstance(core, JaxBigraded):
        return JaxPartiallySymmetrizedBigraded(
            dims=core.dims,
            max_trunc=core.max_truncation,
            default_trunc=core.default_truncation,
            precompute_shuffle=shuffle,
        )
    if isinstance(core, JaxShearBigraded):
        return JaxPartiallySymmetrizedShearBigraded(
            dims=core.dims,
            max_trunc=core.max_truncation,
            default_trunc=core.default_truncation,
            precompute_shuffle=shuffle,
        )

    if getattr(core, "representation", None) == "partially_symmetrized":
        raise TypeError(
            "symmetrized_core requires an ordered source core; the supplied "
            f"{type(core).__name__} is already partially symmetrized."
        )
    if getattr(core, "grading", None) == "total_degree":
        raise TypeError(
            "symmetrized_core requires a bidegree source core; total-degree "
            "partial symmetrization is not supported."
        )
    raise TypeError(
        "symmetrized_core supports built-in ordered JaxBigraded and "
        f"JaxShearBigraded source cores, got {type(core).__name__}."
    )


__all__ = ["symmetrized_core"]
