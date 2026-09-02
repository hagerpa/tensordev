"""JAX concrete core for ordered rectangular bidegree truncation."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from tensordev.core.bigraded.jax_backend import _JaxBigradedBackend
from tensordev.core.bigraded.jax_bindings import _bind_bigraded_jax_methods
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.bigraded.shuffle import BigradedShufflePlanStore
from tensordev.core.bigraded.standard import StandardBigradedCore
from tensordev.core.bigraded.types import Bidegree


class JaxBigraded(_JaxBigradedBackend, StandardBigradedCore):
    """Fixed-capacity JAX core for standard ordered bidegree coordinates."""

    def __init__(
        self,
        *,
        dims: Bidegree | None = None,
        max_trunc: Bidegree | None = None,
        default_trunc: Bidegree | None = None,
        plan_store: BigradedPlanStore | None = None,
        shuffle_plan_store: BigradedShufflePlanStore | None = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        super().__init__(
            jnp,
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            plan_store=plan_store,
            shuffle_plan_store=shuffle_plan_store,
            precompute_shuffle=precompute_shuffle,
        )
        _bind_bigraded_jax_methods(self)


def bigraded_core(
    *,
    dims: Bidegree,
    max_trunc: Bidegree,
    default_trunc: Bidegree | None = None,
    precompute_shuffle: bool | Literal["generator"] = False,
    representation: Literal[
        "ordered", "partially_symmetrized"
    ] = "ordered",
):
    """Construct a bounded standard-coordinate JAX bidegree core.

    ``representation="ordered"`` retains ordinary word placements;
    ``"partially_symmetrized"`` selects compact partially symmetrized
    bidegree blocks.  Shuffle plans
    are an explicit capability because their bounded precomputation can be
    much larger than the concatenation plans.  Pass ``True`` for arbitrary
    shuffles, or ``"generator"`` for the lightweight generator actions used
    by Volterra algorithms.
    """

    if not isinstance(representation, str):
        raise TypeError(
            "representation must be a string, got "
            f"{type(representation).__name__}."
        )
    if representation == "ordered":
        return JaxBigraded(
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            precompute_shuffle=precompute_shuffle,
        )
    if representation == "partially_symmetrized":
        from tensordev.core.bigraded.symmetrized.jax import (
            JaxPartiallySymmetrizedBigraded,
        )

        return JaxPartiallySymmetrizedBigraded(
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            precompute_shuffle=precompute_shuffle,
        )
    raise ValueError(
        "representation must be either 'ordered' or "
        f"'partially_symmetrized', got {representation!r}."
    )


__all__ = ["JaxBigraded", "bigraded_core"]
