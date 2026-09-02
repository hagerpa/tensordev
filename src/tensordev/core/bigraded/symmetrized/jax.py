"""Concrete JAX partially symmetrized bidegree core."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from tensordev.core.bigraded.jax_backend import (
    _JaxPartiallySymmetrizedBigradedBackend,
)
from tensordev.core.bigraded.jax_bindings import _bind_bigraded_jax_methods
from tensordev.core.bigraded.symmetrized.algebra import (
    PartiallySymmetrizedBigradedCore,
)
from tensordev.core.bigraded.symmetrized.bridge import (
    SymmetrizationBridgePlanStore,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
)
from tensordev.core.bigraded.types import Bidegree


class JaxPartiallySymmetrizedBigraded(
    _JaxPartiallySymmetrizedBigradedBackend,
    PartiallySymmetrizedBigradedCore,
):
    """JAX core for a partially symmetrized bidegree representation."""

    def __init__(
        self,
        *,
        dims: Bidegree | None = None,
        max_trunc: Bidegree | None = None,
        default_trunc: Bidegree | None = None,
        plan_store: PartiallySymmetrizedPlanStore | None = None,
        bridge_plan_store: SymmetrizationBridgePlanStore | None = None,
        shear_plan_store: PartiallySymmetrizedShearPlanStore | None = None,
        shuffle_plan_store: (
            PartiallySymmetrizedShearShufflePlanStore | None
        ) = None,
        precompute_shuffle: bool | Literal["generator"] = False,
    ) -> None:
        super().__init__(
            jnp,
            dims=dims,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            plan_store=plan_store,
            bridge_plan_store=bridge_plan_store,
            shear_plan_store=shear_plan_store,
            shuffle_plan_store=shuffle_plan_store,
            precompute_shuffle=precompute_shuffle,
        )
        _bind_bigraded_jax_methods(self)


__all__ = ["JaxPartiallySymmetrizedBigraded"]
