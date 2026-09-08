"""JAX concrete core for ordered rectangular bidegree truncation."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from tensordev.core.capabilities import _WORDWISE_SIGNATURE_PROTOCOL
from tensordev.core.bigraded.jax_backend import _JaxBigradedBackend
from tensordev.core.bigraded.jax_bindings import _bind_bigraded_jax_methods
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.bigraded.shuffle import BigradedShufflePlanStore
from tensordev.core.bigraded.standard import StandardBigradedCore
from tensordev.core.bigraded.types import Bidegree


class JaxBigraded(_JaxBigradedBackend, StandardBigradedCore):
    """Fixed-capacity JAX core for standard ordered bidegree coordinates."""

    _wordwise_signature_protocol = _WORDWISE_SIGNATURE_PROTOCOL

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
__all__ = ["JaxBigraded"]
