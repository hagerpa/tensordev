"""Safe construction of shear cores from standard-coordinate JAX cores."""

from __future__ import annotations

from typing import Any, Literal

from tensordev.core._factory import _resolved_precompute_shuffle
from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.types import _bidegree
from tensordev.core.jax import Jax
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal
from tensordev.core.shear.total import _dims, _non_negative_int
from tensordev.core.shuffle import _normalize_precompute_shuffle


def shear_core(
    core: Any,
    *,
    dims: tuple[int, int] | None = None,
    max_trunc: int | tuple[int, int] | None = None,
    precompute_shuffle: bool | Literal["generator"] | None = None,
) -> Any:
    """Construct the matching shear core from a standard JAX core.

    A standard total-degree core does not record the alphabet split, so
    ``dims=(d_prime, d_doubleprime)`` is required.  Its finite capacity is
    inherited when available; converting an unbounded ``Jax()`` additionally
    requires ``max_trunc``.  A bidegree core already records both pieces and
    shares every compatible representation-level store with the returned
    shear core.  This applies to ordered and partially symmetrized sources.

    Unless explicitly overridden, the ordinary core's shuffle-precomputation
    scope is carried over.  The target core constructs or reuses the plans
    required by its shear coordinates.
    """
    if getattr(core, "coordinates", None) != "standard":
        raise TypeError(
            "shear_core requires a standard-coordinate JAX core, got "
            f"{type(core).__name__} with "
            f"coordinates={getattr(core, 'coordinates', None)!r}."
        )

    shuffle = _resolved_precompute_shuffle(core, precompute_shuffle)

    if isinstance(core, JaxPartiallySymmetrizedBigraded):
        from tensordev.core.shear.symmetrized import (
            JaxPartiallySymmetrizedShearBigraded,
        )

        _validate_bidegree_source_assertions(
            core,
            dims=dims,
            max_trunc=max_trunc,
        )
        requested_scope = _normalize_precompute_shuffle(
            shuffle,
            allow_generator=True,
        )
        shuffle_plan_store = core.shuffle_plan_store
        if (
            shuffle_plan_store is None
            or shuffle_plan_store.scope != requested_scope
        ):
            shuffle_plan_store = None
        return JaxPartiallySymmetrizedShearBigraded(
            plan_store=core.plan_store,
            bridge_plan_store=core.bridge_plan_store,
            shear_plan_store=core.shear_plan_store,
            shuffle_plan_store=shuffle_plan_store,
            default_trunc=core.default_truncation,
            precompute_shuffle=(
                False if shuffle_plan_store is not None else shuffle
            ),
        )

    if isinstance(core, JaxBigraded):
        _validate_bidegree_source_assertions(
            core,
            dims=dims,
            max_trunc=max_trunc,
        )
        return JaxShearBigraded(
            plan_store=core.plan_store,
            default_trunc=core.default_truncation,
            precompute_shuffle=shuffle,
        )

    if isinstance(core, Jax):
        if dims is None:
            raise TypeError(
                "dims=(d_prime, d_doubleprime) is required because a "
                "total-degree core does not determine the alphabet split."
            )
        normalized_dims = _dims(dims)
        if core.d is not None and sum(normalized_dims) != core.d:
            raise ValueError(
                f"split dimensions {normalized_dims} total "
                f"{sum(normalized_dims)}, but the source core has d={core.d}."
            )

        if core.max_truncation is None:
            if max_trunc is None:
                raise TypeError(
                    "max_trunc is required when constructing a shear core "
                    "from an unbounded total-degree core."
                )
            normalized_max = _non_negative_int(max_trunc, name="max_trunc")
            default_trunc = None
        else:
            normalized_max = core.max_truncation
            default_trunc = core.default_truncation
            if max_trunc is not None:
                asserted_max = _non_negative_int(max_trunc, name="max_trunc")
                if asserted_max != normalized_max:
                    raise ValueError(
                        "max_trunc disagrees with the source total-degree "
                        f"core: {asserted_max} != {normalized_max}."
                    )

        return JaxShearTotal(
            dims=normalized_dims,
            max_trunc=normalized_max,
            default_trunc=default_trunc,
            precompute_shuffle=shuffle,
        )

    raise TypeError(
        "shear_core supports built-in Jax, JaxBigraded, and partially "
        "symmetrized JAX bidegree source cores, got "
        f"{type(core).__name__}."
    )


def _validate_bidegree_source_assertions(
    core: Any,
    *,
    dims: tuple[int, int] | None,
    max_trunc: int | tuple[int, int] | None,
) -> None:
    """Validate optional assertions shared by both bidegree representations."""
    if dims is not None:
        normalized_dims = _bidegree(dims, name="dims")
        if normalized_dims != core.dims:
            raise ValueError(
                "dims disagree with the source bidegree core: "
                f"{normalized_dims} != {core.dims}."
            )
    if max_trunc is not None:
        normalized_max = _bidegree(max_trunc, name="max_trunc")
        if normalized_max != core.max_truncation:
            raise ValueError(
                "max_trunc disagrees with the source bidegree core: "
                f"{normalized_max} != {core.max_truncation}."
            )


__all__ = ["shear_core"]
