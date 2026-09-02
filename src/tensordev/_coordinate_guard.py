"""Boundary guards for algorithms with fixed coordinate semantics."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from tensordev._backend import get_default_core, require_total_degree_default


_Function = TypeVar("_Function", bound=Callable[..., Any])


def require_standard_total_degree_default(feature: str) -> None:
    """Reject a configured core ignored by a standard dense implementation."""
    require_total_degree_default(feature)
    core = get_default_core()
    coordinates = getattr(core, "coordinates", None)
    if coordinates != "standard":
        raise RuntimeError(
            f"{feature} requires standard coordinates and cannot "
            f"use the configured {coordinates!r} coordinate core. Reset the "
            "default core before calling it."
        )


def standard_total_degree_only(feature: str) -> Callable[[_Function], _Function]:
    """Guard a public boundary outside any nested JAX compilation wrapper."""
    def decorate(function: _Function) -> _Function:
        @wraps(function)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            require_standard_total_degree_default(feature)
            return function(*args, **kwargs)

        restriction = (
            "This specialized entry point requires the active default core "
            "to use standard total-degree coordinates. Bidegree and shear "
            "defaults are rejected before numerical work."
        )
        if guarded.__doc__:
            guarded.__doc__ = f"{guarded.__doc__.rstrip()}\n\nNotes\n-----\n{restriction}"
        else:
            guarded.__doc__ = restriction

        return guarded  # type: ignore[return-value]

    return decorate


__all__ = [
    "require_standard_total_degree_default",
    "standard_total_degree_only",
]
