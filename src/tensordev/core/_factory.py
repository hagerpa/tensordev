"""Shared validation for factories which derive one core from another."""

from __future__ import annotations

from typing import Any

from tensordev.core.shuffle import (
    _normalize_precompute_shuffle,
    _precompute_shuffle_argument,
)


def _resolved_precompute_shuffle(core: Any, requested: Any):
    """Resolve an explicit scope or inherit the source core's scope."""
    if requested is not None:
        return _precompute_shuffle_argument(
            _normalize_precompute_shuffle(requested, allow_generator=True)
        )
    store = getattr(core, "shuffle_plan_store", None)
    if store is None:
        return False
    return _precompute_shuffle_argument(getattr(store, "scope", "full"))


__all__ = []
