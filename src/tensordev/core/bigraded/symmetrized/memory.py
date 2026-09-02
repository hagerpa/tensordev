"""Exact retained-memory estimates for partially symmetrized cores."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping

from tensordev.core.bigraded.symmetrized.bridge import (
    _expected_bridge_memory_bytes_by_category,
)
from tensordev.core.bigraded.symmetrized.gamma import (
    _expected_gamma_memory_bytes_by_category,
)
from tensordev.core.bigraded.symmetrized.plans import (
    _expected_plan_memory_bytes_by_category,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    _expected_transform_memory_bytes_by_category,
)
from tensordev.core.bigraded.types import Bidegree
from tensordev.core.shear.symmetrized import (
    _expected_partially_symmetrized_shear_generator_memory_bytes_by_category,
)
from tensordev.core.shuffle import _normalize_precompute_shuffle


def _expected_partially_symmetrized_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
    *,
    coordinates: Literal["standard", "shear"] = "standard",
    precompute_shuffle: bool | Literal["generator"] = False,
) -> Mapping[str, int]:
    """Return exact retained plan payload without constructing any store.

    Partially symmetrized tensors in standard coordinates require
    coordinate-conversion plans only when shuffle is enabled, because their
    shuffle is evaluated through shear coordinates.  Partially symmetrized
    tensors in shear coordinates retain coordinate-conversion plans and
    generator-action plans unconditionally.  Both coordinate systems retain
    shuffle plans exactly when the requested shuffle scope is nonempty.
    """

    if not isinstance(coordinates, str):
        raise TypeError(
            "coordinates must be a string, got "
            f"{type(coordinates).__name__}."
        )
    if coordinates not in {"standard", "shear"}:
        raise ValueError(
            "coordinates must be either 'standard' or 'shear', got "
            f"{coordinates!r}."
        )
    scope = _normalize_precompute_shuffle(
        precompute_shuffle,
        allow_generator=True,
    )

    categories = dict(
        _expected_plan_memory_bytes_by_category(dims, max_truncation)
    )
    categories.update(
        _expected_bridge_memory_bytes_by_category(dims, max_truncation)
    )

    if coordinates == "shear" or scope != "none":
        categories.update(
            _expected_transform_memory_bytes_by_category(
                dims,
                max_truncation,
            )
        )
    if scope != "none":
        categories.update(
            _expected_gamma_memory_bytes_by_category(
                dims,
                max_truncation,
                scope=scope,
            )
        )
    if coordinates == "shear":
        categories.update(
            _expected_partially_symmetrized_shear_generator_memory_bytes_by_category(
                dims,
                max_truncation,
            )
        )
    return MappingProxyType(categories)


__all__ = []
