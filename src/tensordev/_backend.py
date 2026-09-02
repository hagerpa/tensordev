from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

_INITIAL_BACKEND = os.environ.get("TENSORDEV_BACKEND", "jax")

_core_cache: dict[str, Any] = {}
_seq_core_cache: dict[str, Any] = {}
_default_pair: tuple[Any, Any] | None = None
_default_core_callbacks: list[Callable[[Any, Any], None]] = []
_MISSING = object()


def _build_jax() -> tuple[Any, Any]:
    from tensordev.core.jax import Jax, JaxSequentialCore
    return Jax(), JaxSequentialCore()


_REGISTRY: dict[str, Any] = {
    "jax": _build_jax,
}


def _get(backend: str) -> tuple[Any, Any]:
    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Available: {list(_REGISTRY)}. "
            f"Set via TENSORDEV_BACKEND environment variable or pass core/seq_core explicitly."
        )
    if backend not in _core_cache:
        core, seq_core = _REGISTRY[backend]()
        _core_cache[backend] = core
        _seq_core_cache[backend] = seq_core
    return _core_cache[backend], _seq_core_cache[backend]


def get_default_core_pair() -> tuple[Any, Any]:
    """Return the active algebra and sequential cores as one coherent pair.

    The environment-selected backend is constructed lazily on first access.
    Reading the pair through this function avoids observing a core and a
    sequential core from two different default configurations.
    """
    global _default_pair
    if _default_pair is None:
        _default_pair = _get(_INITIAL_BACKEND)
    return _default_pair


def get_default_core() -> Any:
    return get_default_core_pair()[0]


def get_default_seq_core() -> Any:
    return get_default_core_pair()[1]


def require_total_degree_default(feature: str) -> None:
    """Reject total-degree-only entry points under another configured grading.

    Specialized Volterra, state-space, and kernel solvers that require total
    degree call this at their public boundary so a bidegree default cannot be
    silently ignored.
    """
    core = get_default_core()
    if getattr(core, "grading", None) != "total_degree":
        raise RuntimeError(
            f"{feature} is total-degree-specific and cannot use the "
            f"configured {getattr(core, 'grading', type(core).__name__)} core. "
            "Reset the default core before calling it."
        )


def _resolve_seq_core(core: Any, seq_core: Any | None) -> Any:
    if seq_core is not None:
        return seq_core

    for name in ("make_sequential_core", "default_sequential_core", "sequential_core"):
        provider = getattr(core, name, _MISSING)
        if provider is not _MISSING:
            supplied = provider() if callable(provider) else provider
            if supplied is None:
                raise TypeError(f"{type(core).__name__}.{name} returned None.")
            return supplied

    # All JAX algebra cores can share the built-in JAX sequential singleton.
    # The namespace identity check also covers composed JAX cores which
    # deliberately do not inherit from Jax (for example a bounded bigraded
    # core).  Missing ``xp`` attributes must not compare equal accidentally.
    from tensordev.core.jax import Jax
    jax_core, jax_seq_core = _get("jax")
    core_xp = getattr(core, "xp", _MISSING)
    jax_xp = getattr(jax_core, "xp", _MISSING)
    if isinstance(core, Jax) or (core_xp is not _MISSING and core_xp is jax_xp):
        return jax_seq_core

    raise TypeError(
        "seq_core must be provided when the algebra core does not expose "
        "make_sequential_core() or default_sequential_core."
    )


def _grading_from_dims(dims: Any) -> str:
    """Infer the core grading family from the public ``dims`` convention."""
    if isinstance(dims, bool):
        raise TypeError(
            "dims must be a positive integer or a bidegree tuple, "
            f"got {dims!r}."
        )
    if isinstance(dims, Integral):
        return "total_degree"
    if isinstance(dims, tuple):
        return "bidegree"
    raise TypeError(
        "dims must be a positive integer or a bidegree tuple, "
        f"got {dims!r}."
    )


@dataclass(frozen=True, slots=True)
class _CoreConfiguration:
    """Normalized constructor selected by the complete public configuration."""

    grading: Literal["total_degree", "bidegree"]
    representation: Literal["ordered", "partially_symmetrized"]
    coordinates: Literal["standard", "shear"]
    dims: int | tuple[int, int]
    max_trunc: int | tuple[int, int]
    precompute_shuffle: bool | Literal["generator"]

    @property
    def family(self) -> str:
        """Compact core-family label derived from independent axes."""
        prefix = (
            "standard" if self.coordinates == "standard" else "shear"
        )
        suffix = "total" if self.grading == "total_degree" else "bidegree"
        if self.representation == "partially_symmetrized":
            prefix = f"{prefix}_partially_symmetrized"
        return f"{prefix}_{suffix}"

    @property
    def shuffle_scope(self) -> Literal["none", "generator", "full"]:
        if self.precompute_shuffle is False:
            return "none"
        if self.precompute_shuffle == "generator":
            return "generator"
        return "full"

    def construct(self, *, default_trunc: Any = None) -> Any:
        """Construct the selected JAX core without changing global state."""
        kwargs = {
            "max_trunc": self.max_trunc,
            "default_trunc": default_trunc,
            "precompute_shuffle": self.precompute_shuffle,
        }
        if self.grading == "total_degree" and self.coordinates == "standard":
            from tensordev.core.jax import total_degree_core

            return total_degree_core(d=int(self.dims), **kwargs)
        if self.grading == "bidegree" and self.coordinates == "standard":
            from tensordev.core.bigraded import bigraded_core

            return bigraded_core(
                dims=self.dims,
                representation=self.representation,
                **kwargs,
            )
        if self.grading == "total_degree":
            from tensordev.core.shear import JaxShearTotal

            return JaxShearTotal(dims=self.dims, **kwargs)
        if self.representation == "ordered":
            from tensordev.core.shear import JaxShearBigraded

            return JaxShearBigraded(dims=self.dims, **kwargs)

        from tensordev.core.shear.symmetrized import (
            JaxPartiallySymmetrizedShearBigraded,
        )

        return JaxPartiallySymmetrizedShearBigraded(
            dims=self.dims,
            **kwargs,
        )

    def expected_memory_bytes_by_category(self) -> Mapping[str, int]:
        """Return exact retained plan payload without constructing a core."""
        if self.grading == "total_degree" and self.coordinates == "standard":
            if self.shuffle_scope != "full":
                return {}
            from tensordev.core.jax import JaxTotalDegreeShufflePlanStore

            return JaxTotalDegreeShufflePlanStore._expected_memory_bytes_by_category(
                self.dims,
                self.max_trunc,
            )
        if (
            self.grading == "bidegree"
            and self.representation == "partially_symmetrized"
        ):
            from tensordev.core.bigraded.symmetrized.memory import (
                _expected_partially_symmetrized_memory_bytes_by_category,
            )

            return _expected_partially_symmetrized_memory_bytes_by_category(
                self.dims,
                self.max_trunc,
                coordinates=self.coordinates,
                precompute_shuffle=self.precompute_shuffle,
            )
        if self.grading == "bidegree" and self.coordinates == "standard":
            from tensordev.core.bigraded.precompute import (
                _expected_plan_memory_bytes_by_category,
            )
            from tensordev.core.bigraded.shuffle import (
                _expected_shuffle_memory_bytes_by_category,
            )

            categories = dict(
                _expected_plan_memory_bytes_by_category(
                    self.dims,
                    self.max_trunc,
                )
            )
            if self.shuffle_scope != "none":
                shuffle = _expected_shuffle_memory_bytes_by_category(
                    self.dims,
                    self.max_trunc,
                    scope=self.shuffle_scope,
                )
                categories.update(
                    {
                        "shuffle_rank_lists": shuffle["rank_lists"],
                        "shuffle_dense_permutations": shuffle[
                            "dense_permutations"
                        ],
                    }
                )
            return categories
        if self.grading == "total_degree":
            from tensordev.core.shear.total import (
                _expected_total_shear_memory_bytes_by_category,
            )

            return _expected_total_shear_memory_bytes_by_category(
                self.dims,
                self.max_trunc,
                precompute_shuffle=self.precompute_shuffle,
            )

        from tensordev.core.shear.bigraded import (
            _expected_shear_bigraded_memory_bytes_by_category,
        )

        return _expected_shear_bigraded_memory_bytes_by_category(
            self.dims,
            self.max_trunc,
            precompute_shuffle=self.precompute_shuffle,
        )


def _resolve_core_configuration(
    *,
    dims: Any,
    max_trunc: Any,
    representation: Any = "ordered",
    coordinates: Any = "standard",
    precompute_shuffle: Any = False,
) -> _CoreConfiguration:
    """Resolve one supported core family from all dispatch-relevant inputs.

    This is the single inference path used by both :func:`set_default_core`
    and the public memory estimator.  In particular, a pair-valued ``dims``
    denotes either bidegree standard coordinates or total/bidegree shear
    coordinates depending on ``max_trunc`` and ``coordinates``.  The
    representation axis is independent but partially symmetrized cores are
    deliberately restricted to bidegree grading.
    """
    if not isinstance(representation, str):
        raise TypeError(
            "representation must be a string, got "
            f"{type(representation).__name__}."
        )
    if representation not in {"ordered", "partially_symmetrized"}:
        raise ValueError(
            "representation must be either 'ordered' or "
            f"'partially_symmetrized', got {representation!r}."
        )
    if not isinstance(coordinates, str):
        raise TypeError(
            "coordinates must be a string, got "
            f"{type(coordinates).__name__}."
        )
    if coordinates not in {"standard", "shear"}:
        raise ValueError(
            "coordinates must be either 'standard' or 'shear', "
            f"got {coordinates!r}."
        )

    dims_grading = _grading_from_dims(dims)
    from tensordev.core.bigraded.types import BigradedSpec, _bidegree
    from tensordev.core.shuffle import (
        _non_negative_int,
        _normalize_precompute_shuffle,
    )

    if dims_grading == "total_degree":
        normalized_dims: int | tuple[int, int] = _non_negative_int(
            dims, name="dims"
        )
        if normalized_dims == 0:
            raise ValueError("dims must be strictly positive, got 0.")
    else:
        normalized_dims = _bidegree(dims, name="dims")
        # Reuse the public metadata invariant for positive alphabet splits.
        BigradedSpec(*normalized_dims, (0, 0))

    if isinstance(max_trunc, bool):
        raise TypeError(
            "max_trunc must be a non-negative integer or a bidegree pair, "
            f"got {max_trunc!r}."
        )
    if isinstance(max_trunc, Integral):
        max_grading = "total_degree"
        normalized_max_trunc: int | tuple[int, int] = _non_negative_int(
            max_trunc, name="max_trunc"
        )
    elif isinstance(max_trunc, (tuple, list)):
        max_grading = "bidegree"
        normalized_max_trunc = _bidegree(max_trunc, name="max_trunc")
    else:
        raise TypeError(
            "max_trunc must be a non-negative integer or a bidegree pair, "
            f"got {max_trunc!r}."
        )

    grading: Literal["total_degree", "bidegree"] | None = None
    if coordinates == "standard":
        if dims_grading == max_grading == "total_degree":
            grading = "total_degree"
        elif dims_grading == max_grading == "bidegree":
            grading = "bidegree"
    elif dims_grading == "bidegree":
        grading = max_grading

    if grading is None:
        raise ValueError(
            "Unsupported core configuration. Accepted combinations are "
            "dims=int, max_trunc=int, coordinates='standard'; "
            "dims=pair, max_trunc=pair, coordinates='standard'; "
            "dims=pair, max_trunc=int, coordinates='shear'; or "
            "dims=pair, max_trunc=pair, coordinates='shear'."
        )

    if representation == "partially_symmetrized" and grading != "bidegree":
        raise ValueError(
            "representation='partially_symmetrized' requires bidegree "
            "truncation: dims and max_trunc must both be pairs."
        )

    shuffle_scope = _normalize_precompute_shuffle(
        precompute_shuffle,
        allow_generator=not (
            grading == "total_degree" and coordinates == "standard"
        ),
    )
    normalized_shuffle: bool | Literal["generator"] = {
        "none": False,
        "generator": "generator",
        "full": True,
    }[shuffle_scope]
    return _CoreConfiguration(
        grading=grading,
        representation=representation,
        dims=normalized_dims,
        max_trunc=normalized_max_trunc,
        coordinates=coordinates,
        precompute_shuffle=normalized_shuffle,
    )


def _notify_default_core_callbacks(core: Any, seq_core: Any) -> None:
    # Iterate over a snapshot so callbacks may unregister themselves safely.
    for callback in tuple(_default_core_callbacks):
        callback(core, seq_core)


def _replace_default_core_pair(core: Any, seq_core: Any) -> None:
    """Install and announce a pair, rolling back if an internal hook fails."""
    global _default_pair
    previous = get_default_core_pair()
    _default_pair = (core, seq_core)
    try:
        _notify_default_core_callbacks(core, seq_core)
    except Exception:
        _default_pair = previous
        # Restore callbacks which may already have rebound external state.  A
        # rollback failure must not hide the exception from the attempted set.
        for callback in tuple(_default_core_callbacks):
            try:
                callback(*previous)
            except Exception:
                pass
        raise


def register_default_core_callback(
        callback: Callable[[Any, Any], None],
        *,
        notify: bool = False,
) -> Callable[[], None]:
    """Register an internal hook invoked after the default pair changes.

    This is used by :mod:`tensordev` to rebind its direct module-level method
    exports without adding proxy overhead to every tensor operation.  The
    returned function unregisters the callback.
    """
    if not callable(callback):
        raise TypeError("callback must be callable.")
    if callback not in _default_core_callbacks:
        _default_core_callbacks.append(callback)
    if notify:
        callback(*get_default_core_pair())

    def unregister() -> None:
        try:
            _default_core_callbacks.remove(callback)
        except ValueError:
            pass

    return unregister


def set_default_core(
        core: Any | None = None,
        seq_core: Any | None = None,
        *,
        dims: int | tuple[int, int] | None = None,
        max_trunc: int | tuple[int, int] | None = None,
        default_trunc: int | tuple[int, int] | None = None,
        representation: Literal[
            "ordered", "partially_symmetrized"
        ] = "ordered",
        coordinates: Literal["standard", "shear"] = "standard",
        precompute_shuffle: bool | Literal["generator"] = False,
) -> Any:
    """Set or construct the process-wide default algebra core.

    Pass an already constructed ``core`` to install it directly.  To construct
    and install a core in one call, omit ``core`` and provide ``dims`` and
    ``max_trunc``.  The supported configurations are:

    - integer ``dims`` and integer ``max_trunc``: standard total degree;
    - pair ``dims`` and pair ``max_trunc``: bidegree, with ordered or
      partially symmetrized representation;
    - pair ``dims`` and integer ``max_trunc`` with ``coordinates="shear"``:
      dense total-degree shear;
    - pair ``dims`` and pair ``max_trunc`` with ``coordinates="shear"``:
      placement-factored bidegree shear.

    Partially symmetrized and shear cores require finite capacity because
    their discrete plans are precomputed.  ``precompute_shuffle=False``
    omits the optional shuffle law, ``"generator"`` retains only
    first-level-factor shuffles, and ``True`` retains arbitrary shuffles
    within capacity.  Standard total degree accepts only the boolean modes.

    When ``seq_core`` is omitted, the resolved core may provide either a
    ``make_sequential_core`` method or a ``default_sequential_core`` attribute.
    Built-in and composed JAX cores automatically reuse the shared
    :class:`~tensordev.core.jax.JaxSequentialCore` singleton.

    Returns
    -------
    Any
        The installed algebra core.
    """
    construction_arguments = {
        "dims": dims,
        "max_trunc": max_trunc,
        "default_trunc": default_trunc,
        "representation": representation,
        "coordinates": coordinates,
        "precompute_shuffle": precompute_shuffle,
    }
    supplied_construction_arguments = tuple(
        name
        for name, value in construction_arguments.items()
        if (
            (name == "precompute_shuffle" and value is not False)
            or (name == "representation" and value != "ordered")
            or (name == "coordinates" and value != "standard")
            or (
                name not in {
                    "precompute_shuffle",
                    "representation",
                    "coordinates",
                }
                and value is not None
            )
        )
    )

    if core is not None:
        # Validate the shuffle argument before rejecting construction options
        # that cannot accompany an already constructed core.
        from tensordev.core.shuffle import _normalize_precompute_shuffle

        _normalize_precompute_shuffle(
            precompute_shuffle,
            allow_generator=True,
        )
        if supplied_construction_arguments:
            joined = ", ".join(supplied_construction_arguments)
            raise TypeError(
                "core cannot be combined with construction arguments: "
                f"{joined}."
            )
        resolved_core = core
    else:
        if dims is None:
            raise TypeError(
                "core must not be None; set_default_core requires either a "
                "constructed core or dims=."
            )
        if max_trunc is None:
            raise TypeError(
                "max_trunc is required when set_default_core constructs a core."
            )

        configuration = _resolve_core_configuration(
            dims=dims,
            max_trunc=max_trunc,
            representation=representation,
            coordinates=coordinates,
            precompute_shuffle=precompute_shuffle,
        )
        resolved_core = configuration.construct(default_trunc=default_trunc)

    if resolved_core is None:
        raise TypeError("core must not be None.")
    resolved_seq_core = _resolve_seq_core(resolved_core, seq_core)
    if resolved_seq_core is None:
        raise TypeError("The sequential-core provider returned None.")

    _replace_default_core_pair(resolved_core, resolved_seq_core)
    return resolved_core


def reset_default_core() -> None:
    """Restore the singleton pair selected by ``TENSORDEV_BACKEND``."""
    core, seq_core = _get(_INITIAL_BACKEND)
    _replace_default_core_pair(core, seq_core)
