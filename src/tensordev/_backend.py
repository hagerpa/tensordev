from __future__ import annotations

import os
from typing import Any

_DEFAULT_BACKEND = os.environ.get("TENSORDEV_BACKEND", "jax")

_core_cache: dict[str, Any] = {}
_seq_core_cache: dict[str, Any] = {}


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


def get_default_core() -> Any:
    return _get(_DEFAULT_BACKEND)[0]


def get_default_seq_core() -> Any:
    return _get(_DEFAULT_BACKEND)[1]
