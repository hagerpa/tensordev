# universal/annotations.py
from __future__ import annotations
from typing import Callable, Iterable, Optional, Any

JIT_TAG = "__is_jittable__"
JIT_KW = "__jit_kwargs__"


def jit(
        func: Optional[Callable] = None,
        *,
        # convenience kwargs (typed so IDEs help you)
        static_argnums: Optional[Iterable[int] | int] = None,
        static_argnames: Optional[Iterable[str] | str] = None,
        dynamic_batchtime: Optional[Iterable[str] | str] = None,
        full_dynamic: Optional[Iterable[str] | str] = None,
        nopython: Optional[bool] = None,  # for numba etc.
        no_python: Optional[bool] = None,  # alias for convenience
        **extra: Any,  # anything else gets recorded too
):
    """Metadata-only: mark as jittable and stash ALL kwargs exactly as given."""

    def decorate(f: Callable):
        recorded = {}
        if static_argnums is not None:  recorded["static_argnums"] = static_argnums
        if static_argnames is not None: recorded["static_argnames"] = static_argnames
        if dynamic_batchtime is not None: recorded["dynamic_batchtime"] = dynamic_batchtime
        if full_dynamic is not None:     recorded["full_dynamic"] = full_dynamic
        if nopython is not None:        recorded["nopython"] = nopython
        if no_python is not None and "nopython" not in recorded:
            recorded["nopython"] = no_python  # alias

        recorded.update(extra)  # keep everything else verbatim

        setattr(f, JIT_TAG, True)
        setattr(f, JIT_KW, recorded)
        return f

    return decorate if func is None else decorate(func)


def is_jittable(obj: Any) -> bool:
    return bool(getattr(obj, JIT_TAG, False))


def get_jit_kwargs(obj: Any) -> dict:
    return getattr(obj, JIT_KW, {})


def iter_class_jittables(cls: type):
    for name in dir(cls):
        try:
            attr = getattr(cls, name)
        except Exception:
            continue
        if callable(attr) and is_jittable(attr):
            yield name, attr, get_jit_kwargs(attr)
