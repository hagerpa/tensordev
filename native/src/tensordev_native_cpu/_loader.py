"""Load and retain the TensorDev CPU FFI shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from threading import RLock


_REGISTRATION_SYMBOLS = {
    "tensordev_cpu_sym_horner_f32_v2": {
        "instantiate": "TensordevFusedRaggedHornerInstantiate",
        "execute": "TensordevFusedRaggedHornerF32Dynamic",
    },
    "tensordev_cpu_sym_horner_f64_v2": {
        "instantiate": "TensordevFusedRaggedHornerInstantiate",
        "execute": "TensordevFusedRaggedHornerF64Dynamic",
    },
}
_TYPE_REGISTRATION_SYMBOLS = {
    "tensordev.ragged_horner_state.v1": {
        "type_id": "TensordevRaggedHornerStateTypeId",
        "type_info": "TensordevRaggedHornerStateTypeInfo",
    }
}
_LOCK = RLock()
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_PATH: Path | None = None
_CAPSULES: dict[str, object] | None = None
_TYPE_CAPSULES: dict[str, object] | None = None


def _shared_library_path() -> Path:
    package_dir = Path(__file__).resolve().parent
    if sys.platform == "darwin":
        suffixes = (".dylib", ".so")
    elif sys.platform.startswith("linux"):
        suffixes = (".so",)
    else:
        raise ImportError("tensordev-native-cpu supports Linux and macOS only")

    candidates = sorted(
        path
        for path in package_dir.glob("_tensordev_native_cpu*")
        if path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)
    )
    if len(candidates) != 1:
        found = ", ".join(path.name for path in candidates) or "none"
        raise ImportError(
            "expected exactly one TensorDev CPU shared library in "
            f"{package_dir}, found {found}"
        )
    return candidates[0]


def _load_library() -> ctypes.CDLL:
    global _LIBRARY, _LIBRARY_PATH
    with _LOCK:
        if _LIBRARY is None:
            path = _shared_library_path()
            try:
                library = ctypes.CDLL(str(path))
            except OSError as error:
                raise ImportError(
                    f"could not load TensorDev CPU library {path}"
                ) from error
            symbols = {
                symbol
                for stages in _REGISTRATION_SYMBOLS.values()
                for symbol in stages.values()
            }
            symbols.update(
                symbol
                for registration in _TYPE_REGISTRATION_SYMBOLS.values()
                for symbol in registration.values()
            )
            for symbol in symbols:
                try:
                    getattr(library, symbol)
                except AttributeError as error:
                    raise ImportError(
                        f"TensorDev CPU library does not export {symbol}"
                    ) from error
            _LIBRARY = library
            _LIBRARY_PATH = path
        return _LIBRARY


def library_path() -> Path:
    """Return the loaded native library's absolute path."""
    _load_library()
    if _LIBRARY_PATH is None:
        raise AssertionError("native library path was not retained")
    return _LIBRARY_PATH


def registrations() -> dict[str, object]:
    """Return fresh registration mapping backed by retained FFI capsules."""
    global _CAPSULES
    with _LOCK:
        library = _load_library()
        if _CAPSULES is None:
            from jax import ffi

            _CAPSULES = {
                name: {
                    stage: ffi.pycapsule(getattr(library, symbol))
                    for stage, symbol in stages.items()
                }
                for name, stages in _REGISTRATION_SYMBOLS.items()
            }
        return dict(_CAPSULES)


def type_registrations() -> dict[str, object]:
    """Return state-type registrations backed by the loaded library."""
    global _TYPE_CAPSULES
    with _LOCK:
        library = _load_library()
        if _TYPE_CAPSULES is None:
            from jax import ffi

            registrations = {}
            for name, symbols in _TYPE_REGISTRATION_SYMBOLS.items():
                registration = {}
                for field, symbol in symbols.items():
                    function = getattr(library, symbol)
                    function.restype = ctypes.c_void_p
                    pointer = function()
                    if pointer is None:
                        raise ImportError(
                            f"TensorDev CPU library returned no {field} for {name}"
                        )
                    registration[field] = ffi.pycapsule(ctypes.c_void_p(pointer))
                registrations[name] = registration
            _TYPE_CAPSULES = registrations
        return dict(_TYPE_CAPSULES)
