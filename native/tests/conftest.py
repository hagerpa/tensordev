from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="session")
def native_extension():
    try:
        return importlib.import_module("tensordev._native_cpu")
    except ModuleNotFoundError as error:
        if error.name not in {"tensordev", "tensordev._native_cpu"}:
            raise
        return importlib.import_module("tensordev_native_cpu")
