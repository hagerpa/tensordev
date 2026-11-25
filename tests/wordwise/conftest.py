"""Fixtures of the wordwise tests."""

import jax
import pytest


@pytest.fixture
def fresh_jit_caches():
    """Trace the jitted public routes afresh, so that patched helpers inside them take effect."""
    jax.clear_caches()
    yield
    jax.clear_caches()
