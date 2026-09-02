from __future__ import annotations

import math
import re
from collections import Counter

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td


_DIMS = (2, 2)
_ACTIVE = (2, 1)
_INCREMENTS = jnp.asarray(
    (
        (0.10, -0.20, 0.05, 0.15),
        (-0.04, 0.12, 0.20, -0.10),
        (0.08, 0.03, -0.09, 0.11),
    ),
    dtype=jnp.float64,
)
_PATH = jnp.concatenate(
    (
        jnp.zeros((1, 4), dtype=_INCREMENTS.dtype),
        jnp.cumsum(_INCREMENTS, axis=0),
    ),
    axis=0,
)
_KERNEL = td.FractionalKernel(
    beta=jnp.asarray([0.7], dtype=jnp.float64),
    A=jnp.eye(4, dtype=jnp.float64)[None, :, :],
)


@pytest.fixture(scope="module")
def capacity_cores():
    small = td.bigraded_core(
        dims=_DIMS,
        max_trunc=_ACTIVE,
        default_trunc=_ACTIVE,
        precompute_shuffle=False,
    )
    large = td.bigraded_core(
        dims=_DIMS,
        max_trunc=(3, 2),
        default_trunc=_ACTIVE,
        precompute_shuffle=False,
    )
    return small, large


def _stablehlo_fingerprint(function, argument):
    lowered = jax.jit(function).lower(argument)
    text = str(lowered.compiler_ir(dialect="stablehlo"))
    operations = Counter(re.findall(r"stablehlo\.([a-zA-Z0-9_]+)", text))
    tensor_shapes = Counter(re.findall(r"tensor<[^>]+>", text))
    return operations, tensor_shapes


def _expected_coordinate_count():
    d_prime, d_doubleprime = _DIMS
    return sum(
        math.comb(n + m, n) * d_prime**n * d_doubleprime**m
        for n in range(_ACTIVE[0] + 1)
        for m in range(_ACTIVE[1] + 1)
    )


def _assert_same_active_tensor(small_result, large_result):
    assert small_result.spec == large_result.spec
    assert small_result.truncation == _ACTIVE
    assert large_result.truncation == _ACTIVE
    assert small_result.grades == (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (2, 1),
    )
    assert large_result.grades == small_result.grades

    small_leaves = jax.tree_util.tree_leaves(small_result)
    large_leaves = jax.tree_util.tree_leaves(large_result)
    assert [leaf.shape for leaf in small_leaves] == [
        leaf.shape for leaf in large_leaves
    ]
    assert sum(int(leaf.size) for leaf in small_leaves) == _expected_coordinate_count()
    assert sum(int(leaf.size) for leaf in large_leaves) == _expected_coordinate_count()

    for small_block, large_block in zip(small_result.blocks, large_result.blocks):
        np.testing.assert_allclose(
            np.asarray(small_block),
            np.asarray(large_block),
            atol=1e-12,
            rtol=1e-12,
        )


def test_signature_active_program_and_payload_are_capacity_independent(
    capacity_cores,
):
    small, large = capacity_cores

    def signature_function(core):
        seq_core = core.make_sequential_core()

        def evaluate(value):
            return td.path_signature(
                value,
                trunc=_ACTIVE,
                axis=-2,
                increment_input=True,
                core=core,
                seq_core=seq_core,
            )

        return evaluate

    small_function = signature_function(small)
    large_function = signature_function(large)
    small_result = small_function(_INCREMENTS)
    large_result = large_function(_INCREMENTS)

    _assert_same_active_tensor(small_result, large_result)
    assert _stablehlo_fingerprint(
        small_function, _INCREMENTS
    ) == _stablehlo_fingerprint(large_function, _INCREMENTS)


def test_volterra_active_program_and_payload_are_capacity_independent(
    capacity_cores,
):
    small, large = capacity_cores

    def volterra_function(core):
        seq_core = core.make_sequential_core()

        def evaluate(value):
            return td.vsig(
                value,
                kernel=_KERNEL,
                trunc=_ACTIVE,
                dt=0.25,
                axis=-2,
                order=0,
                scheme="quadratic",
                core=core,
                seq_core=seq_core,
            )

        return evaluate

    small_function = volterra_function(small)
    large_function = volterra_function(large)
    small_result = small_function(_PATH)
    large_result = large_function(_PATH)

    _assert_same_active_tensor(small_result, large_result)
    assert _stablehlo_fingerprint(
        small_function, _PATH
    ) == _stablehlo_fingerprint(large_function, _PATH)
