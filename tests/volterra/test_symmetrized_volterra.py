"""Volterra integration for both partially symmetrized coordinate systems."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev.volterra import FractionalKernel, vsig


_PATH = jnp.asarray(
    [[0.0, 0.0], [0.2, -0.1], [0.15, 0.15]],
    dtype=jnp.float64,
)


def _kernel(component_count):
    if component_count == 1:
        return FractionalKernel(
            beta=jnp.asarray([0.7], dtype=jnp.float64),
            A=jnp.eye(2, dtype=jnp.float64)[None, :, :],
        )
    return FractionalKernel(
        beta=jnp.asarray([0.65, 1.2], dtype=jnp.float64),
        A=jnp.asarray(
            [
                [[1.0, 0.0], [0.2, 0.7]],
                [[-0.3, 0.4], [0.8, 0.1]],
            ],
            dtype=jnp.float64,
        ),
    )


def _assert_bigraded_close(actual, expected):
    assert actual.spec == expected.spec
    for grade in actual.grades:
        np.testing.assert_allclose(
            actual[grade],
            expected[grade],
            atol=2e-10,
            rtol=2e-10,
        )


def _forbid_ordered_expansion(monkeypatch, core):
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "native quotient Volterra execution expanded through ordered layout"
        )

    for name in (
        "_lift_partially_symmetrized",
        "_lift_partially_symmetrized_block",
        "tensor_to_total",
        "tensor_from_total",
    ):
        monkeypatch.setattr(core, name, forbidden)


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
@pytest.mark.parametrize(
    ("component_count", "precompute_shuffle"),
    ((1, False), (2, "generator")),
)
def test_quotient_volterra_is_q_of_ordered_native_result(
    monkeypatch,
    coordinates,
    component_count,
    precompute_shuffle,
):
    active = (1, 1)
    ordered_standard = td.bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=precompute_shuffle,
    )
    ordered = (
        ordered_standard
        if coordinates == "standard"
        else td.shear_core(ordered_standard)
    )
    quotient = td.symmetrized_core(ordered)
    options = dict(
        kernel=_kernel(component_count),
        trunc=active,
        dt=0.4,
        scheme="quadratic",
    )

    ordered_result = vsig(_PATH, core=ordered, **options)
    _forbid_ordered_expansion(monkeypatch, quotient)
    direct = vsig(_PATH, core=quotient, **options)
    expected = quotient.tensor_partially_symmetrize(ordered_result)

    _assert_bigraded_close(direct, expected)


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_multicomponent_quotient_volterra_requires_generator_shuffle(
    coordinates,
):
    standard = td.bigraded_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        representation="partially_symmetrized",
        precompute_shuffle=False,
    )
    core = standard if coordinates == "standard" else td.shear_core(standard)

    with pytest.raises(RuntimeError, match="precompute_shuffle='generator'"):
        vsig(
            _PATH[:2],
            kernel=_kernel(2),
            trunc=(1, 1),
            dt=0.4,
            scheme="quadratic",
            core=core,
        )


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
@pytest.mark.parametrize(
    ("scheme", "component_count", "precompute_shuffle"),
    (
        ("fft", 1, False),
        ("fft", 2, "generator"),
        ("adams", 1, False),
        ("auto", 1, False),
        ("auto", 2, "generator"),
    ),
)
def test_other_supported_quotient_volterra_routes_are_native_q_projections(
    monkeypatch,
    coordinates,
    scheme,
    component_count,
    precompute_shuffle,
):
    active = (1, 1)
    ordered_standard = td.bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=precompute_shuffle,
    )
    ordered = (
        ordered_standard
        if coordinates == "standard"
        else td.shear_core(ordered_standard)
    )
    quotient = td.symmetrized_core(ordered)

    _forbid_ordered_expansion(monkeypatch, quotient)

    options = dict(
        kernel=_kernel(component_count),
        trunc=active,
        dt=0.4,
        scheme=scheme,
    )
    ordered_result = vsig(_PATH, core=ordered, **options)
    direct = vsig(_PATH, core=quotient, **options)
    expected = quotient.tensor_partially_symmetrize(ordered_result)

    _assert_bigraded_close(direct, expected)
