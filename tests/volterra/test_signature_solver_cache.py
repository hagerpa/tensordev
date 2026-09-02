from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from tensordev import Jax
from tensordev.volterra import (
    FractionalKernel,
    GammaKernel,
    VolterraSignature,
    quadratic_iteration,
    vsig,
)
from tensordev.volterra.iteration_fft import fft_iteration, precompute_lag_tables
from tensordev.volterra.signature import (
    _SOLVER_CACHE_MAXSIZE,
    _cached_compiled_solver,
    _clear_solver_cache,
    _kernel_rebuilder,
    _solver_cache_info,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _NestedFractionalKernel(FractionalKernel):
    """Test-only kernel whose nested PyTree requires the eager fallback."""

    nested: tuple[jax.Array, ...]


def _kernel(scale: float, beta: float = 0.7) -> FractionalKernel:
    return FractionalKernel(
        beta=jnp.asarray([beta], dtype=jnp.float64),
        A=jnp.asarray([[[scale]]], dtype=jnp.float64),
    )


def _path(scale: float = 1.0):
    return scale * jnp.asarray([[0.0], [0.2], [-0.1]], dtype=jnp.float64)


def _assert_tree_allclose(got, expected):
    got_leaves = jax.tree_util.tree_leaves(got)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(got_leaves) == len(expected_leaves)
    for got_leaf, expected_leaf in zip(got_leaves, expected_leaves):
        np.testing.assert_allclose(got_leaf, expected_leaf, atol=1e-12, rtol=1e-12)


def test_whole_solver_cache_reuses_structure_but_not_kernel_or_path_values():
    core = Jax()
    kernel_one = _kernel(1.0, beta=0.65)
    kernel_two = _kernel(2.0, beta=1.1)
    _clear_solver_cache()

    first = vsig(
        _path(),
        kernel=kernel_one,
        trunc=2,
        dt=0.25,
        scheme="quadratic",
        core=core,
    )
    jax.block_until_ready(first)
    after_first = _solver_cache_info()

    changed = vsig(
        _path(1.5),
        kernel=kernel_two,
        trunc=2,
        dt=0.4,
        scheme="quadratic",
        core=core,
    )
    jax.block_until_ready(changed)
    after_second = _solver_cache_info()

    reference = quadratic_iteration(
        jnp.diff(_path(1.5), axis=0),
        kernel=kernel_two,
        trunc=2,
        dt=0.4,
        core=core,
    )
    for got, expected in zip(changed, reference):
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)

    assert after_first.misses == 1
    assert after_first.hits == 0
    assert after_second.misses == 1
    assert after_second.hits == 1
    assert not np.allclose(np.asarray(first[1]), np.asarray(changed[1]))


def test_volterra_signature_wrappers_share_a_structural_solver_boundary():
    core = Jax()
    _clear_solver_cache()
    signature_one = VolterraSignature(kernel=_kernel(1.0), trunc=2, core=core)
    signature_two = VolterraSignature(kernel=_kernel(1.7), trunc=2, core=core)

    first = signature_one.vsig(_path(), dt=0.3, scheme="quadratic")
    second = signature_two.vsig(_path(), dt=0.3, scheme="quadratic")
    jax.block_until_ready((first, second))

    info = _solver_cache_info()
    assert info.currsize == 1
    assert info.misses == 1
    assert info.hits == 1
    assert not np.allclose(np.asarray(first[1]), np.asarray(second[1]))


def test_whole_solver_cache_has_a_hard_size_bound():
    core = Jax()
    kernel = _kernel(1.0)
    rebuilder, _, _ = _kernel_rebuilder(kernel)
    _clear_solver_cache()

    kwargs = dict(
        rebuilder=rebuilder,
        trunc=1,
        axis=-2,
        block_size=None,
        accumulate=True,
        output_starting_point=False,
        increment_input=False,
        order=0,
        dyadic_order=0,
        scheme="quadratic",
        core=core,
        seq_core=VolterraSignature(kernel=kernel, trunc=1, core=core).seq_core,
    )
    for index in range(_SOLVER_CACHE_MAXSIZE + 3):
        _cached_compiled_solver(("bounded-cache-test", index), **kwargs)

    info = _solver_cache_info()
    assert info.currsize == _SOLVER_CACHE_MAXSIZE
    assert info.misses == _SOLVER_CACHE_MAXSIZE + 3
    _clear_solver_cache()


def test_gamma_static_metadata_and_higher_order_beta_separate_solvers():
    core = Jax()
    gamma_four = GammaKernel(
        beta=jnp.asarray([0.8]),
        A=jnp.ones((1, 1, 1)),
        scale=1.0,
        rate=0.7,
        quad_order=4,
    )
    gamma_eight = GammaKernel(
        beta=jnp.asarray([0.8]),
        A=jnp.ones((1, 1, 1)),
        scale=1.0,
        rate=0.7,
        quad_order=8,
    )
    fractional_a = _kernel(1.0, beta=0.6)
    fractional_b = _kernel(1.5, beta=0.6)
    fractional_c = _kernel(1.5, beta=0.9)
    _clear_solver_cache()

    for kernel in (gamma_four, gamma_eight):
        jax.block_until_ready(
            vsig(_path(), kernel=kernel, trunc=1, scheme="quadratic", core=core)
        )
    assert _solver_cache_info().misses == 2

    _clear_solver_cache()
    for kernel in (fractional_a, fractional_b, fractional_c):
        jax.block_until_ready(
            vsig(
                _path(),
                kernel=kernel,
                trunc=1,
                order=1,
                scheme="quadratic",
                core=core,
            )
        )
    info = _solver_cache_info()
    assert info.misses == 2
    assert info.hits == 1
    assert info.currsize == 2


def test_precomputed_lag_table_values_are_dynamic_cache_inputs():
    core = Jax()
    kernel = _kernel(1.0, beta=0.75)
    tables = precompute_lag_tables(
        kernel,
        S=2,
        h=0.25,
        order=0,
        trunc=2,
        dtype=jnp.float64,
    )
    scaled_tables = jax.tree.map(lambda leaf: 0.5 * leaf, tables)
    _clear_solver_cache()

    first = vsig(
        _path(),
        kernel=kernel,
        trunc=2,
        dt=0.25,
        scheme="fft",
        lag_tables=tables,
        core=core,
    )
    changed = vsig(
        _path(),
        kernel=kernel,
        trunc=2,
        dt=0.25,
        scheme="fft",
        lag_tables=scaled_tables,
        core=core,
    )
    reference = fft_iteration(
        jnp.diff(_path(), axis=0),
        kernel=kernel,
        trunc=2,
        dt=0.25,
        lag_tables=scaled_tables,
        core=core,
    )
    jax.block_until_ready((first, changed, reference))

    _assert_tree_allclose(changed, reference)
    info = _solver_cache_info()
    assert info.misses == 1
    assert info.hits == 1
    assert not np.allclose(np.asarray(first[1]), np.asarray(changed[1]))


def test_nested_kernel_pytree_uses_correct_eager_fallback():
    core = Jax()
    kernel = _NestedFractionalKernel(
        beta=jnp.asarray([0.7]),
        A=jnp.ones((1, 1, 1)),
        nested=(jnp.asarray([1.0]),),
    )
    assert _kernel_rebuilder(kernel) is None
    _clear_solver_cache()

    got = vsig(
        _path(),
        kernel=kernel,
        trunc=2,
        dt=0.3,
        scheme="quadratic",
        core=core,
    )
    expected = quadratic_iteration(
        jnp.diff(_path(), axis=0),
        kernel=kernel,
        trunc=2,
        dt=0.3,
        core=core,
    )
    _assert_tree_allclose(got, expected)
    assert _solver_cache_info().currsize == 0


def test_cached_boundary_composes_with_nested_jit_vmap_and_grad():
    core = Jax()
    kernel = _kernel(1.0, beta=0.8)
    _clear_solver_cache()

    def terminal(path, dt):
        result = vsig(
            path,
            kernel=kernel,
            trunc=1,
            dt=dt,
            scheme="quadratic",
            core=core,
        )
        return jnp.sum(result[1])

    paths = jnp.stack((_path(), 1.5 * _path()))
    dts = jnp.asarray([0.2, 0.4])
    transformed = jax.jit(jax.vmap(terminal))(paths, dts)
    expected = jnp.stack(tuple(terminal(path, dt) for path, dt in zip(paths, dts)))
    np.testing.assert_allclose(transformed, expected, atol=1e-12, rtol=1e-12)

    path_grad = jax.jit(jax.grad(terminal, argnums=0))(_path(), 0.3)
    assert bool(jnp.all(jnp.isfinite(path_grad)))
    assert _solver_cache_info().currsize == 1
