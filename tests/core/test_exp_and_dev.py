from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tensordev import Jax

CORE = Jax()


def _random_first_on_elem(key, *, dim, trunc, batch_shape=(), scale=0.10):
    keys = jr.split(key, trunc)
    return tuple(
        scale * jr.normal(keys[k - 1], batch_shape + (dim ** k,), dtype=jnp.float64)
        for k in range(1, trunc + 1)
    )


def _random_dense_elem(key, *, dim, trunc, batch_shape=(), scale=0.12):
    keys = jr.split(key, trunc + 1)
    return tuple(
        scale * jr.normal(keys[k], batch_shape + (dim ** k,), dtype=jnp.float64)
        for k in range(trunc + 1)
    )


def _unit_dense(*, batch_shape=(), dtype=jnp.float64):
    return (jnp.ones(batch_shape + (1,), dtype=dtype),)


def _assert_tuple_allclose(a, b, *, rtol=1e-10, atol=1e-10):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


def _random_first_on_increments(key, *, dim, trunc, steps, batch_shape=(), scale=0.08):
    keys = jr.split(key, trunc)
    return tuple(
        scale * jr.normal(keys[k - 1], batch_shape + (steps, dim ** k), dtype=jnp.float64)
        for k in range(1, trunc + 1)
    )


def _stepwise_fmexp_scan(increments, *, trunc):
    """
    Reference scan:
        G_{r+1} = G_r ⊗ exp(ΔX_{r+1})
    implemented via tensor_fmexp.
    """
    steps = increments[0].shape[-2]
    state = _unit_dense(batch_shape=increments[0].shape[:-2])
    states = []

    for s in range(steps):
        inc_s = tuple(level[..., s, :] for level in increments)
        state = CORE.tensor_fmexp(
            state,
            inc_s,
            trunc=trunc,
            output_zero_level=True,
        )
        states.append(state)

    return state, tuple(
        jnp.stack([state_s[k] for state_s in states], axis=-2)
        for k in range(trunc + 1)
    )


from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.random as jr
import jax.numpy as jnp

from tensordev import Jax

CORE = Jax()


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_tensor_exponential_degree_3_matches_manual_formula(batch_shape):
    """
    For X = (X1, X2, X3), check the degree-0..3 formula for exp(X) directly.
    This isolates cubic-order mistakes in tensor_exponential / tensor_fmexp.
    """
    dim = 2
    trunc = 3
    key = jr.PRNGKey(1234 + len(batch_shape))
    k1, k2, k3 = jr.split(key, 3)

    X1 = 0.1 * jr.normal(k1, batch_shape + (dim,), dtype=jnp.float64)
    X2 = 0.1 * jr.normal(k2, batch_shape + (dim ** 2,), dtype=jnp.float64)
    X3 = 0.1 * jr.normal(k3, batch_shape + (dim ** 3,), dtype=jnp.float64)

    X = (X1, X2, X3)

    E = CORE.tensor_exponential(
        X,
        trunc=trunc,
        output_zero_level=True,
    )

    one = jnp.ones(batch_shape, dtype=jnp.float64)

    X1X1 = CORE.tensor_product_homogeneous(X1, X1)
    X1X2 = CORE.tensor_product_homogeneous(X1, X2)
    X2X1 = CORE.tensor_product_homogeneous(X2, X1)
    X1X1X1 = CORE.tensor_product_homogeneous(X1X1, X1)

    ref0 = jnp.ones(batch_shape + (1,), dtype=jnp.float64)
    ref1 = X1
    ref2 = X2 + 0.5 * X1X1
    ref3 = X3 + 0.5 * (X1X2 + X2X1) + (1.0 / 6.0) * X1X1X1

    ref = (ref0, ref1, ref2, ref3)

    assert len(E) == 4
    for a, b in zip(E, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("dim", "trunc", "batch_shape"), [
    (2, 1, ()),
    (2, 2, ()),
    (2, 3, ()),
    (2, 3, (2,)),
    (3, 2, ()),
    (3, 2, (2,)),
])
def test_tensor_logarithm_of_tensor_exponential_recovers_first_on_input(dim, trunc, batch_shape):
    """
    tensor_logarithm takes a first-on input representing the positive levels of
    a dense element with implicit level-0 equal to 1.

    Hence, if G = exp(X) is returned densely, then
        log(G[1:]) = X.
    """
    key = jr.PRNGKey(1000 + 100 * dim + 10 * trunc + len(batch_shape))
    X = _random_first_on_elem(
        key,
        dim=dim,
        trunc=trunc,
        batch_shape=batch_shape,
        scale=0.10,
    )

    G = CORE.tensor_exponential(
        X,
        trunc=trunc,
        output_zero_level=True,
    )
    L = CORE.tensor_logarithm(
        G[1:],
        trunc=trunc,
        output_zero_level=False,
    )

    _assert_tuple_allclose(L, X)


@pytest.mark.parametrize(("dim", "trunc", "batch_shape"), [
    (2, 1, ()),
    (2, 2, ()),
    (2, 3, ()),
    (2, 3, (2,)),
    (3, 2, ()),
    (3, 2, (2,)),
])
def test_tensor_logarithm_of_tensor_exponential_returns_zero_level_then_input(dim, trunc, batch_shape):
    """
    With output_zero_level=True we should get
        log(exp(X)[1:]) = (0, X_1, ..., X_trunc).
    """
    key = jr.PRNGKey(1500 + 100 * dim + 10 * trunc + len(batch_shape))
    X = _random_first_on_elem(
        key,
        dim=dim,
        trunc=trunc,
        batch_shape=batch_shape,
        scale=0.10,
    )

    G = CORE.tensor_exponential(
        X,
        trunc=trunc,
        output_zero_level=True,
    )
    L = CORE.tensor_logarithm(
        G[1:],
        trunc=trunc,
        output_zero_level=True,
    )

    assert len(L) == trunc + 1
    np.testing.assert_allclose(
        np.asarray(L[0]),
        np.zeros_like(np.asarray(G[0])),
        rtol=1e-10,
        atol=1e-10,
    )
    _assert_tuple_allclose(L[1:], X)


@pytest.mark.parametrize(("dim", "trunc", "batch_shape"), [
    (2, 1, ()),
    (2, 2, ()),
    (2, 3, ()),
    (2, 3, (2,)),
    (3, 2, ()),
    (3, 2, (2,)),
])
def test_tensor_exponential_of_tensor_logarithm_recovers_group_like_element(dim, trunc, batch_shape):
    """
    For G = exp(X), we should have
        exp(log(G[1:])) = G.
    """
    key = jr.PRNGKey(2000 + 100 * dim + 10 * trunc + len(batch_shape))
    X = _random_first_on_elem(
        key,
        dim=dim,
        trunc=trunc,
        batch_shape=batch_shape,
        scale=0.10,
    )

    G = CORE.tensor_exponential(
        X,
        trunc=trunc,
        output_zero_level=True,
    )
    L = CORE.tensor_logarithm(
        G[1:],
        trunc=trunc,
        output_zero_level=False,
    )
    G_recovered = CORE.tensor_exponential(
        L,
        trunc=trunc,
        output_zero_level=True,
    )

    _assert_tuple_allclose(G_recovered, G)


@pytest.mark.parametrize(("dim", "trunc", "batch_shape"), [
    (2, 1, ()),
    (2, 2, ()),
    (2, 3, ()),
    (2, 3, (2,)),
    (3, 2, ()),
    (3, 2, (2,)),
])
def test_tensor_fmexp_matches_left_multiplication_by_tensor_exponential(dim, trunc, batch_shape):
    """
    fmexp(g, X) = g ⊗ exp(X).
    """
    key = jr.PRNGKey(3500 + 100 * dim + 10 * trunc + len(batch_shape))
    kg, kx = jr.split(key, 2)

    g = _random_dense_elem(
        kg,
        dim=dim,
        trunc=trunc,
        batch_shape=batch_shape,
        scale=0.12,
    )
    X = _random_first_on_elem(
        kx,
        dim=dim,
        trunc=trunc,
        batch_shape=batch_shape,
        scale=0.12,
    )

    fm = CORE.tensor_fmexp(
        g,
        X,
        trunc=trunc,
        output_zero_level=True,
    )

    expX = CORE.tensor_exponential(
        X,
        trunc=trunc,
        output_zero_level=True,
    )

    ref = CORE.tensor_product(
        g,
        expX,
        trunc=trunc,
        a_first_on=False,
        b_first_on=False,
    )

    _assert_tuple_allclose(fm, ref)


@pytest.mark.parametrize(("dim", "trunc", "batch_shape"), [
    (2, 1, ()),
    (2, 2, ()),
    (2, 3, ()),
    (2, 3, (2,)),
    (3, 2, ()),
    (3, 2, (2,)),
])
def test_tensor_logarithm_of_tensor_fmexp_with_unit_recovers_input(dim, trunc, batch_shape):
    """
    Since fmexp(1, X) = exp(X), we should have
        log(fmexp(1, X)[1:]) = X.
    """
    key = jr.PRNGKey(4000 + 100 * dim + 10 * trunc + len(batch_shape))
    X = _random_first_on_elem(
        key,
        dim=dim,
        trunc=trunc,
        batch_shape=batch_shape,
        scale=0.10,
    )

    one = _unit_dense(batch_shape=batch_shape)

    G = CORE.tensor_fmexp(
        one,
        X,
        trunc=trunc,
        output_zero_level=True,
    )

    L = CORE.tensor_logarithm(
        G[1:],
        trunc=trunc,
        output_zero_level=False,
    )

    _assert_tuple_allclose(L, X)


@pytest.mark.parametrize(("dim", "trunc", "batch_shape"), [
    (2, 1, ()),
    (2, 2, ()),
    (2, 3, ()),
    (2, 3, (2,)),
    (3, 2, ()),
    (3, 2, (2,)),
])
def test_tensor_development_single_step_matches_tensor_exponential(dim, trunc, batch_shape):
    """
    For a single increment ΔX, tensor_development should return exp(ΔX).
    """
    key = jr.PRNGKey(5000 + 100 * dim + 10 * trunc + len(batch_shape))
    X = _random_first_on_increments(
        key,
        dim=dim,
        trunc=trunc,
        steps=1,
        batch_shape=batch_shape,
        scale=0.08,
    )

    dev = CORE.tensor_development(
        X,
        axis=-2,
        trunc=trunc,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )

    ref = CORE.tensor_exponential(
        tuple(level[..., 0, :] for level in X),
        trunc=trunc,
        output_zero_level=True,
    )

    _assert_tuple_allclose(dev, ref)


@pytest.mark.parametrize(("dim", "trunc", "steps", "batch_shape"), [
    (2, 2, 4, ()),
    (2, 3, 5, ()),
    (2, 3, 5, (2,)),
    (3, 2, 4, ()),
    (3, 2, 4, (2,)),
])
def test_tensor_development_terminal_matches_stepwise_fmexp_scan(dim, trunc, steps, batch_shape):
    """
    The terminal tensor development over increments should match repeated
    fmexp updates.
    """
    key = jr.PRNGKey(6000 + 100 * dim + 10 * trunc + steps + len(batch_shape))
    X = _random_first_on_increments(
        key,
        dim=dim,
        trunc=trunc,
        steps=steps,
        batch_shape=batch_shape,
        scale=0.08,
    )

    dev_terminal = CORE.tensor_development(
        X,
        axis=-2,
        trunc=trunc,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )

    ref_terminal, _ = _stepwise_fmexp_scan(X, trunc=trunc)

    _assert_tuple_allclose(dev_terminal, ref_terminal)


@pytest.mark.parametrize(("dim", "trunc", "steps", "batch_shape"), [
    (2, 2, 4, ()),
    (2, 3, 5, ()),
    (2, 3, 5, (2,)),
    (3, 2, 4, ()),
    (3, 2, 4, (2,)),
])
def test_tensor_development_accumulate_has_no_effect_for_single_block(dim, trunc, steps, batch_shape):
    """
    If tensor_development emits only a single block value, then accumulate=True
    and accumulate=False should agree.

    This is the intended semantics: accumulation only matters across blocks
    (or with a nontrivial starting point), not within a block.
    """
    key = jr.PRNGKey(7000 + 100 * dim + 10 * trunc + steps + len(batch_shape))
    X = _random_first_on_increments(
        key,
        dim=dim,
        trunc=trunc,
        steps=steps,
        batch_shape=batch_shape,
        scale=0.08,
    )

    dev_no_acc = CORE.tensor_development(
        X,
        axis=-2,
        trunc=trunc,
        block_size=None,  # one single block
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )
    dev_acc = CORE.tensor_development(
        X,
        axis=-2,
        trunc=trunc,
        block_size=None,  # one single block
        accumulate=True,
        output_starting_point=False,
        increment_input=True,
    )

    _assert_tuple_allclose(dev_acc, dev_no_acc)


@pytest.mark.parametrize(("dim", "trunc", "steps", "batch_shape", "block_size"), [
    (2, 2, 4, (), 2),
    (2, 3, 6, (), 2),
    (2, 3, 6, (2,), 2),
    (3, 2, 4, (), 2),
    (3, 2, 4, (2,), 2),
])
def test_tensor_development_accumulate_returns_block_prefix_developments(
        dim, trunc, steps, batch_shape, block_size
):
    """
    With multiple blocks, accumulate=True should return the prefix Chen products
    of the blockwise developments.

    Concretely, if
        B_0, ..., B_{L-1}
    are the contiguous blocks, and
        D_l = S(B_l),
    then tensor_development(..., accumulate=True) should return
        [D_0, D_0⊗D_1, ..., D_0⊗...⊗D_{L-1}].
    """
    key = jr.PRNGKey(8000 + 100 * dim + 10 * trunc + steps + len(batch_shape))
    X = _random_first_on_increments(
        key,
        dim=dim,
        trunc=trunc,
        steps=steps,
        batch_shape=batch_shape,
        scale=0.08,
    )

    block_devs = CORE.tensor_development(
        X,
        axis=-2,
        trunc=trunc,
        block_size=block_size,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )

    dev_prefixes = CORE.tensor_development(
        X,
        axis=-2,
        trunc=trunc,
        block_size=block_size,
        accumulate=True,
        output_starting_point=False,
        increment_input=True,
    )

    ref_prefixes = CORE.tensor_abra(
        block_devs,
        op="product",
        axis=-2,
        trunc=trunc,
        block_size=1,  # treat each emitted block-development as one step
        accumulate=True,
        output_starting_point=False,
    )

    _assert_tuple_allclose(dev_prefixes, ref_prefixes)


@pytest.mark.parametrize(("dim", "trunc", "steps", "batch_shape", "block_size"), [
    (2, 2, 4, (), 2),
    (2, 3, 6, (), 2),
    (2, 3, 6, (2,), 2),
    (3, 2, 4, (), 2),
    (3, 2, 4, (2,), 2),
])
def test_tensor_development_nonaccumulate_returns_block_developments(
        dim, trunc, steps, batch_shape, block_size
):
    """
    With accumulate=False and multiple blocks, tensor_development should return
    the development of each contiguous block separately.
    """
    key = jr.PRNGKey(9000 + 100 * dim + 10 * trunc + steps + len(batch_shape))
    X = _random_first_on_increments(
        key,
        dim=dim,
        trunc=trunc,
        steps=steps,
        batch_shape=batch_shape,
        scale=0.08,
    )

    out = CORE.tensor_development(
        X,
        axis=-2,
        trunc=trunc,
        block_size=block_size,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )

    n_blocks = steps // block_size
    ref_blocks = []

    for b in range(n_blocks):
        X_block = tuple(level[..., b * block_size:(b + 1) * block_size, :] for level in X)
        block_terminal, _ = _stepwise_fmexp_scan(X_block, trunc=trunc)
        ref_blocks.append(block_terminal)

    ref = tuple(
        jnp.stack([blk[k] for blk in ref_blocks], axis=-2)
        for k in range(trunc + 1)
    )

    _assert_tuple_allclose(out, ref)



def test_tensor_development_level1_matches_first_level_increment():
    """
    The level-1 component of the free/tensor development depends only on the
    level-1 increments of the driving tensor path.

    In particular, higher tensor levels in the input must not affect level 1.
    """
    # 4 increments in a 2D truncated tensor algebra, with nonzero levels 1,2,3
    dX1 = np.array([
        [0.10, -0.20],
        [0.30,  0.40],
        [-0.50, 0.60],
        [0.20, -0.10],
    ], dtype=float)

    dX2 = np.array([
        [1.0,  2.0,  3.0,  4.0],
        [0.5, -1.0,  1.5, -2.0],
        [2.0,  0.0, -1.0,  1.0],
        [3.0, -2.0,  0.5,  1.5],
    ], dtype=float)

    dX3 = np.array([
        [ 1.0,  0.0,  2.0, -1.0,  0.5,  1.5, -0.5,  2.5],
        [ 0.2, -0.3,  0.4,  0.1, -0.2,  0.7,  1.2, -0.8],
        [-1.0,  0.5,  0.0,  1.0,  2.0, -1.5,  0.3,  0.9],
        [ 0.6,  1.1, -0.4,  0.8, -0.7,  0.2,  1.4, -0.1],
    ], dtype=float)

    dev = CORE.tensor_development(
        (dX1, dX2, dX3),
        axis=-2,
        trunc=3,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )

    expected_level1 = dX1.sum(axis=0)
    actual_level1 = np.asarray(dev[1], dtype=float)

    np.testing.assert_allclose(
        actual_level1,
        expected_level1,
        atol=1e-12,
        rtol=1e-12,
        err_msg="level 1 of tensor development should equal total level-1 increment",
    )


def test_tensor_development_higher_levels_do_not_create_level1():
    dX1 = np.zeros((3, 2), dtype=float)
    dX2 = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 0.0, 1.5, 2.0],
        [-1.0, 1.0, 0.0, 0.5],
    ], dtype=float)
    dX3 = np.array([
        [1.0, 0.0, 2.0, 1.0, 0.5, 1.5, 0.5, 2.5],
        [0.2, 0.3, 0.4, 0.1, 0.2, 0.7, 1.2, 0.8],
        [1.0, 0.5, 0.0, 1.0, 2.0, 1.5, 0.3, 0.9],
    ], dtype=float)

    dev = CORE.tensor_development(
        (dX1, dX2, dX3),
        axis=-2,
        trunc=3,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )

    np.testing.assert_allclose(np.asarray(dev[1], dtype=float), 0.0, atol=1e-12, rtol=1e-12)