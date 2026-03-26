from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.random as jr
import jax.numpy as jnp

from tensordev import Jax

CORE = Jax()


@pytest.mark.parametrize(("d", "i", "j"), [(2, 1, 1), (2, 1, 2), (2, 2, 1), (3, 1, 1), (3, 1, 2)])
def test_tensor_product_homogeneous_matches_outer_flatten(d, i, j):
    """
    tensor_product_homogeneous(A_i, B_j) should be the row-major flattening of
    the outer product A_i ⊗ B_j.
    """
    key = jr.PRNGKey(1000 + 100 * d + 10 * i + j)
    ka, kb = jr.split(key, 2)

    a = jr.normal(ka, (d ** i,), dtype=jnp.float64)
    b = jr.normal(kb, (d ** j,), dtype=jnp.float64)

    out = CORE.tensor_product_homogeneous(a, b)
    ref = np.outer(np.asarray(a), np.asarray(b)).reshape(-1)

    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "i", "n"), [(2, 1, 1), (2, 1, 2), (2, 2, 1), (3, 1, 1), (3, 1, 2)])
def test_tensor_adjoint_left_homogeneous_matches_manual_contraction(d, i, n):
    """
    Left homogeneous adjoint should contract the first tensor block:
        Adj_left(A_i, Y_{n+i}) in degree n.
    """
    key = jr.PRNGKey(2000 + 100 * d + 10 * i + n)
    ka, ky = jr.split(key, 2)

    a = jr.normal(ka, (d ** i,), dtype=jnp.float64)
    y = jr.normal(ky, (d ** (n + i),), dtype=jnp.float64)

    out = CORE.tensor_adjoint_left_homogeneous(a, y)

    y_reshaped = np.asarray(y).reshape(d ** i, d ** n)
    ref = (np.asarray(a)[:, None] * y_reshaped).sum(axis=0)

    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "j", "n"), [(2, 1, 1), (2, 1, 2), (2, 2, 1), (3, 1, 1), (3, 1, 2)])
def test_tensor_adjoint_right_homogeneous_matches_manual_contraction(d, j, n):
    """
    Right homogeneous adjoint should contract the last tensor block:
        Adj_right(B_j, Y_{n+j}) in degree n.
    """
    key = jr.PRNGKey(3000 + 100 * d + 10 * j + n)
    kb, ky = jr.split(key, 2)

    b = jr.normal(kb, (d ** j,), dtype=jnp.float64)
    y = jr.normal(ky, (d ** (n + j),), dtype=jnp.float64)

    out = CORE.tensor_adjoint_right_homogeneous(b, y)

    y_reshaped = np.asarray(y).reshape(d ** n, d ** j)
    ref = (y_reshaped * np.asarray(b)[None, :]).sum(axis=1)

    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "i", "n"), [(2, 1, 1), (2, 1, 2), (2, 2, 1), (3, 1, 1), (3, 1, 2)])
def test_tensor_adjoint_left_homogeneous_satisfies_inner_product_identity(d, i, n):
    """
    <A_i ⊗ X_n, Y_{n+i}> = <X_n, Adj_left(A_i, Y_{n+i})>.
    """
    key = jr.PRNGKey(4000 + 100 * d + 10 * i + n)
    ka, kx, ky = jr.split(key, 3)

    a = jr.normal(ka, (d ** i,), dtype=jnp.float64)
    x = jr.normal(kx, (d ** n,), dtype=jnp.float64)
    y = jr.normal(ky, (d ** (n + i),), dtype=jnp.float64)

    lhs = np.dot(
        np.asarray(CORE.tensor_product_homogeneous(a, x)),
        np.asarray(y),
    )
    rhs = np.dot(
        np.asarray(x),
        np.asarray(CORE.tensor_adjoint_left_homogeneous(a, y)),
    )

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "j", "n"), [(2, 1, 1), (2, 1, 2), (2, 2, 1), (3, 1, 1), (3, 1, 2)])
def test_tensor_adjoint_right_homogeneous_satisfies_inner_product_identity(d, j, n):
    """
    <X_n ⊗ B_j, Y_{n+j}> = <X_n, Adj_right(B_j, Y_{n+j})>.
    """
    key = jr.PRNGKey(5000 + 100 * d + 10 * j + n)
    kb, kx, ky = jr.split(key, 3)

    b = jr.normal(kb, (d ** j,), dtype=jnp.float64)
    x = jr.normal(kx, (d ** n,), dtype=jnp.float64)
    y = jr.normal(ky, (d ** (n + j),), dtype=jnp.float64)

    lhs = np.dot(
        np.asarray(CORE.tensor_product_homogeneous(x, b)),
        np.asarray(y),
    )
    rhs = np.dot(
        np.asarray(x),
        np.asarray(CORE.tensor_adjoint_right_homogeneous(b, y)),
    )

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "i", "n"), [(2, 1, 2), (2, 2, 1), (3, 1, 2)])
def test_tensor_adjoint_left_homogeneous_on_basis_vectors_picks_first_block(d, i, n):
    """
    Basis-vector check: the left adjoint should slice along the first tensor block.
    """
    a = jnp.zeros((d ** i,), dtype=jnp.float64).at[1].set(1.0)
    y = jnp.arange(d ** (n + i), dtype=jnp.float64)

    out = CORE.tensor_adjoint_left_homogeneous(a, y)
    ref = np.asarray(y).reshape(d ** i, d ** n)[1]

    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "j", "n"), [(2, 1, 2), (2, 2, 1), (3, 1, 2)])
def test_tensor_adjoint_right_homogeneous_on_basis_vectors_picks_last_block(d, j, n):
    """
    Basis-vector check: the right adjoint should slice along the last tensor block.
    """
    b = jnp.zeros((d ** j,), dtype=jnp.float64).at[1].set(1.0)
    y = jnp.arange(d ** (n + j), dtype=jnp.float64)

    out = CORE.tensor_adjoint_right_homogeneous(b, y)
    ref = np.asarray(y).reshape(d ** n, d ** j)[:, 1]

    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12, atol=1e-12)


def _random_dense_elem(key, *, dim, trunc, batch_shape=(), scale=0.2):
    keys = jr.split(key, trunc + 1)
    return tuple(
        scale * jr.normal(keys[k], batch_shape + (dim**k,), dtype=jnp.float64)
        for k in range(trunc + 1)
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
def test_adjoint_product_inner_identity(side, dim, trunc):
    """
    Validate

        <W ⊗ X, Y> = <X, Adj_left(W, Y)>
        <X ⊗ W, Y> = <X, Adj_right(W, Y)>

    for dense graded elements without batch axes.
    """
    key = jr.PRNGKey(1000 + 100 * (side == "right") + 10 * dim + trunc)
    kw, kx, ky = jr.split(key, 3)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, scale=0.2)
    X = _random_dense_elem(kx, dim=dim, trunc=trunc, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, scale=0.2)

    Z = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    lhs = jnp.array(0.0, dtype=jnp.float64)
    for n in range(trunc + 1):
        for i in range(n + 1):
            if side == "left":
                term = CORE.tensor_product_homogeneous(W[i], X[n - i])
            else:
                term = CORE.tensor_product_homogeneous(X[n - i], W[i])

            lhs = lhs + CORE.tensor_inner_product((term,), (Y[n],))

    rhs = CORE.tensor_inner_product(X, Z)

    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
def test_adjoint_product_degree_formula(side, dim, trunc):
    """
    Compare tensor_adjoint_product against the literal degree formula

        Z_n = sum_i Adj(W_i, Y_{n+i}).
    """
    key = jr.PRNGKey(2000 + 100 * (side == "right") + 10 * dim + trunc)
    kw, ky = jr.split(key, 2)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, scale=0.2)

    out = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    manual = []
    for n in range(trunc + 1):
        z = None
        for i in range(trunc - n + 1):
            if side == "left":
                term = CORE.tensor_adjoint_left_homogeneous(W[i], Y[n + i])
            else:
                term = CORE.tensor_adjoint_right_homogeneous(W[i], Y[n + i])
            z = term if z is None else z + term
        manual.append(z)

    assert len(out) == trunc + 1
    for a, b in zip(out, manual):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
def test_adjoint_product_truncation(side, dim, trunc):
    """
    Reducing trunc should crop the highest output degree.
    """
    key = jr.PRNGKey(3000 + 100 * (side == "right") + 10 * dim + trunc)
    kw, ky = jr.split(key, 2)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, scale=0.2)

    full = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    cropped = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc - 1,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    assert len(full) == trunc + 1
    assert len(cropped) == trunc

    for a, b in zip(cropped, full[:-1]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
def test_adjoint_product_first_on_out(side, dim, trunc):
    """
    first_on_out=True should drop degree 0 and keep the higher degrees unchanged.
    """
    key = jr.PRNGKey(4000 + 100 * (side == "right") + 10 * dim + trunc)
    kw, ky = jr.split(key, 2)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, scale=0.2)

    dense = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    first_on = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=True,
    )

    assert len(dense) == trunc + 1
    assert len(first_on) == trunc

    for a, b in zip(first_on, dense[1:]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_adjoint_product_inner_identity_batched(side, dim, trunc, batch_shape):
    """
    Batched version of the adjoint inner-product identity for equal batch shapes.
    """
    key = jr.PRNGKey(
        5000 + 1000 * (side == "right") + 100 * dim + 10 * trunc + len(batch_shape)
    )
    kw, kx, ky = jr.split(key, 3)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)
    X = _random_dense_elem(kx, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)

    Z = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    lhs = jnp.zeros(batch_shape, dtype=jnp.float64)
    for n in range(trunc + 1):
        for i in range(n + 1):
            if side == "left":
                term = CORE.tensor_product_homogeneous(W[i], X[n - i])
            else:
                term = CORE.tensor_product_homogeneous(X[n - i], W[i])

            lhs = lhs + CORE.tensor_inner_product((term,), (Y[n],))

    rhs = CORE.tensor_inner_product(X, Z)

    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_adjoint_product_degree_formula_batched(side, dim, trunc, batch_shape):
    """
    Batched degree-by-degree formula check for equal batch shapes.
    """
    key = jr.PRNGKey(
        6000 + 1000 * (side == "right") + 100 * dim + 10 * trunc + len(batch_shape)
    )
    kw, ky = jr.split(key, 2)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)

    out = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    manual = []
    for n in range(trunc + 1):
        z = None
        for i in range(trunc - n + 1):
            if side == "left":
                term = CORE.tensor_adjoint_left_homogeneous(W[i], Y[n + i])
            else:
                term = CORE.tensor_adjoint_right_homogeneous(W[i], Y[n + i])
            z = term if z is None else z + term
        manual.append(z)

    assert len(out) == trunc + 1
    for a, b in zip(out, manual):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_adjoint_product_truncation_batched(side, dim, trunc, batch_shape):
    """
    Batched truncation check for equal batch shapes.
    """
    key = jr.PRNGKey(
        7000 + 1000 * (side == "right") + 100 * dim + 10 * trunc + len(batch_shape)
    )
    kw, ky = jr.split(key, 2)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)

    full = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    cropped = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc - 1,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    assert len(full) == trunc + 1
    assert len(cropped) == trunc

    for a, b in zip(cropped, full[:-1]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_adjoint_product_first_on_out_batched(side, dim, trunc, batch_shape):
    """
    Batched first_on_out check for equal batch shapes.
    """
    key = jr.PRNGKey(
        8000 + 1000 * (side == "right") + 100 * dim + 10 * trunc + len(batch_shape)
    )
    kw, ky = jr.split(key, 2)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, batch_shape=batch_shape, scale=0.2)

    dense = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    first_on = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=True,
    )

    assert len(dense) == trunc + 1
    assert len(first_on) == trunc

    for a, b in zip(first_on, dense[1:]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(("dim", "trunc"), [(2, 3), (3, 2)])
def test_adjoint_product_degree_formula_broadcast(side, dim, trunc):
    """
    Explicit broadcast check for tensor_adjoint_product itself:

        W batch  = (2, 1)
        Y batch  = (2, 3)
        out batch = (2, 3)

    This keeps the broadcast pattern simple and matches the kind of outer
    broadcasting used elsewhere.
    """
    key = jr.PRNGKey(9000 + 1000 * (side == "right") + 100 * dim + 10 * trunc)
    kw, ky = jr.split(key, 2)

    W = _random_dense_elem(kw, dim=dim, trunc=trunc, batch_shape=(2, 1), scale=0.2)
    Y = _random_dense_elem(ky, dim=dim, trunc=trunc, batch_shape=(2, 3), scale=0.2)

    out = CORE.tensor_adjoint_product(
        W,
        Y,
        trunc=trunc,
        side=side,
        w_first_on=False,
        y_first_on=False,
        first_on_out=False,
    )

    manual = []
    for n in range(trunc + 1):
        z = None
        for i in range(trunc - n + 1):
            if side == "left":
                term = CORE.tensor_adjoint_left_homogeneous(W[i], Y[n + i])
            else:
                term = CORE.tensor_adjoint_right_homogeneous(W[i], Y[n + i])
            z = term if z is None else z + term
        manual.append(z)

    assert len(out) == trunc + 1
    for n, (a, b) in enumerate(zip(out, manual)):
        assert a.shape == (2, 3, dim**n)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)