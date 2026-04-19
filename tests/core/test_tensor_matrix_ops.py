import numpy as np
import pytest
import jax.numpy as jnp

from tensordev.core.jax import Jax


def _as_numpy(x):
    return np.asarray(x)


def _assert_allclose(x, y, atol=1e-6, rtol=1e-6):
    np.testing.assert_allclose(_as_numpy(x), _as_numpy(y), atol=atol, rtol=rtol)


def _assert_elem_allclose(x, y, atol=1e-6, rtol=1e-6):
    assert len(x) == len(y)
    for a, b in zip(x, y):
        _assert_allclose(a, b, atol=atol, rtol=rtol)


def _canonicalize_matrix_axes(x, row_axis, col_axis):
    row_axis = row_axis % x.ndim
    col_axis = col_axis % x.ndim
    if row_axis == col_axis:
        raise ValueError("row_axis and col_axis must be distinct.")
    return np.moveaxis(x, (row_axis, col_axis), (-3, -2)), row_axis, col_axis


def _restore_matrix_axes(x, row_axis, col_axis):
    return np.moveaxis(x, (-3, -2), (row_axis, col_axis))


def _ref_right_homogeneous(a, m, row_axis=-3, col_axis=-2):
    ac, row_axis, col_axis = _canonicalize_matrix_axes(np.asarray(a), row_axis, col_axis)
    out = np.einsum("...nka,...kl->...nla", ac, np.asarray(m))
    return _restore_matrix_axes(out, row_axis, col_axis)


def _ref_left_homogeneous(m, a, row_axis=-3, col_axis=-2):
    ac, row_axis, col_axis = _canonicalize_matrix_axes(np.asarray(a), row_axis, col_axis)
    out = np.einsum("...nk,...kla->...nla", np.asarray(m), ac)
    return _restore_matrix_axes(out, row_axis, col_axis)


def _ref_product_homogeneous(a, b, row_axis=-3, col_axis=-2):
    ac, row_axis, col_axis = _canonicalize_matrix_axes(np.asarray(a), row_axis, col_axis)
    bc, _, _ = _canonicalize_matrix_axes(np.asarray(b), row_axis, col_axis)
    out = np.einsum("...nka,...klb->...nlab", ac, bc)
    out = out.reshape(out.shape[:-2] + (out.shape[-2] * out.shape[-1],))
    return _restore_matrix_axes(out, row_axis, col_axis)


def _ref_right(a, m, trunc=None, row_axis=-3, col_axis=-2):
    n = len(a) - 1
    if trunc is not None:
        n = min(n, trunc)
    return tuple(
        _ref_right_homogeneous(a[r], m, row_axis=row_axis, col_axis=col_axis)
        for r in range(n + 1)
    )


def _ref_left(m, a, trunc=None, row_axis=-3, col_axis=-2):
    n = len(a) - 1
    if trunc is not None:
        n = min(n, trunc)
    return tuple(
        _ref_left_homogeneous(m, a[r], row_axis=row_axis, col_axis=col_axis)
        for r in range(n + 1)
    )


def _ref_product(a, b, trunc=None, row_axis=-3, col_axis=-2):
    n = len(a) + len(b) - 2
    if trunc is not None:
        n = min(n, trunc)

    out = []
    for r in range(n + 1):
        i_min = max(0, r - (len(b) - 1))
        i_max = min(len(a) - 1, r)
        term = _ref_product_homogeneous(
            a[i_min], b[r - i_min], row_axis=row_axis, col_axis=col_axis
        )
        for i in range(i_min + 1, i_max + 1):
            term = term + _ref_product_homogeneous(
                a[i], b[r - i], row_axis=row_axis, col_axis=col_axis
            )
        out.append(term)
    return tuple(out)


def _make_elem_with_time(trunc, t, n, k, d, batch=(), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for r in range(trunc + 1):
        dr = d ** r
        shape = batch + (t, n, k, dr)
        out.append(jnp.asarray(rng.normal(size=shape)))
    return tuple(out)


def _make_elem_custom_axes(trunc, batch, n, t, k, d, seed=1):
    """
    Levels have shape batch + (n, T, k, d_r), so row_axis=-4, col_axis=-2.
    """
    rng = np.random.default_rng(seed)
    out = []
    for r in range(trunc + 1):
        dr = d ** r
        shape = batch + (n, t, k, dr)
        out.append(jnp.asarray(rng.normal(size=shape)))
    return tuple(out)


@pytest.fixture(scope="module")
def core():
    return Jax()


def test_tensor_matrix_product_right_homogeneous_default_axes(core):
    rng = np.random.default_rng(2)
    a = jnp.asarray(rng.normal(size=(2, 5, 3, 4, 7)))   # batch=2, T=5, n=3, k=4, d_r=7
    m = jnp.asarray(rng.normal(size=(2, 5, 4, 6)))      # batch/time aware right matrix

    got = core.tensor_matrix_product_right_homogeneous(a, m)
    want = _ref_right_homogeneous(a, m)

    _assert_allclose(got, want)


def test_tensor_matrix_product_left_homogeneous_default_axes(core):
    rng = np.random.default_rng(3)
    m = jnp.asarray(rng.normal(size=(2, 5, 6, 4)))      # batch/time aware left matrix
    a = jnp.asarray(rng.normal(size=(2, 5, 4, 3, 8)))   # batch=2, T=5, k=4, l=3, d_r=8

    got = core.tensor_matrix_product_left_homogeneous(m, a)
    want = _ref_left_homogeneous(m, a)

    _assert_allclose(got, want)


def test_tensor_matrix_product_homogeneous_default_axes(core):
    rng = np.random.default_rng(4)
    a = jnp.asarray(rng.normal(size=(2, 5, 3, 4, 7)))   # batch=2, T=5, n=3, k=4, d_i=7
    b = jnp.asarray(rng.normal(size=(2, 5, 4, 6, 9)))   # batch=2, T=5, k=4, l=6, d_j=9

    got = core.tensor_matrix_product_homogeneous(a, b)
    want = _ref_product_homogeneous(a, b)

    _assert_allclose(got, want)


def test_tensor_matrix_product_homogeneous_custom_axes(core):
    rng = np.random.default_rng(5)
    a = jnp.asarray(rng.normal(size=(2, 3, 5, 4, 7)))   # batch=2, n=3, T=5, k=4, d_i=7
    b = jnp.asarray(rng.normal(size=(2, 4, 5, 6, 9)))   # batch=2, k=4, T=5, l=6, d_j=9

    got = core.tensor_matrix_product_homogeneous(a, b, row_axis=-4, col_axis=-2)
    want = _ref_product_homogeneous(a, b, row_axis=-4, col_axis=-2)

    _assert_allclose(got, want)


def test_tensor_matrix_product_right_full(core):
    a = _make_elem_with_time(trunc=3, t=5, n=3, k=4, d=2, batch=(2,), seed=6)
    rng = np.random.default_rng(6)
    m = jnp.asarray(rng.normal(size=(2, 5, 4, 6)))

    got = core.tensor_matrix_product_right(a, m)
    want = _ref_right(a, m)

    _assert_elem_allclose(got, want)


def test_tensor_matrix_product_left_full(core):
    a = _make_elem_with_time(trunc=3, t=5, n=4, k=3, d=2, batch=(2,), seed=7)
    rng = np.random.default_rng(7)
    m = jnp.asarray(rng.normal(size=(2, 5, 6, 4)))

    got = core.tensor_matrix_product_left(m, a)
    want = _ref_left(m, a)

    _assert_elem_allclose(got, want)


def test_tensor_matrix_product_full_with_trunc(core):
    rng = np.random.default_rng(8)

    a = tuple(
        jnp.asarray(rng.normal(size=(2, 5, 3, 4, 2 ** r)))
        for r in range(4)
    )
    b = tuple(
        jnp.asarray(rng.normal(size=(2, 5, 4, 6, 2 ** r)))
        for r in range(3)
    )

    got = core.tensor_matrix_product(a, b, trunc=3)
    want = _ref_product(a, b, trunc=3)

    _assert_elem_allclose(got, want)


def test_tensor_matrix_product_full_custom_axes(core):
    a = _make_elem_custom_axes(trunc=3, batch=(2,), n=3, t=5, k=4, d=2, seed=9)
    rng = np.random.default_rng(9)
    b = tuple(
        jnp.asarray(rng.normal(size=(2, 4, 5, 6, 2 ** r)))
        for r in range(4)
    )  # batch + (k, T, l, d_r), row_axis=-4, col_axis=-2

    got = core.tensor_matrix_product(a, b, row_axis=-4, col_axis=-2)
    want = _ref_product(a, b, row_axis=-4, col_axis=-2)

    _assert_elem_allclose(got, want)