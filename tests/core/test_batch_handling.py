from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from tensordev import Jax

CORE = Jax()


def _level_size(dim: int, degree: int) -> int:
    return dim ** degree


def _random_homogeneous_level(
    key,
    *,
    dim: int,
    degree: int,
    batch_shape=(),
    scale: float = 0.2,
    dtype=jnp.float64,
):
    return scale * jr.normal(
        key,
        shape=batch_shape + (_level_size(dim, degree),),
        dtype=dtype,
    )


def _manual_tensor_product_homogeneous(a, b, *, dim: int, deg_a: int, deg_b: int):
    """
    Reference implementation of the homogeneous tensor product that preserves
    arbitrary leading axes and only combines the final flattened tensor axes.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    batch_shape = a.shape[:-1]
    assert b.shape[:-1] == batch_shape

    a_shaped = a.reshape(batch_shape + (dim,) * deg_a + (1,) * deg_b)
    b_shaped = b.reshape(batch_shape + (1,) * deg_a + (dim,) * deg_b)

    out = a_shaped * b_shaped
    return out.reshape(batch_shape + (_level_size(dim, deg_a + deg_b),))


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize(
    ("dim", "deg_a", "deg_b"),
    [
        (2, 0, 0),
        (2, 0, 1),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
        (3, 1, 1),
        (3, 1, 2),
    ],
)
def test_tensor_product_homogeneous_matches_manual_reference(
    batch_shape,
    dim,
    deg_a,
    deg_b,
):
    """
    The homogeneous tensor product should preserve all leading axes and agree
    with the obvious outer-product construction on the trailing tensor axes.
    """
    key = jr.PRNGKey(
        1000
        + 100 * dim
        + 10 * deg_a
        + deg_b
        + len(batch_shape)
    )
    ka, kb = jr.split(key, 2)

    a = _random_homogeneous_level(
        ka,
        dim=dim,
        degree=deg_a,
        batch_shape=batch_shape,
    )
    b = _random_homogeneous_level(
        kb,
        dim=dim,
        degree=deg_b,
        batch_shape=batch_shape,
    )

    out = CORE.tensor_product_homogeneous(a, b)
    ref = _manual_tensor_product_homogeneous(
        a,
        b,
        dim=dim,
        deg_a=deg_a,
        deg_b=deg_b,
    )

    assert out.shape == batch_shape + (_level_size(dim, deg_a + deg_b),)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
@pytest.mark.parametrize(
    ("dim", "deg_a", "deg_b"),
    [
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
        (3, 1, 1),
        (3, 1, 2),
    ],
)
def test_tensor_product_homogeneous_matches_pointwise_application(
    batch_shape,
    dim,
    deg_a,
    deg_b,
):
    """
    Applying the homogeneous tensor product to a batched input should be the
    same as applying it independently to every batch entry.
    """
    key = jr.PRNGKey(
        2000
        + 100 * dim
        + 10 * deg_a
        + deg_b
        + 7 * len(batch_shape)
    )
    ka, kb = jr.split(key, 2)

    a = _random_homogeneous_level(
        ka,
        dim=dim,
        degree=deg_a,
        batch_shape=batch_shape,
    )
    b = _random_homogeneous_level(
        kb,
        dim=dim,
        degree=deg_b,
        batch_shape=batch_shape,
    )

    batched = np.asarray(CORE.tensor_product_homogeneous(a, b))

    pointwise = np.empty_like(batched)
    for idx in np.ndindex(batch_shape):
        pointwise[idx] = np.asarray(
            CORE.tensor_product_homogeneous(a[idx], b[idx])
        )

    np.testing.assert_allclose(batched, pointwise, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_tensor_product_homogeneous_is_bilinear_with_batch_axes(dim):
    """
    Bilinearity should hold entrywise even when arbitrary leading batch axes are
    present.
    """
    batch_shape = (2, 3)
    deg_a = 1
    deg_b = 2

    key = jr.PRNGKey(3000 + dim)
    ka1, ka2, kb1, kb2 = jr.split(key, 4)

    a1 = _random_homogeneous_level(ka1, dim=dim, degree=deg_a, batch_shape=batch_shape)
    a2 = _random_homogeneous_level(ka2, dim=dim, degree=deg_a, batch_shape=batch_shape)
    b1 = _random_homogeneous_level(kb1, dim=dim, degree=deg_b, batch_shape=batch_shape)
    b2 = _random_homogeneous_level(kb2, dim=dim, degree=deg_b, batch_shape=batch_shape)

    lhs_a = CORE.tensor_product_homogeneous(a1 + a2, b1)
    rhs_a = CORE.tensor_product_homogeneous(a1, b1) + CORE.tensor_product_homogeneous(a2, b1)

    lhs_b = CORE.tensor_product_homogeneous(a1, b1 + b2)
    rhs_b = CORE.tensor_product_homogeneous(a1, b1) + CORE.tensor_product_homogeneous(a1, b2)

    np.testing.assert_allclose(np.asarray(lhs_a), np.asarray(rhs_a), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(lhs_b), np.asarray(rhs_b), rtol=1e-12, atol=1e-12)

    def test_tensor_product_homogeneous_matches_pde_outer_product_broadcasting():
        """
        The PDE solver uses an outer-product broadcasting convention for pairwise
        batches:

            x : batch_x + (1,)*ny + (d_x,)
            y : (1,)*nx + batch_y + (d_y,)

        and expects the homogeneous tensor product to land in

            batch_x + batch_y + (d_x * d_y,).

        This test checks that convention directly, both without and with an
        additional shared leading batch axis.
        """
        # --- case 1: pure outer-product broadcasting
        x = jnp.array(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ],
            dtype=jnp.float64,
        )  # shape (2, 1, 2)

        y = jnp.array(
            [
                [[10.0, 20.0, 30.0],
                 [40.0, 50.0, 60.0],
                 [70.0, 80.0, 90.0]]
            ],
            dtype=jnp.float64,
        )  # shape (1, 3, 3)

        out = CORE.tensor_product_homogeneous(x, y)
        expected = np.einsum("...i,...j->...ij", np.asarray(x), np.asarray(y)).reshape(2, 3, 6)

        assert out.shape == (2, 3, 6)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12)

        # --- case 2: same convention, but with one shared leading batch axis
        x2 = jnp.array(
            [
                [[[1.0, 2.0]],
                 [[3.0, 4.0]]],
                [[[5.0, 6.0]],
                 [[7.0, 8.0]]],
            ],
            dtype=jnp.float64,
        )  # shape (2, 2, 1, 2)

        y2 = jnp.array(
            [
                [[[10.0, 20.0, 30.0],
                  [40.0, 50.0, 60.0],
                  [70.0, 80.0, 90.0]]],
                [[[15.0, 25.0, 35.0],
                  [45.0, 55.0, 65.0],
                  [75.0, 85.0, 95.0]]],
            ],
            dtype=jnp.float64,
        )  # shape (2, 1, 3, 3)

        out2 = CORE.tensor_product_homogeneous(x2, y2)
        expected2 = np.einsum("...i,...j->...ij", np.asarray(x2), np.asarray(y2)).reshape(2, 2, 3, 6)

        assert out2.shape == (2, 2, 3, 6)
        np.testing.assert_allclose(np.asarray(out2), expected2, rtol=1e-12, atol=1e-12)