from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem, DenseElemFirstOn
from tensordev.sss.coeffs import FSSKCoefficients


Array = jax.Array


@partial(jax.jit, static_argnames=("core",))
def init_state(
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> DenseElemFirstOn:
    """
    Initialize the zero scalar FSSK state for a single-step coefficient object.

    This module assumes that ``coef`` is already step-local. Its leading axes,
    if any, are treated as ordinary batch axes.

    The scalar recursion keeps the explicit singleton family axis, so each
    homogeneous level has shape

        ``batch + (1, 1, R, m**r)``,

    for ``r = 0, ..., trunc``.

    Parameters
    ----------
    coef : FSSKCoefficients
        Scalar step-local FSSK coefficients. This function requires
        ``coef.q == 1``.
    core : Jax
        JAX core object. It is accepted for interface consistency with the
        non-scalar recursion module, but is not used directly here.

    Returns
    -------
    DenseElemFirstOn
        Zero state as a matrix-valued dense element.
    """
    del core

    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")

    # First-on format: trunc levels, index r = degree r+1 (degree-0 is always zero).
    batch_shape = coef.E.shape[:-2]
    return tuple(
        jnp.zeros(batch_shape + (1, 1, coef.R, coef.m ** (r + 1)), dtype=coef.E.dtype)
        for r in range(coef.trunc)
    )


@partial(jax.jit, static_argnames=("core",))
def eval_fg(
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> tuple[DenseElem, DenseElem]:
    """
    Evaluate the scalar Horner recursions for ``f`` and ``G``.

    This function assumes that ``coef`` is already step-local, so:

    - ``coef.E`` has shape ``batch + (R, R)``,
    - ``coef.psi`` has shape ``batch + (trunc, R)``,
    - ``coef.phi`` has shape ``batch + (1, trunc - 1, R, R)``.

    The input ``y`` may have any batch shape broadcastable against the leading
    batch shape of the coefficients.

    The returned dense elements are:

    - ``f`` in ``(mathfrak R_{trunc-1})^{1 x R}``,
    - ``G`` in ``((mathfrak R_{trunc-1})^{R x R})^1``.

    In particular, both outputs are returned only up to degree ``trunc - 1``.

    Parameters
    ----------
    y : Array
        Projected path increment with shape ``batch + (m,)``.
    coef : FSSKCoefficients
        Scalar step-local FSSK coefficients. This function requires
        ``coef.q == 1``.
    core : Jax
        JAX core object.

    Returns
    -------
    f : DenseElem
        Dense element with homogeneous levels shaped

            ``batch + (1, R, m**r)``,

        for ``r = 0, ..., trunc - 1``.

    G : DenseElem
        Dense element with homogeneous levels shaped

            ``batch + (1, R, R, m**r)``,

        for ``r = 0, ..., trunc - 1``. The leading singleton axis is the
        explicit family axis.
    """
    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")
    if y.shape[-1] != coef.m:
        raise ValueError(
            f"y must have trailing shape ({coef.m},), got {tuple(y.shape)}."
        )
    if coef.psi.shape[-2:] != (coef.trunc, coef.R):
        raise ValueError(
            "Scalar step-local coefficients must satisfy "
            f"psi.shape[-2:] == ({coef.trunc}, {coef.R}), got {coef.psi.shape[-2:]}."
        )
    if coef.trunc > 1 and coef.phi.shape[-4:] != (1, coef.trunc - 1, coef.R, coef.R):
        raise ValueError(
            "Scalar step-local coefficients must satisfy "
            f"phi.shape[-4:] == (1, {coef.trunc - 1}, {coef.R}, {coef.R}), "
            f"got {coef.phi.shape[-4:]}."
        )

    N = coef.trunc
    psi = coef.psi
    phi = coef.phi[..., 0, :, :, :] if N > 1 else None

    f: DenseElem = (psi[..., N - 1, :][..., None, :, None],)
    for n in range(N - 2, -1, -1):
        f = (
            psi[..., n, :][..., None, :, None],
        ) + core.tensor_product(
            f,
            (y[..., None, None, :],),
            trunc=N - 1,
            b_first_on=True,
        )

    if N == 1:
        G = (
            jnp.zeros(
                jnp.broadcast_shapes(coef.E.shape[:-2], y.shape[:-1]) + (1, coef.R, coef.R, 1),
                dtype=coef.E.dtype,
            ),
        )
    else:
        G: DenseElem = (phi[..., N - 2, :, :][..., None, :, :, None],)
        for n in range(N - 3, -1, -1):
            G = (
                phi[..., n, :, :][..., None, :, :, None],
            ) + core.tensor_product(
                G,
                (y[..., None, None, None, :],),
                trunc=N - 1,
                b_first_on=True,
            )

    return f, G


@partial(jax.jit, static_argnames=("core",))
def update_state(
        Z: DenseElemFirstOn,
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> DenseElemFirstOn:
    """
    Perform one scalar FSSK state update.

    This function assumes that ``coef`` is already step-local, so ``coef.E``,
    ``coef.psi`` and ``coef.phi`` carry only ordinary leading batch axes.
    Those batch axes may broadcast against the batch axes of ``y`` and ``Z``.

    The state is stored in **first-on format**: the degree-0 level is always
    identically zero and is not stored.  ``Z[r]`` carries degree ``r+1``, so
    each homogeneous level has shape

        ``batch + (1, 1, R, m**(r+1))``,

    for ``r = 0, ..., trunc - 1``.

    Parameters
    ----------
    Z : DenseElemFirstOn
        First-on state with ``trunc`` homogeneous levels:
        level ``r`` has shape ``batch + (1, 1, R, m**(r+1))``.
    y : Array
        Projected path increment with shape ``batch + (m,)``.
    coef : FSSKCoefficients
        Scalar step-local FSSK coefficients. This function requires
        ``coef.q == 1``.
    core : Jax
        JAX core object.

    Returns
    -------
    DenseElemFirstOn
        Updated state in the same representation as ``Z``.
    """
    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")
    if len(Z) != coef.trunc:
        raise ValueError(
            f"Z must have {coef.trunc} homogeneous levels (first-on), got {len(Z)}."
        )
    if y.shape[-1] != coef.m:
        raise ValueError(
            f"y must have trailing shape ({coef.m},), got {tuple(y.shape)}."
        )

    f, G = eval_fg(y, coef, core=core)

    # Lift first-on Z to dense by prepending the degree-0 zero level.
    # Degree-0 is always zero, so this adds no information and costs no compute
    # beyond one zero allocation that XLA can eliminate.
    zero_level = jnp.zeros(Z[0].shape[:-1] + (1,), dtype=Z[0].dtype)
    Z_dense = (zero_level,) + tuple(Z)

    ZE = core.tensor_matrix_product_right(
        Z_dense,
        coef.E[..., None, :, :],
        trunc=coef.trunc,
    )

    ZG = core.tensor_matrix_product(
        Z_dense,
        G,
        trunc=coef.trunc - 1,
    )

    B = core.tensor_summation(
        f,
        tuple(jnp.sum(level, axis=-4) for level in ZG),
        trunc=coef.trunc - 1,
    )

    By = (jnp.zeros_like(Z_dense[0]),) + tuple(
        jnp.expand_dims(level, axis=-3)
        for level in core.tensor_product(
            B,
            (y[..., None, None, :],),
            trunc=coef.trunc,
            b_first_on=True,
        )
    )

    result_dense = core.tensor_summation(ZE, By, trunc=coef.trunc)
    # Return first-on: the degree-0 level is zero by construction, drop it.
    return result_dense[1:]


__all__ = [
    "init_state",
    "eval_fg",
    "update_state",
]
