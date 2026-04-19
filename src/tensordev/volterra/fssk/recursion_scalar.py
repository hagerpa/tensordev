from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from tensordev.core.jax import Jax
from tensordev.volterra.fssk.coeffs import FSSKCoefficients


Array = jax.Array
DenseElem = tuple[Array, ...]


@partial(jax.jit, static_argnames=("core",))
def init_state(
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> DenseElem:
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
    DenseElem
        Zero state as a matrix-valued dense element.
    """
    del core

    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")

    batch_shape = coef.E.shape[:-2]
    return tuple(
        jnp.zeros(batch_shape + (1, 1, coef.R, coef.m ** r), dtype=coef.E.dtype)
        for r in range(coef.trunc + 1)
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

    y_elem: DenseElem = (
        jnp.zeros(y.shape[:-1] + (1,), dtype=y.dtype),
        y,
    )

    f: DenseElem = (psi[..., N - 1, :][..., None, :, None],)
    for n in range(N - 2, -1, -1):
        f = core.tensor_summation(
            (psi[..., n, :][..., None, :, None],),
            core.tensor_product(f, y_elem, trunc=N - 1),
            trunc=N - 1,
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
            G = core.tensor_summation(
                (phi[..., n, :, :][..., None, :, :, None],),
                core.tensor_product(G, y_elem, trunc=N - 1),
                trunc=N - 1,
            )

    return f, G


@partial(jax.jit, static_argnames=("core",))
def update_state(
        Z: DenseElem,
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> DenseElem:
    """
    Perform one scalar FSSK state update.

    This function assumes that ``coef`` is already step-local, so ``coef.E``,
    ``coef.psi`` and ``coef.phi`` carry only ordinary leading batch axes.
    Those batch axes may broadcast against the batch axes of ``y`` and ``Z``.

    The state is kept in the same representation as in the non-scalar module,
    namely with an explicit singleton family axis. If ``Z[r]`` has shape

        ``batch + (1, 1, R, m**r)``,

    then the updated state has the same shape.

    Parameters
    ----------
    Z : DenseElem
        Current state with homogeneous levels shaped

            ``batch + (1, 1, R, m**r)``,

        for ``r = 0, ..., trunc``.
    y : Array
        Projected path increment with shape ``batch + (m,)``.
    coef : FSSKCoefficients
        Scalar step-local FSSK coefficients. This function requires
        ``coef.q == 1``.
    core : Jax
        JAX core object.

    Returns
    -------
    DenseElem
        Updated state in the same representation as ``Z``.
    """
    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")
    if len(Z) != coef.trunc + 1:
        raise ValueError(
            f"Z must have {coef.trunc + 1} homogeneous levels, got {len(Z)}."
        )
    if y.shape[-1] != coef.m:
        raise ValueError(
            f"y must have trailing shape ({coef.m},), got {tuple(y.shape)}."
        )

    f, G = eval_fg(y, coef, core=core)

    ZE = core.tensor_matrix_product_right(
        Z,
        coef.E,
        trunc=coef.trunc,
    )

    ZG = core.tensor_matrix_product(
        Z,
        G,
        trunc=coef.trunc - 1,
    )

    B = core.tensor_summation(
        f,
        tuple(jnp.sum(level, axis=-4) for level in ZG),
        trunc=coef.trunc - 1,
    )

    y_elem: DenseElem = (
        jnp.zeros(y.shape[:-1] + (1,), dtype=y.dtype),
        y,
    )
    By = tuple(
        jnp.expand_dims(level, axis=-3)
        for level in core.tensor_product(B, y_elem, trunc=coef.trunc)
    )

    return core.tensor_summation(
        ZE,
        By,
        trunc=coef.trunc,
    )


__all__ = [
    "init_state",
    "eval_fg",
    "update_state",
]