from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax import lax

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem
from tensordev.volterra.coeffs import VolterraCoefficients
from tensordev.volterra.kernel import ConvolutionKernel
from tensordev.volterra.eval_scalar import eval_vte as eval_vte_scalar
from tensordev.volterra.eval_general import eval_vte as eval_vte_general

Array = jax.Array

_CORE = Jax()


def vsig(
        X: Array,
        *,
        kernel: ConvolutionKernel,
        trunc: int,
        dt: Array | float = 1.0,
        axis: int = -2,
        output_starting_point: bool = False,
        increment_input: bool = False,
        dyadic_order: int = 0,
        order: int = 0,
) -> DenseElem:
    r"""
    Compute a truncated Volterra signature from path nodes or increments.

    This is the general quadratic Volterra-Chen recursion under the coefficient
    symmetry hypothesis implemented by :class:`ConvolutionKernel`.  Unlike the SSS
    recursion there is no fixed-size hidden Markov state: internally we carry a
    padded history buffer ``[1, V_0, ..., V_{j-1}, 0, ...]`` so that the outer
    recursion is a :func:`jax.lax.scan` and each inner source-interval sum is a
    batched local ``eval_vte`` call.

    Parameters
    ----------
    X:
        Path nodes or increments.  The trailing axis is the path dimension
        ``kernel.path_dim``; ``axis`` is the step/node axis.  Set
        ``increment_input=True`` to skip :func:`jnp.diff`.
    kernel:
        Volterra kernel supplying projections and coefficient builders.
    trunc:
        Tensor truncation level (positive integer).
    dt:
        Step size(s).  A scalar gives a uniform grid; a 1-D array of length
        ``S`` gives a non-uniform grid via cumulative sums (default ``1.0``).
    axis:
        Step/node axis of ``X``.
    output_starting_point:
        If ``False`` (default), return the terminal Volterra signature.  If
        ``True``, return the whole trajectory with the tensor unit prepended,
        i.e. ``[1, V_0, ..., V_{S-1}]`` with the trajectory axis at ``axis``.
    increment_input:
        Treat ``X`` as increments rather than path nodes.
    dyadic_order:
        Non-negative integer.  Each original increment is split into
        ``2**dyadic_order`` equal sub-increments (each multiplied by
        ``1 / 2**dyadic_order``) and ``dt`` is refined accordingly.
        ``dyadic_order=0`` (default) leaves the path unchanged.
    order:
        Quadrature order for the higher-order basis-expansion scheme.
        ``0`` (default) left point approximation.  ``1`` uses the
        basis ``{1, s^beta, s}``; ``2`` uses ``{1, s^beta, s, s^(beta+1), s^2}``.
        Works on non-uniform grids; coefficients are computed inside the
        scan to keep memory O(S) rather than O(S²).

    Returns
    -------
    DenseElem
        Terminal signature by default.  With ``output_starting_point=True``,
        each level carries an additional trajectory axis at ``axis``.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}.")

    X = jnp.asarray(X)
    if X.ndim < 2:
        raise ValueError("X must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % X.ndim
    if axis_norm == X.ndim - 1:
        raise ValueError("axis must identify the step axis, not the trailing path dimension.")
    if X.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"X trailing dimension must be {kernel.path_dim}, got {X.shape[-1]}."
        )

    dtype = X.dtype
    dX = (X if increment_input else jnp.diff(X, axis=axis_norm)).astype(dtype)
    S = dX.shape[axis_norm]
    if S == 0:
        raise ValueError("volterra_vsig requires at least one increment.")

    # Dyadic refinement: split each increment into 2**dyadic_order equal sub-increments.
    if dyadic_order > 0:
        factor = 1 << int(dyadic_order)
        dX = jnp.repeat(dX / factor, factor, axis=axis_norm)
        S = dX.shape[axis_norm]
        dt = _refine_dt(dt, factor=factor, dtype=dtype)

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    y = projected[..., 0, :] if kernel.q == 1 else projected

    y_time = jnp.moveaxis(y, axis_norm, 0)
    y_time = _normalize_projected_y_time(y_time, kernel)
    times_arr = _normalize_times(dt, S=S, dtype=dtype)

    batch_shape = tuple(y_time.shape[1:-1]) if kernel.q == 1 else tuple(y_time.shape[1:-2])
    unit = _make_unit(batch_shape=batch_shape, m=kernel.m, trunc=trunc, dtype=dtype)
    source = jnp.arange(S, dtype=jnp.int32)

    # Unified interval-basis scan for all orders and all q.
    # For q > 1 the basis includes all beta_p values; for q=1 it reduces to
    # the scalar basis used by the FFT branch.
    rhos = _basis_rhos_multicomp(order, betas=kernel.beta.astype(dtype), dtype=dtype)
    thetas = _chebyshev_lobatto_thetas(n=len(rhos), dtype=dtype)
    B = len(rhos)
    components0 = _make_basis_seed(
        B=B, S=S, batch_shape=batch_shape, m=kernel.m, trunc=trunc, dtype=dtype,
    )
    source_indices = jnp.arange(S, dtype=jnp.int32)
    h_all = times_arr[1:] - times_arr[:-1]
    interp_inv_all = jnp.linalg.inv(
        _basis_interpolation_matrix_batched(h_all, thetas=thetas, rhos=rhos, dtype=dtype)
    )

    def step_basis(
            components: tuple,
            j: Array,
    ) -> tuple[tuple, DenseElem]:
        t_j = times_arr[j]
        t_jp1 = times_arr[j + 1]
        h_j = t_jp1 - t_j

        past_mask = source_indices < j

        evals: list[DenseElem] = [
            _basis_readout(
                components,
                tau=t_j + thetas[k] * h_j,
                source_mask=past_mask,
                kernel=kernel,
                times_arr=times_arr,
                y_time=y_time,
                rhos=rhos,
                unit=unit,
                batch_shape=batch_shape,
                trunc=trunc,
                dtype=dtype,
            )
            for k in range(B)
        ]

        interp_inv = interp_inv_all[j]
        components_next = components
        for n in range(1, trunc + 1):
            evals_n = jnp.stack([evals[k][n] for k in range(B)], axis=0)
            new_vals_n = jnp.einsum("bk,k...->b...", interp_inv, evals_n)
            components_next = tuple(
                tuple(
                    (level.at[j].set(new_vals_n[b]) if lvl_n == n else level)
                    for lvl_n, level in enumerate(components_next[b])
                )
                for b in range(B)
            )

        current_mask = source_indices <= j
        V_jp1 = _basis_readout(
            components_next,
            tau=t_jp1,
            source_mask=current_mask,
            kernel=kernel,
            times_arr=times_arr,
            y_time=y_time,
            rhos=rhos,
            unit=unit,
            batch_shape=batch_shape,
            trunc=trunc,
            dtype=dtype,
        )

        return components_next, V_jp1

    _, traj = lax.scan(step_basis, components0, source)

    if output_starting_point:
        traj_with_unit = tuple(
            jnp.concatenate([unit[n][None], traj[n]], axis=0)
            for n in range(trunc + 1)
        )
        if axis_norm != 0:
            traj_with_unit = tuple(
                jnp.moveaxis(level, 0, axis_norm) for level in traj_with_unit
            )
        return traj_with_unit

    return tuple(level[-1] for level in traj)


def _normalize_projected_y_time(y_time: Array, kernel: ConvolutionKernel) -> Array:
    """Validate projected increments after moving the source axis to zero."""
    if kernel.q == 1:
        if y_time.shape[-1] == kernel.m:
            if y_time.ndim >= 3 and y_time.shape[-2:] == (1, kernel.m):
                return y_time[..., 0, :]
            return y_time
        raise ValueError(
            f"For q=1, y must have trailing shape ({kernel.m},) or (1, {kernel.m}); "
            f"got shape {y_time.shape}."
        )

    if y_time.ndim < 3 or y_time.shape[-2:] != (kernel.q, kernel.m):
        raise ValueError(
            f"For q>1, y must have trailing shape ({kernel.q}, {kernel.m}); "
            f"got shape {y_time.shape}."
        )
    return y_time


def _refine_dt(
        dt: Array | float,
        *,
        factor: int,
        dtype: jnp.dtype,
) -> Array | float:
    """Subdivide ``dt`` for dyadic refinement by ``factor``.

    Scalar ``dt`` divides by ``factor``.  A per-step array produces a refined
    array repeated ``factor`` times per original step.
    """
    dt_arr = jnp.asarray(dt, dtype=dtype)
    if dt_arr.ndim == 0:
        return dt_arr / factor
    return jnp.repeat(dt_arr / factor, factor)


def _normalize_times(
        dt: Array | float,
        *,
        S: int,
        dtype: jnp.dtype,
) -> Array:
    """Return node times of shape ``(S + 1,)`` from a scalar or per-step ``dt``.

    - Scalar ``dt``: uniform grid ``[0, dt, 2·dt, …, S·dt]``.
    - 1-D array of length ``S``: ``[0, dt[0], dt[0]+dt[1], …]``.
    """
    dt_arr = jnp.asarray(dt, dtype=dtype)
    if dt_arr.ndim == 0:
        return jnp.arange(S + 1, dtype=dtype) * dt_arr
    if dt_arr.ndim == 1 and dt_arr.shape[0] == S:
        return jnp.concatenate([jnp.zeros((1,), dtype=dtype), jnp.cumsum(dt_arr)])
    raise ValueError(
        f"dt must be a scalar or a 1-D array of length S={S}, got shape {dt_arr.shape}."
    )


def _make_unit(
        *,
        batch_shape: tuple[int, ...],
        m: int,
        trunc: int,
        dtype: jnp.dtype,
) -> DenseElem:
    """Tensor unit with zero positive levels."""
    return (
        jnp.ones(batch_shape + (1,), dtype=dtype),
        *(
            jnp.zeros(batch_shape + (m ** n,), dtype=dtype)
            for n in range(1, trunc + 1)
        ),
    )




def _make_basis_seed(
        *,
        B: int,
        S: int,
        batch_shape: tuple[int, ...],
        m: int,
        trunc: int,
        dtype: jnp.dtype,
) -> tuple:
    """Initial interval basis coefficients for the higher-order scan.

    Returns B DenseElems of shape ``(S, *batch_shape, m**n)`` per level n.
    Level-0 of component b=0 is all-ones; everything else is zero.
    """
    components = []
    for b in range(B):
        levels = []
        for n in range(trunc + 1):
            shape = (S,) + batch_shape + (m ** n,)
            if n == 0 and b == 0:
                levels.append(jnp.ones(shape, dtype=dtype))
            else:
                levels.append(jnp.zeros(shape, dtype=dtype))
        components.append(tuple(levels))
    return tuple(components)


def _basis_readout(
        components: tuple,
        *,
        tau: Array,
        source_mask: Array,
        kernel: ConvolutionKernel,
        times_arr: Array,
        y_time: Array,
        rhos: tuple,
        unit: DenseElem,
        batch_shape: tuple,
        trunc: int,
        dtype: jnp.dtype,
) -> DenseElem:
    """Evaluate V at time ``tau`` from interval basis coefficients with masking.

    ``source_mask[i]`` is True for source intervals i that should contribute.
    """
    contributions: list[DenseElem] = []
    for b, rho in enumerate(rhos):
        coef = kernel.coef(
            times_arr[:-1], times_arr[1:], tau,
            trunc=trunc, rho=rho, dtype=dtype,
        )
        coef = _insert_singleton_batch_axes(coef, batch_ndim=len(batch_shape))
        terms = (
            eval_vte_scalar(components[b], y_time, coef)
            if kernel.q == 1
            else eval_vte_general(components[b], y_time, coef)
        )
        mask = source_mask.reshape(source_mask.shape + (1,) * (terms[0].ndim - 1))
        terms = tuple(jnp.where(mask, level, jnp.zeros_like(level)) for level in terms)
        contributions.append(tuple(jnp.sum(level, axis=0) for level in terms))

    acc = tuple(
        jnp.sum(jnp.stack([c[n] for c in contributions], axis=0), axis=0)
        for n in range(trunc + 1)
    )
    return _CORE.tensor_summation(unit, acc, trunc=trunc)


def _insert_singleton_batch_axes(
        coef: VolterraCoefficients,
        *,
        batch_ndim: int,
) -> VolterraCoefficients:
    """Make a source-row coefficient broadcast over path batch axes."""
    if batch_ndim <= 0:
        return coef
    leading = coef.leading_shape + (1,) * int(batch_ndim)
    return replace(
        coef,
        alpha=coef.alpha.reshape(leading + coef.alpha.shape[-2:]),
        valid=coef.valid.reshape(leading),
    )


@dataclass(frozen=True, slots=True)
class BasisExpansionSpec:
    """Basis/interpolation specification for higher-order schemes.

    ``rhos`` and ``thetas`` have the same length ``B = 2*order + 1``.
    The constant basis element ``s^0 = 1`` is always ``rhos[0] = 0``.
    """

    rhos: tuple[Array, ...]
    thetas: tuple[Array, ...]
    interpolation_inverse: Array


def _basis_rhos(
        order: int,
        *,
        beta: Array,
        dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Basis exponents for the fractional higher-order scheme of the given order.

    order=0: ``{1}``                              → ``(0,)``
    order=1: ``{1, s^beta, s}``                   → ``(0, beta, 1)``
    order=2: ``{1, s^beta, s, s^(beta+1), s^2}``  → ``(0, beta, 1, beta+1, 2)``
    """
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    if order == 0:
        return (zero,)
    if order == 1:
        return (zero, beta, one)
    if order == 2:
        return (zero, beta, one, beta + one, jnp.asarray(2.0, dtype=dtype))
    raise NotImplementedError(
        f"Basis-expansion scheme order={order} is not implemented yet."
    )


def _basis_rhos_multicomp(
        order: int,
        *,
        betas: Array,
        dtype: jnp.dtype,
) -> tuple:
    """Basis exponents for the higher-order scheme with q >= 1 kernel components.

    Each ``beta_p`` in ``betas`` contributes its own fractional exponent(s) to
    the basis so that the interpolant spans all singularity types present in K.
    For q=1 this reduces to :func:`_basis_rhos`.

    order=0: ``(0,)``                              — B = 1
    order=1: ``(0, beta_1, ..., beta_q, 1)``       — B = q + 2
    order=2: additionally adds ``(beta_p+1, 2)``   — B = 2q + 3
    """
    betas_1d = jnp.atleast_1d(betas).astype(dtype)
    q = int(betas_1d.shape[0])
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    if order == 0:
        return (zero,)
    beta_rhos = tuple(betas_1d[p] for p in range(q))
    if order == 1:
        return (zero,) + beta_rhos + (one,)
    if order == 2:
        return (zero,) + beta_rhos + (one,) + tuple(b + one for b in beta_rhos) + (jnp.asarray(2.0, dtype=dtype),)
    raise NotImplementedError(
        f"Basis-expansion scheme order={order} is not implemented yet."
    )


def _chebyshev_lobatto_thetas(
        *,
        n: int,
        dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Chebyshev-Lobatto nodes on ``[0, 1]``.

    The nodes include both endpoints.  For ``n=1`` the single node is ``0``.
    For ``n=5`` this gives ``0, 1/2 - sqrt(2)/4, 1/2, 1/2 + sqrt(2)/4, 1``.
    """
    if n < 1:
        raise ValueError(f"Need at least one interpolation node, got n={n}.")
    if n == 1:
        return (jnp.asarray(0.0, dtype=dtype),)

    k = jnp.arange(n, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    nodes = (one - jnp.cos(jnp.pi * k / jnp.asarray(n - 1, dtype=dtype))) / jnp.asarray(2.0, dtype=dtype)

    return tuple(nodes[i] for i in range(n))


def _basis_interpolation_matrix(
        *,
        h: Array,
        thetas: tuple[Array, ...],
        rhos: tuple[Array, ...],
        dtype: jnp.dtype,
) -> Array:
    """Matrix mapping basis coefficients to point evaluations.

    Entry ``V[a, b]`` is the value of basis component ``b`` at interpolation
    point ``theta[a]``.  ``h`` must be a scalar.
    """
    rows = []
    for theta in thetas:
        row = []
        for rho in rhos:
            is_const = rho == jnp.asarray(0.0, dtype=dtype)
            value = jnp.where(
                is_const,
                jnp.asarray(1.0, dtype=dtype),
                (theta * h) ** rho,
            )
            row.append(value)
        rows.append(jnp.stack(row, axis=0))
    return jnp.stack(rows, axis=0)


def _basis_interpolation_matrix_batched(
        h_all: Array,
        *,
        thetas: tuple[Array, ...],
        rhos: tuple[Array, ...],
        dtype: jnp.dtype,
) -> Array:
    """Vectorised version of :func:`_basis_interpolation_matrix` over a grid.

    Parameters
    ----------
    h_all:
        Per-step sizes, shape ``(S,)``.

    Returns
    -------
    Array
        Shape ``(S, B, B)`` — one interpolation matrix per step, suitable for
        a single batched :func:`jnp.linalg.inv` call.
    """
    rows = []
    for theta in thetas:
        row = []
        for rho in rhos:
            is_const = rho == jnp.asarray(0.0, dtype=dtype)
            value = jnp.where(
                is_const,
                jnp.ones_like(h_all),
                (theta * h_all) ** rho,
            )
            row.append(value)
        rows.append(jnp.stack(row, axis=0))
    return jnp.moveaxis(jnp.stack(rows, axis=0), -1, 0)


__all__ = ["vsig", "BasisExpansionSpec"]
