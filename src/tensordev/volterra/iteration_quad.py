from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem
from tensordev.volterra.coeffs import VolterraCoefficients
from tensordev.volterra.kernel import ConvolutionKernel
from tensordev.volterra.eval_scalar import eval_vte as eval_vte_scalar
from tensordev.volterra.eval_general import eval_vte as eval_vte_general

Array = jax.Array

_CORE = Jax()


def quadratic_iteration(
        dX: Array,
        *,
        kernel: ConvolutionKernel,
        trunc: int,
        dt: Array | float = 1.0,
        axis: int = -2,
        return_trajectory: bool = False,
        order: int = 0,
) -> DenseElem:
    r"""
    Quadratic Volterra-Chen recursion on pre-processed increments.

    Expects ``dX`` to already be in increment form (no differencing) and on
    the final time grid (dyadic refinement applied externally if needed).
    All preprocessing is the responsibility of the caller — typically the
    high-level :func:`~tensordev.volterra.signature.vsig`.

    Parameters
    ----------
    dX:
        Increments.  Shape ``(*batch, S, d)`` with step axis at ``axis``
        and trailing path dimension ``d = kernel.path_dim``.
    kernel:
        Volterra kernel supplying projections and coefficient builders.
    trunc:
        Tensor truncation level (positive integer).
    dt:
        Step size(s).  Scalar → uniform grid; 1-D array of length ``S``
        → non-uniform grid (default ``1.0``).
    axis:
        Step axis of ``dX`` (default ``-2``).
    return_trajectory:
        If ``True``, return ``[V_1, ..., V_S]`` with the step axis at
        ``axis``.  If ``False`` (default), return the terminal ``V_S``.
    order:
        Quadrature order for the basis-expansion scheme.  ``0`` (default)
        left-point; ``1`` uses ``{1, s^beta, s}``; ``2`` uses
        ``{1, s^beta, s, s^(beta+1), s^2}``.

    Returns
    -------
    DenseElem
        Terminal signature, or full trajectory when ``return_trajectory=True``.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}.")

    dX = jnp.asarray(dX)
    if dX.ndim < 2:
        raise ValueError("dX must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % dX.ndim
    if axis_norm == dX.ndim - 1:
        raise ValueError("axis must identify the step axis, not the trailing path dimension.")
    if dX.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"dX trailing dimension must be {kernel.path_dim}, got {dX.shape[-1]}."
        )

    dtype = dX.dtype
    S = dX.shape[axis_norm]
    if S == 0:
        raise ValueError("quadratic_iteration requires at least one increment.")

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX.astype(dtype))
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
    rhos = _basis_rhos_multicomp(order, betas=kernel.beta, dtype=dtype)
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

    if return_trajectory:
        if axis_norm != 0:
            traj = tuple(jnp.moveaxis(level, 0, axis_norm) for level in traj)
        return traj


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

    Duplicate exponents (e.g. when ``beta == 1``) are removed so that the
    interpolation matrix remains non-singular.  The returned tuple is sorted
    in ascending order.

    order=0: ``{0}``
    order=1: ``{0} | {beta} | {1}``
    order=2: ``{0} | {beta} | {1} | {beta+1} | {2}``
    """
    zero = jnp.asarray(0.0, dtype=dtype)
    if order == 0:
        return (zero,)

    b = float(jnp.asarray(beta, dtype=dtype))
    candidates: list[float] = [0.0, b, 1.0]
    if order == 2:
        candidates += [b + 1.0, 2.0]
    elif order != 1:
        raise NotImplementedError(
            f"Basis-expansion scheme order={order} is not implemented yet."
        )

    seen: set[float] = set()
    unique: list[float] = []
    for v in candidates:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    unique.sort()

    return tuple(jnp.asarray(v, dtype=dtype) for v in unique)


def _basis_rhos_multicomp(
        order: int,
        *,
        betas: Array,
        dtype: jnp.dtype,
) -> tuple:
    """Basis exponents for the higher-order scheme with q >= 1 kernel components.

    Each ``beta_p`` in ``betas`` contributes its own fractional exponent(s) to
    the basis so that the interpolant spans all singularity types present in K.
    Duplicate exponents (e.g. when ``beta_p == 1``) are removed so that the
    interpolation matrix remains non-singular.  The returned tuple is sorted in
    ascending order.

    order=0: ``{0}``
    order=1: ``{0} | {beta_p} | {1}``
    order=2: ``{0} | {beta_p} | {1} | {beta_p + 1} | {2}``
    """
    # Use numpy to extract concrete float values — jnp indexing inside a jit
    # trace produces abstract tracers, which cannot be converted to Python float.
    betas_np = np.asarray(betas).reshape(-1).astype(float)
    q = int(betas_np.shape[0])
    zero = jnp.asarray(0.0, dtype=dtype)
    if order == 0:
        return (zero,)

    # Collect candidates as Python floats so exact deduplication via a set works.
    beta_vals = [float(betas_np[p]) for p in range(q)]
    candidates: list[float] = [0.0] + beta_vals + [1.0]
    if order == 2:
        candidates += [b + 1.0 for b in beta_vals] + [2.0]
    elif order != 1:
        raise NotImplementedError(
            f"Basis-expansion scheme order={order} is not implemented yet."
        )

    seen: set[float] = set()
    unique: list[float] = []
    for v in candidates:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    unique.sort()

    return tuple(jnp.asarray(v, dtype=dtype) for v in unique)


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


__all__ = ["quadratic_iteration", "BasisExpansionSpec"]
