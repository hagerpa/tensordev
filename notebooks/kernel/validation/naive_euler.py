r"""Naive forward-Euler reference scheme for the FSSK signature kernel.

This module implements a first-order reference discretisation of the 2D Goursat
PDE that defines the FSSK signature kernel. It uses the same K/A/B strip
decomposition as the production ETD1 scheme (tensordev.kernel.fssk), but
replaces the exact matrix-exponential transport with a plain forward Euler step:

    expm(-Λh) A  →  (I - Λh) A         [first-order Taylor of the exponential]
    phi1(-Λh) src  →  src               [phi1(0) = I, leading term]

The area term (Lambda-Lambda cross-coupling and source cancellation) is kept
identical to ETD1 so that the source contributions are consistently cancelled.

Cell update (naive Euler vs ETD1)
----------------------------------
ETD1 strip update (exact):
    A_next = expm(-Λ ds) A_n  +  phi1(-Λ ds) γ η_n

Naive Euler strip update:
    A_next = (I - Λ ds) A_n  +  γ η_n
           = A_n  +  ds (-Λ A_n)  +  γ η_n

The area term is the same trapezoidal formula as ETD1:
    area = 0.5 ds dt (Λ K_w Λᵀ + Λ K_n Λᵀ) − 0.5 γ (η_w + η_n)

Error analysis
--------------
The strip error per cell is O(ds² · |A|), and the accumulated strip |A| is
O(gij · η / Λ) = O(h). So the per-cell error is O(h³), and summing over all
O(1/h²) cells gives a total error of O(h) — first-order convergence.

Public API
----------
    naive_euler_fssk_kernel(X, Y, kernel, dt_x, dt_y, ...)
        Compute the terminal kernel value(s) using the naive Euler scheme.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from tensordev.kernel.fssk import (
    CellData,
    RowState,
    _build_gamma_grid_static,
    _lambda_left_right,
    _prepare_dt,
    _scale_like,
    _source,
    _to_nodes,
)
from tensordev.kernel.static_kernels import LinearKernel, StaticKernel
from tensordev.sss.kernel import FSSK
from tensordev.util.path_preprocessing import DyadicOrder, normalize_dyadic_order

Array = jax.Array


# ---------------------------------------------------------------------------
# Naive Euler cell step
# ---------------------------------------------------------------------------


def _naive_euler_cell_step(lambda_op, cell: CellData, *, dtype) -> tuple:
    """One cell update via plain forward Euler (no matrix exponentials).

    Left strip (s-direction):
        ETD1: A_next = expm(-Λ ds) A_n  + phi1(-Λ ds) src_n
        Euler: A_next = (I - Λ ds) A_n  + src_n
            [replaces expm with first-order Taylor; phi1(0) = I so src unchanged]

    Right strip (t-direction):
        ETD1: B_next = B_w expm(-Λ dt)  + src_w phi1(-Λ dt)
        Euler: B_next = B_w (I - Λ dt)  + src_w

    Area (same trapezoidal structure as ETD1; handles source cancellation):
        area = 0.5 ds dt (Λ K_w Λᵀ + Λ K_n Λᵀ) − 0.5 γ (η_w + η_n)
    """
    gij, ds, dt = cell.gamma_nw, cell.hx, cell.hy
    A_prev, B_curr = cell.A_n, cell.B_w

    src_n = _source(gij, cell.eta_n)
    A_next = (A_prev
              - _scale_like(ds, A_prev) * lambda_op.lambda_multiply_left(A_prev, dtype=dtype)
              + src_n)

    src_w = _source(gij, cell.eta_w)
    B_next = (B_curr
              - _scale_like(dt, B_curr) * lambda_op.lambda_multiply_right(B_curr, dtype=dtype)
              + src_w)

    dA = A_next - A_prev
    dB = B_next - B_curr

    # Same area formula as ETD1: trapezoidal Lambda^2 K and source cancellation.
    L2K_w = _lambda_left_right(lambda_op, cell.K_w, dtype=dtype)
    L2K_n = _lambda_left_right(lambda_op, cell.K_n, dtype=dtype)
    dsdt = _scale_like(ds * dt, L2K_w)
    area = 0.5 * dsdt * (L2K_w + L2K_n) - 0.5 * _source(gij, cell.eta_w + cell.eta_n)

    K_se = cell.K_w + cell.K_n - cell.K_nw + area + dA + dB
    eta_se = 1.0 + jnp.sum(K_se, axis=(-2, -1))
    return K_se, eta_se, A_next, B_next


# ---------------------------------------------------------------------------
# Scan-based solver (terminal value only)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _get_naive_scan_solver(*, dyadic_order):
    dyadic_order_x, dyadic_order_y = dyadic_order

    @jax.jit
    def solver(gamma, dt_x, dt_y, *, lambda_op):
        batch, s_coarse, t_coarse, r, _ = gamma.shape
        dtype = gamma.dtype
        n_x = 1 << dyadic_order_x
        n_y = 1 << dyadic_order_y
        s_steps = (s_coarse - 1) * n_x
        t_steps = (t_coarse - 1) * n_y
        t_nodes = t_steps + 1

        zero_node = jnp.zeros((batch, r, r), dtype=dtype)
        one_node = jnp.ones((batch,), dtype=dtype)
        initial_row = RowState.zeros(batch=batch, t_nodes=t_nodes, r=r, dtype=dtype)

        def _g(i, j):
            return gamma[
                :,
                jnp.minimum(i >> dyadic_order_x, s_coarse - 1),
                jnp.minimum(j >> dyadic_order_y, t_coarse - 1),
            ]

        def row_step(north, xs):
            hx, ix = xs

            def cell_scan(carry, iy):
                K_w, eta_w, A_w, B_w = carry
                cell = CellData(
                    hx=hx, hy=dt_y[iy],
                    K_w=K_w, K_n=north.K[:, iy + 1], K_nw=north.K[:, iy],
                    eta_w=eta_w, eta_n=north.eta[:, iy + 1], eta_nw=north.eta[:, iy],
                    A_w=A_w, A_n=north.A[:, iy + 1], A_nw=north.A[:, iy],
                    B_w=B_w, B_n=north.B[:, iy + 1], B_nw=north.B[:, iy],
                    gamma_nw=_g(ix, iy),
                )
                out = _naive_euler_cell_step(lambda_op, cell, dtype=dtype)
                return out, out

            _, cell_hist = jax.lax.scan(
                cell_scan,
                (zero_node, one_node, zero_node, zero_node),
                jnp.arange(t_steps, dtype=jnp.int32),
            )
            south = RowState.from_scan_hist(
                zero_node=zero_node, one_node=one_node, hist=cell_hist
            )
            return south, None

        last_row, _ = jax.lax.scan(
            row_step,
            initial_row,
            (dt_x, jnp.arange(s_steps, dtype=jnp.int32)),
        )
        return last_row.eta[:, -1], last_row.K[:, -1], last_row.A[:, -1], last_row.B[:, -1]

    return solver


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def naive_euler_fssk_kernel(
    X,
    Y,
    *,
    kernel: FSSK,
    dt_x: float | Array,
    dt_y: float | Array,
    pairwise: bool = False,
    dyadic_order: DyadicOrder = 0,
    increment_in: bool = False,
    static_kernel: StaticKernel = LinearKernel(scale=1.0),
) -> Array:
    """Compute FSSK terminal kernel values using the naive forward Euler scheme.

    Replaces the ETD1 matrix-exponential transport with a first-order Taylor
    approximation ``(I - Λh) A`` while keeping the same area (source-correction)
    formula as ETD1. The result converges first-order in the step size.

    Parameters
    ----------
    X, Y : array-like
        Path data of shape ``(batch, length, d)`` or ``(length, d)``.
    kernel : FSSK
        Algebraic kernel specification.
    dt_x, dt_y : float or 1-D Array
        Step sizes for X and Y paths.
    pairwise : bool
        If True, compute the full (batch_x, batch_y) Gram matrix.
    dyadic_order : int or (int, int)
        Dyadic refinement level(s) — halves the step size 2^k times.
    increment_in : bool
        If True, X and Y are path increments rather than path nodes.
    static_kernel : StaticKernel
        Static kernel for building gamma (default: linear/dot-product).

    Returns
    -------
    Array of shape (batch,) or (batch_x, batch_y).
    """
    dim = kernel.path_dim
    X_nodes = _to_nodes(X, path_dim=dim, name="X", increment_in=increment_in)
    Y_nodes = _to_nodes(Y, path_dim=dim, name="Y", increment_in=increment_in)

    dtype = jnp.result_type(X_nodes.dtype, Y_nodes.dtype, kernel.A.dtype, kernel.b.dtype)
    s_intervals = int(X_nodes.shape[-2]) - 1
    t_intervals = int(Y_nodes.shape[-2]) - 1

    dt_x_arr, _ = _prepare_dt(dt_x, s_intervals, name="dt_x", dtype=dtype)
    dt_y_arr, _ = _prepare_dt(dt_y, t_intervals, name="dt_y", dtype=dtype)

    gamma_coarse, batch_shape = _build_gamma_grid_static(
        X_nodes, Y_nodes,
        kernel=kernel, static_kernel=static_kernel,
        pairwise=pairwise, dtype=dtype,
    )

    dyadic_order_x, dyadic_order_y = normalize_dyadic_order(dyadic_order)
    n_x = 1 << dyadic_order_x
    n_y = 1 << dyadic_order_y
    if dyadic_order_x > 0 or dyadic_order_y > 0:
        gamma_coarse = gamma_coarse * dtype.type(0.5 ** (dyadic_order_x + dyadic_order_y))
    if dyadic_order_x > 0:
        dt_x_arr = jnp.repeat(dt_x_arr * dtype.type(0.5 ** dyadic_order_x), n_x, axis=0)
    if dyadic_order_y > 0:
        dt_y_arr = jnp.repeat(dt_y_arr * dtype.type(0.5 ** dyadic_order_y), n_y, axis=0)

    solver = _get_naive_scan_solver(dyadic_order=(dyadic_order_x, dyadic_order_y))
    eta_t, *_ = solver(gamma_coarse, dt_x_arr, dt_y_arr, lambda_op=kernel.Lambda)
    return eta_t.reshape(batch_shape)


__all__ = ["naive_euler_fssk_kernel"]