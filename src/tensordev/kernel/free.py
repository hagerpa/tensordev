# TODO: Abstract this to general cores using tensor_abra etc...

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass

from tensordev import Jax
from tensordev.core.universal import DenseElemFirstOn
from tensordev.kernel.base_kernel import BaseKernel
from tensordev.kernel.parallel import pmap_batch
from tensordev.kernel.util import DyadicOrder, normalize_dyadic_order

JaxCore = Jax()
Array = jnp.ndarray


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FreeCellData:
    """Backend-independent neighbourhood data for one free-kernel cell update.

    Bundles the three-corner stencil (nw, n, w) together with the driving
    increments so that the cell step function receives a single structured
    input.  This is the free-kernel analogue of ``fssk.CellData``.
    """

    # driving increments
    dx_i: tuple   # x-increment for row i, M levels
    dy_j: tuple   # y-increment for column j, N levels

    # scalar PDE state at three corners
    u_nw: Array   # (i-1, j-1)
    u_n: Array    # (i-1, j)
    u_w: Array    # (i, j-1)

    # left-adjoint (f) at three corners — n levels each
    f_nw: tuple
    f_n: tuple
    f_w: tuple

    # right-adjoint (g) at three corners — m levels each
    g_nw: tuple
    g_n: tuple
    g_w: tuple


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FreeRowState:
    """Full row of the 2-D grid — the free-kernel analogue of ``fssk.RowState``.

    Fields
    ------
    u : Array, shape ``batch + (t_nodes,)``
        Scalar PDE state along the row.
    f : tuple of Array
        Left-adjoint tensor levels, ``n`` arrays each ``batch + (t_nodes, width_k)``.
    g : tuple of Array
        Right-adjoint tensor levels, ``m`` arrays each ``batch + (t_nodes, width_k)``.
    """

    u: Array
    f: tuple
    g: tuple

    # -- assembly helpers (parallel to fssk.RowState) ----------------------

    @classmethod
    def from_scan_hist(cls, *, west_u, west_f, west_g, hist):
        """Build a row from its west-boundary cell and column-scan history.

        Parameters
        ----------
        west_u : Array, shape ``batch``
            Scalar west-boundary value for this row.
        west_f : tuple of Array
            West-boundary f values (``n`` levels), each ``batch + (width_k,)``.
        west_g : tuple of Array
            West-boundary g values (``m`` levels), each ``batch + (width_k,)``.
        hist : ``(u_hist, f_hist, g_hist)``
            Output of the inner ``lax.scan`` over columns.  ``u_hist`` has
            shape ``(T, batch…)``, ``f_hist[k]`` has ``(T, batch…, width_k)``,
            etc.
        """
        u_hist, f_hist, g_hist = hist
        u = jnp.concatenate(
            [west_u[..., None], jnp.moveaxis(u_hist, 0, -1)],
            axis=-1,
        )
        f = tuple(
            jnp.concatenate(
                [west_f[k][..., None, :], jnp.moveaxis(f_hist[k], 0, -2)],
                axis=-2,
            )
            for k in range(len(west_f))
        )
        g = tuple(
            jnp.concatenate(
                [west_g[k][..., None, :], jnp.moveaxis(g_hist[k], 0, -2)],
                axis=-2,
            )
            for k in range(len(west_g))
        )
        return cls(u=u, f=f, g=g)

    @classmethod
    def stack_history(cls, initial: "FreeRowState", hist: "FreeRowState"):
        """Stack the initial row and scanned row-history into a full grid.

        Parameters
        ----------
        initial : FreeRowState
            Row 0 (north boundary).
        hist : FreeRowState
            Stacked rows 1…S from the outer ``lax.scan``.  Each field has an
            extra leading scan axis at position 0.

        Returns
        -------
        FreeRowState
            Grid arrays with shape ``batch + (s_nodes, t_nodes, …)``.
        """
        def _prepend(first, scanned, axis):
            return jnp.concatenate(
                (jnp.expand_dims(first, axis), jnp.moveaxis(scanned, 0, axis)),
                axis=axis,
            )

        u = _prepend(initial.u, hist.u, -2)
        f = tuple(_prepend(f0, fr, -3) for f0, fr in zip(initial.f, hist.f))
        g = tuple(_prepend(g0, gr, -3) for g0, gr in zip(initial.g, hist.g))
        return cls(u=u, f=f, g=g)


def free_kernel(
        x: DenseElemFirstOn,
        y: DenseElemFirstOn,
        *,
        evaluate: Literal["terminal", "grid"] = "terminal",
        return_fg: bool = False,
        pairwise: bool = False,
        backend: Literal["scan", "wavefront"] = "scan",
        dyadic_order: DyadicOrder = 0,
        core=None,
        increment_in: bool = False,
        num_devices: int = 1):
    """
    Compute the truncated kernel of free developments.

    Parameters
    ----------
    x, y :
        Tensor-level tuples, interpreted as path values if
        ``increment_in=False`` and as interval increments if
        ``increment_in=True``.

        If ``increment_in=True``, the k-th level should have shape
        ``batch + (S, d**k)`` for ``x`` and ``batch + (T, d**k)`` for ``y``.
        If ``increment_in=False``, the k-th level should have shape
        ``batch + (S+1, d**k)`` for ``x`` and ``batch + (T+1, d**k)`` for ``y``.

    evaluate : {"terminal", "grid"}, default="terminal"
        Whether to return only the terminal value or the full discrete solution.

    return_fg : bool, default=False
        If ``True``, also return the tensor-valued components ``f`` and ``g``.

    pairwise : bool, default=False
        If ``True``, evaluate the kernel pairwise over the batch axes of ``x`` and ``y``.

    backend : {"scan", "wavefront"}, default="scan"
        Discrete solver backend.

    dyadic_order : int or tuple of int, default=0
        Dyadic refinement level.  A single int applies the same refinement to
        both paths.  A tuple ``(order_x, order_y)`` allows different refinement
        for ``x`` and ``y``.  Each coarse interval is split into
        ``2**order`` sub-intervals; driving data is stored on the coarse grid
        and indexed lazily (memory-efficient).

    core : optional
        Tensor algebra backend. If ``None``, the default ``Jax()`` backend is used.

    increment_in : bool, default=False
        Whether tensor-valued inputs ``x`` and ``y`` are already given as interval
        increments. If ``False``, they are interpreted as path values and converted
        to increments along the interval axis.

    num_devices : int, default=1
        Number of JAX virtual devices to distribute the batch across via
        ``jax.pmap``.  Set to ``jax.device_count()`` to use all available
        devices.  Requires ``XLA_FLAGS`` to be set *before* importing JAX::

            import os
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            import jax

        **Non-pairwise**: both ``x`` and ``y`` batches are sharded along
        axis 0 (must have the same leading size).

        **Pairwise**: only the ``x`` batch is sharded; the full ``y`` batch is
        replicated to every device.  The assembled result has the full
        ``batch_x × batch_y`` shape.

        See :mod:`tensordev.kernel.parallel` for details.

    Returns
    -------
    Depending on ``evaluate`` and ``return_fg``, returns either ``w`` or
    ``(w, f, g)``, either at the terminal point or on the full discrete grid.
    """
    dyadic_order = normalize_dyadic_order(dyadic_order)
    core = JaxCore if core is None else core

    dx = _to_increments(x, increment_in=increment_in)
    dy = _to_increments(y, increment_in=increment_in)

    def _solve(dx_, dy_):
        if backend == "scan":
            return _solve_scan(dx_, dy_, evaluate=evaluate, return_fg=return_fg,
                               dyadic_order=dyadic_order, core=core)
        if backend == "wavefront":
            return _solve_wavefront(dx_, dy_, evaluate=evaluate, return_fg=return_fg,
                                    dyadic_order=dyadic_order, core=core)
        raise ValueError(f"Unknown backend={backend!r}.")

    if num_devices > 1:
        if pairwise:
            # Shard along the x-batch axis (axis 0 of dx after broadcast).
            # dy is replicated via closure; the per-device result has shape
            # (batch_x // num_devices) + batch_y + ... and is reassembled
            # along axis 0 by pmap_batch.
            dx_b, dy_b = _broadcast_pairwise(dx, dy)
            return pmap_batch(lambda dx_s: _solve(dx_s, dy_b), dx_b,
                              num_devices=num_devices)
        else:
            # Shard both dx and dy along axis 0 (same leading batch size).
            return pmap_batch(_solve, dx, dy, num_devices=num_devices)

    if pairwise:
        dx, dy = _broadcast_pairwise(dx, dy)

    return _solve(dx, dy)


def _to_increments(
        arg: DenseElemFirstOn,
        *,
        increment_in: bool,
) -> DenseElemFirstOn:
    """Normalize tensor-level input into interval increments (coarse grid).

    If ``increment_in=False``, path values are differenced along axis ``-2``.
    If ``increment_in=True``, the levels are returned as-is after validation.
    """
    levels = tuple(arg)
    if not levels:
        raise ValueError("Expected at least one positive tensor level.")

    if increment_in:
        normalized = levels
    else:
        for k, lvl in enumerate(levels, start=1):
            if lvl.shape[-2] < 2:
                raise ValueError(
                    f"Path-valued tensor input at level {k} must have at least "
                    f"two nodes along the interval axis, got {lvl.shape[-2]}."
                )
        normalized = tuple(jnp.diff(lvl, axis=-2) for lvl in levels)

    S = int(normalized[0].shape[-2])
    for k, lvl in enumerate(normalized, start=1):
        if lvl.shape[-2] != S:
            raise ValueError(
                f"All tensor levels must have the same interval axis length. "
                f"Level 1 has {S}, level {k} has {lvl.shape[-2]}."
            )
    return normalized


def _broadcast_pairwise(
        dx: DenseElemFirstOn,
        dy: DenseElemFirstOn,
) -> tuple[DenseElemFirstOn, DenseElemFirstOn]:
    """
    Put the batch axes of x- and y-increments into outer-product position
    without flattening them.

    If
        dx_k.shape == batch_x + (S, d**k),
        dy_k.shape == batch_y + (T, d**k),
    then the returned shapes are
        dx_k.shape == batch_x + (1,)*len(batch_y) + (S, d**k),
        dy_k.shape == (1,)*len(batch_x) + batch_y + (T, d**k),
    so that subsequent tensor operations broadcast to batch_x + batch_y + ...
    """
    batch_x = dx[0].shape[:-2]
    batch_y = dy[0].shape[:-2]

    for k, level in enumerate(dx, start=1):
        if level.shape[:-2] != batch_x:
            raise ValueError(f"All levels of x must have the same batch shape; mismatch at level {k}.")
    for k, level in enumerate(dy, start=1):
        if level.shape[:-2] != batch_y:
            raise ValueError(f"All levels of y must have the same batch shape; mismatch at level {k}.")

    nx = len(batch_x)
    ny = len(batch_y)

    dx = tuple(level.reshape(batch_x + (1,) * ny + level.shape[-2:]) for level in dx)
    dy = tuple(level.reshape((1,) * nx + batch_y + level.shape[-2:]) for level in dy)

    return dx, dy


def _build_scan_boundaries(
        dx: DenseElemFirstOn,
        dy: DenseElemFirstOn,
        *,
        dyadic_order: tuple[int, int],
        core,
):
    """
    Build the exact boundary data for the rowwise scan solver.

    Increments ``dx``, ``dy`` are on the **coarse** grid.  When
    ``dyadic_order > (0, 0)`` the boundaries are computed on the coarse grid and
    the solver indexes them lazily via ``i >> dyadic_order_x`` / ``j >> dyadic_order_y``.

    Returns
    -------
    tuple
        (initial_row, g_w_0, dx_steps, dy_steps, u_w_steps, f_w_steps,
         S_coarse, T_coarse)
    """
    dyadic_order_x, dyadic_order_y = dyadic_order

    M, N = len(dx), len(dy)
    m, n = M - 1, N - 1

    S_coarse = dx[0].shape[-2]
    T_coarse = dy[0].shape[-2]

    batch_shape = jnp.broadcast_shapes(dx[0].shape[:-2], dy[0].shape[:-2])
    dtype = jnp.result_type(dx[0], dy[0])
    one = jnp.ones(batch_shape, dtype=dtype)

    r_x = 1 << dyadic_order_x
    r_y = 1 << dyadic_order_y
    S_fine = S_coarse * r_x
    T_fine = T_coarse * r_y

    def _boundary_development(
            driving,
            *,
            out_trunc,
            nodes,
            template,
    ):
        if out_trunc == 0:
            return tuple()

        if len(driving) == 0:
            return tuple(
                jnp.zeros(batch_shape + (nodes, template[k].shape[-1]), dtype=dtype)
                for k in range(out_trunc)
            )

        boundary_full = core.tensor_development(
            driving,
            axis=-2,
            trunc=out_trunc,
            block_size=1,
            accumulate=True,
            output_starting_point=True,
            increment_input=True,
        )
        return tuple(
            jnp.broadcast_to(
                boundary_full[k + 1],
                batch_shape + boundary_full[k + 1].shape[-2:],
            )
            for k in range(out_trunc)
        )

    # Boundary developments on the coarse grid (S_coarse+1 / T_coarse+1 nodes)
    f_boundary_coarse = _boundary_development(
        dx,
        out_trunc=n,
        nodes=S_coarse + 1,
        template=dy,
    )
    g_boundary_coarse = _boundary_development(
        dy,
        out_trunc=m,
        nodes=T_coarse + 1,
        template=dx,
    )

    # Row 0 boundary: needs T_fine+1 nodes.  The coarse boundary at coarse
    # node j maps to fine nodes j*r … (j+1)*r-1.  We index via j >> order.
    def _expand_boundary_to_fine(coarse_levels, fine_nodes, order):
        """Expand coarse-grid boundary to fine grid.

        Fine node ``i`` maps to coarse node ``min(i >> order, n_coarse-1)``.
        """
        if not coarse_levels:
            return tuple()
        if order == 0:
            return coarse_levels
        n_coarse = coarse_levels[0].shape[-2]
        idx = jnp.minimum(jnp.arange(fine_nodes) >> order, n_coarse - 1)
        return tuple(level[..., idx, :] for level in coarse_levels)

    # south / west scalar boundary for w (all ones on fine grid)
    w_row_0 = jnp.broadcast_to(one[..., None], batch_shape + (T_fine + 1,))
    w_col_0 = jnp.broadcast_to(one[..., None], batch_shape + (S_fine + 1,))

    # south boundary for f is zero on fine grid
    f_row_0 = tuple(
        jnp.zeros(batch_shape + (T_fine + 1, dy[k].shape[-1]), dtype=dtype)
        for k in range(n)
    )

    # south boundary for g is the boundary development in y, expanded to fine grid
    g_row_0_coarse = g_boundary_coarse  # (T_coarse+1 nodes)
    g_row_0 = _expand_boundary_to_fine(g_row_0_coarse, T_fine + 1, dyadic_order_y)

    # west boundary for g is zero (per-cell, no grid axis)
    g_col_0 = tuple(
        jnp.zeros(batch_shape + (dx[k].shape[-1],), dtype=dtype)
        for k in range(m)
    )

    # Scale coarse increments for the fine grid: each sub-cell gets dx/r_x, dy/r_y
    dx_scaled = tuple(level / r_x for level in dx) if dyadic_order_x > 0 else dx
    dy_scaled = tuple(level / r_y for level in dy) if dyadic_order_y > 0 else dy

    # Move interval axis to position 0 for scanning (coarse grid)
    dx_steps_coarse = core.tensor_moveaxis(dx_scaled, source=-2, destination=0)
    dy_steps_coarse = core.tensor_moveaxis(dy_scaled, source=-2, destination=0)

    # West boundary values for rows i = 1, ..., S_fine
    w_boundary_steps = jnp.moveaxis(w_col_0[..., 1:], -1, 0)

    # West f-boundary values for rows i = 1, ..., S_fine
    # Expand coarse f_boundary to fine grid, then extract rows 1..S_fine
    f_boundary_fine = _expand_boundary_to_fine(f_boundary_coarse, S_fine + 1, dyadic_order_x)
    f_boundary_steps = core.tensor_moveaxis(
        tuple(level[..., 1:, :] for level in f_boundary_fine),
        source=-2,
        destination=0,
    )

    initial_row = FreeRowState(u=w_row_0, f=f_row_0, g=g_row_0)

    return (
        initial_row,
        g_col_0,
        dx_steps_coarse,
        dy_steps_coarse,
        w_boundary_steps,
        f_boundary_steps,
        S_coarse,
        T_coarse,
    )


def _swap_scan_output(
        out,
        *,
        evaluate: Literal["terminal", "grid"],
        return_fg: bool,
):
    """
    Map the swapped rowwise-scan output back to the original x/y orientation.
    """
    if not return_fg:
        if evaluate == "grid":
            return jnp.swapaxes(out, -2, -1)
        return out

    w_swapped, g_swapped, f_swapped = out

    if evaluate == "terminal":
        return w_swapped, f_swapped, g_swapped

    return (
        jnp.swapaxes(w_swapped, -2, -1),
        tuple(jnp.swapaxes(level, -3, -2) for level in f_swapped),
        tuple(jnp.swapaxes(level, -3, -2) for level in g_swapped),
    )


def _free_cell_step(cell: FreeCellData, *, core, M, N, m, n, P, P_f, P_g):
    """Compute the SE corner ``(u_se, f_se, g_se)`` from a three-corner neighbourhood.

    This is the free-kernel analogue of ``fssk._heun_cell_step``.  It
    implements a Heun-like predictor–corrector scheme with a special fast
    path when ``M == N == 1`` (scalar-only, no tensor adjoint states).

    Parameters
    ----------
    cell : FreeCellData
        Three-corner stencil + driving increments.
    core : tensor algebra backend
        Provides ``tensor_inner_product``, ``tensor_adjoint_product``, etc.
    M, N : int
        Number of x- / y-levels (including level 0 if it were present; here
        ``M = len(dx)``, ``N = len(dy)``).
    m, n : int
        ``M - 1``, ``N - 1`` — truncation depths for g and f respectively.
    P, P_f, P_g : int
        ``min(M, N)``, ``min(M, n)``, ``min(N, m)`` — overlap depths.

    Returns
    -------
    u_se : Array
        Scalar PDE value at the SE corner.
    f_se : tuple
        Left-adjoint tensor at the SE corner (``n`` levels).
    g_se : tuple
        Right-adjoint tensor at the SE corner (``m`` levels).
    """
    adj_left = partial(
        core.tensor_adjoint_product,
        side="left", w_first_on=True, y_first_on=True, first_on_out=True,
    )
    adj_right = partial(
        core.tensor_adjoint_product,
        side="right", w_first_on=True, y_first_on=True, first_on_out=True,
    )
    t_prod = partial(core.tensor_product, a_first_on=True, b_first_on=True)
    t_sum = core.tensor_summation
    t_scal = core.tensor_scalar_multiply
    t_inner = core.tensor_inner_product

    dx_i, dy_j = cell.dx_i, cell.dy_j
    u_nw, u_n, u_w = cell.u_nw, cell.u_n, cell.u_w
    f_nw, f_n, f_w = cell.f_nw, cell.f_n, cell.f_w
    g_nw, g_n, g_w = cell.g_nw, cell.g_n, cell.g_w

    G_ij = t_inner(dx_i[:P], dy_j[:P])

    # --- M == 1, N == 1 fast path (scalar-only, no adjoint states) --------
    if M == 1 and N == 1:
        kap = 1.0 / 12.0
        G2_ij = G_ij * G_ij
        u_se = (u_n + u_w) * (1.0 + 0.5 * G_ij + kap * G2_ij) \
               - u_nw * (1.0 - kap * G2_ij)
        return u_se, f_n, g_w          # f, g are empty tuples — pass through

    # --- general Heun predictor–corrector ---------------------------------
    dx_adj_dy = adj_right(dx_i, dy_j, trunc=n)
    dy_adj_dx = adj_right(dy_j, dx_i, trunc=m)

    def f_increment(u_val, f_val, g_val):
        return t_sum(
            t_sum(
                t_scal(dx_i[:P_f], u_val),
                t_prod(f_val, dx_i[:P_f], trunc=n),
            ),
            adj_left(g_val, dx_i, trunc=n),
        )

    def g_increment(u_val, f_val, g_val):
        return t_sum(
            t_sum(
                t_scal(dy_j[:P_g], u_val),
                t_prod(g_val, dy_j[:P_g], trunc=m),
            ),
            adj_left(f_val, dy_j, trunc=m),
        )

    def forcing(u_val, f_val, g_val):
        return (
                u_val * G_ij
                + t_inner(f_val, dx_adj_dy)
                + t_inner(g_val, dy_adj_dx)
        )

    # stage 0: one-sided provisional edge advances
    df0 = f_increment(u_n, f_n, g_n)
    dg0 = g_increment(u_w, f_w, g_w)

    f_p = t_sum(f_n, df0)
    g_p = t_sum(g_w, dg0)

    F_nw = forcing(u_nw, f_nw, g_nw)
    F_n = forcing(u_n, f_n, g_n)
    F_w = forcing(u_w, f_w, g_w)

    u_base = u_n + u_w - u_nw
    u_p = u_base + F_nw

    # stage 1: coupled correction for f and g at the provisional corner
    df1 = f_increment(u_p, f_p, g_p)
    dg1 = g_increment(u_p, f_p, g_p)

    f_se = t_sum(f_n, t_scal(t_sum(df0, df1), 0.5))
    g_se = t_sum(g_w, t_scal(t_sum(dg0, dg1), 0.5))

    # final scalar correction with corrected corner tensors
    F_c = forcing(u_p, f_se, g_se)
    u_se = u_base + 0.25 * (F_nw + F_n + F_w + F_c)

    return u_se, f_se, g_se


def _solve_scan(
        dx: DenseElemFirstOn,
        dy: DenseElemFirstOn,
        *,
        evaluate: Literal["terminal", "grid"],
        return_fg: bool,
        dyadic_order: tuple[int, int],
        core,
):
    """
    Solve the truncated free-kernel system by a rowwise scan, using symmetry to
    choose the more economical orientation.

    Driving increments ``dx``, ``dy`` live on the **coarse** grid.  When
    ``dyadic_order > (0, 0)`` each coarse interval is split into
    ``2^dyadic_order_x`` / ``2^dyadic_order_y`` sub-intervals; the coarse
    increment is reused (scaled by ``1/r``) for every sub-cell via
    ``i >> dyadic_order`` indexing — no data duplication.
    """
    dyadic_order_x, dyadic_order_y = dyadic_order
    r_x = 1 << dyadic_order_x
    r_y = 1 << dyadic_order_y
    S_fine, T_fine = dx[0].shape[-2] * r_x, dy[0].shape[-2] * r_y

    swapped = T_fine > S_fine
    if swapped:
        dx, dy = dy, dx
        S_fine, T_fine = T_fine, S_fine
        dyadic_order = (dyadic_order_y, dyadic_order_x)
        dyadic_order_x, dyadic_order_y = dyadic_order

    M, N = len(dx), len(dy)
    P = min(M, N)
    m, n = M - 1, N - 1
    P_f, P_g = min(M, n), min(N, m)

    (
        initial_row,
        g_w_0,
        dx_steps_coarse,
        dy_steps_coarse,
        u_w_steps,
        f_w_steps,
        S_coarse,
        T_coarse,
    ) = _build_scan_boundaries(dx, dy, dyadic_order=dyadic_order, core=core)

    def _idx_coarse_x(steps_coarse, fine_idx):
        """Index coarse driving data at fine-grid position ``fine_idx`` (x-axis)."""
        coarse_idx = fine_idx >> dyadic_order_x
        return tuple(level[coarse_idx] for level in steps_coarse)

    def _idx_coarse_y(steps_coarse, fine_idx):
        """Index coarse driving data at fine-grid position ``fine_idx`` (y-axis)."""
        coarse_idx = fine_idx >> dyadic_order_y
        return tuple(level[coarse_idx] for level in steps_coarse)

    def _split_tensor_row(some_row):
        """Split a tensor-row into (north j+1, northwest j) with scan axis at 0."""
        some_n = core.tensor_moveaxis(
            tuple(level[..., 1:, :] for level in some_row),
            source=-2, destination=0,
        )
        some_nw = core.tensor_moveaxis(
            tuple(level[..., :-1, :] for level in some_row),
            source=-2, destination=0,
        )
        return some_n, some_nw

    def row_step(north: FreeRowState, data):
        ix, u_w0, f_w0 = data
        dx_i = _idx_coarse_x(dx_steps_coarse, ix)

        # Extract north (j+1) and northwest (j) columns for inner scan
        u_n = jnp.moveaxis(north.u[..., 1:], -1, 0)
        u_nw = jnp.moveaxis(north.u[..., :-1], -1, 0)
        f_n, f_nw = _split_tensor_row(north.f)
        g_n, g_nw = _split_tensor_row(north.g)

        def inner_step(carry, col_data):
            u_w, f_w, g_w = carry
            iy, u_nj, u_nwj, f_nj, f_nwj, g_nj, g_nwj = col_data
            dy_j = _idx_coarse_y(dy_steps_coarse, iy)

            cell = FreeCellData(
                dx_i=dx_i, dy_j=dy_j,
                u_nw=u_nwj, u_n=u_nj, u_w=u_w,
                f_nw=f_nwj, f_n=f_nj, f_w=f_w,
                g_nw=g_nwj, g_n=g_nj, g_w=g_w,
            )
            u_se, f_se, g_se = _free_cell_step(
                cell, core=core, M=M, N=N, m=m, n=n, P=P, P_f=P_f, P_g=P_g,
            )
            return (u_se, f_se, g_se), (u_se, f_se, g_se)

        _, hist = lax.scan(
            inner_step,
            (u_w0, f_w0, g_w_0),
            (jnp.arange(T_fine, dtype=jnp.int32),
             u_n, u_nw, f_n, f_nw, g_n, g_nw),
        )

        south = FreeRowState.from_scan_hist(
            west_u=u_w0, west_f=f_w0, west_g=g_w_0, hist=hist,
        )
        return south, south

    row_indices = jnp.arange(S_fine, dtype=jnp.int32)

    if evaluate == "terminal":
        def row_step_terminal(north, data):
            south, _ = row_step(north, data)
            return south, None

        last_row, _ = lax.scan(
            row_step_terminal,
            initial_row,
            (row_indices, u_w_steps, f_w_steps),
        )

        out = last_row.u[..., -1]
        if return_fg:
            out = (
                out,
                tuple(level[..., -1, :] for level in last_row.f),
                tuple(level[..., -1, :] for level in last_row.g),
            )

    elif evaluate == "grid":
        if return_fg:
            _, row_hist = lax.scan(
                row_step,
                initial_row,
                (row_indices, u_w_steps, f_w_steps),
            )

            grid = FreeRowState.stack_history(initial_row, row_hist)
            out = (grid.u, grid.f, grid.g)
        else:
            def row_step_uonly(north, data):
                south, _ = row_step(north, data)
                return south, south.u

            _, u_rows = lax.scan(
                row_step_uonly,
                initial_row,
                (row_indices, u_w_steps, f_w_steps),
            )

            def _prepend(first, scanned, axis):
                return jnp.concatenate(
                    (jnp.expand_dims(first, axis), jnp.moveaxis(scanned, 0, axis)),
                    axis=axis,
                )

            out = _prepend(initial_row.u, u_rows, -2)

    else:
        raise ValueError(f"Unknown evaluate={evaluate!r}.")

    return _swap_scan_output(out, evaluate=evaluate, return_fg=return_fg) if swapped else out


def _solve_wavefront(
        dx: DenseElemFirstOn,
        dy: DenseElemFirstOn,
        *,
        evaluate: Literal["terminal", "grid"],
        return_fg: bool,
        dyadic_order: tuple[int, int],
        core,
):
    """Anti-diagonal wavefront solver for the free-kernel PDE.

    Processes cells along anti-diagonals ``d = i + j`` in parallel via
    ``jax.vmap``, keeping only two diagonal buffers in memory.  This trades
    sequential depth for parallelism compared to the row-scan backend.

    Boundary developments (f on column 0, g on row 0) are pre-computed on the
    coarse grid, expanded to the fine grid, and injected into each fresh
    diagonal buffer before interior cells are written.
    """
    dyadic_order_x, dyadic_order_y = dyadic_order
    r_x = 1 << dyadic_order_x
    r_y = 1 << dyadic_order_y

    S_coarse, T_coarse = dx[0].shape[-2], dy[0].shape[-2]
    S_fine, T_fine = S_coarse * r_x, T_coarse * r_y
    s_nodes, t_nodes = S_fine + 1, T_fine + 1

    M, N = len(dx), len(dy)
    m, n = M - 1, N - 1
    P = min(M, N)
    P_f, P_g = min(M, n), min(N, m)

    batch_shape = jnp.broadcast_shapes(dx[0].shape[:-2], dy[0].shape[:-2])
    dtype = jnp.result_type(dx[0], dy[0])
    terminal_only = evaluate != "grid"

    # Scale coarse increments and move interval axis to position 0
    dx_sc = tuple(level / r_x for level in dx) if dyadic_order_x > 0 else dx
    dy_sc = tuple(level / r_y for level in dy) if dyadic_order_y > 0 else dy
    dx_steps = core.tensor_moveaxis(dx_sc, source=-2, destination=0)
    dy_steps = core.tensor_moveaxis(dy_sc, source=-2, destination=0)

    # --- Boundary developments on the fine grid ---

    def _bdev(driving, *, out_trunc):
        if out_trunc == 0:
            return tuple()
        bnd = core.tensor_development(
            driving, axis=-2, trunc=out_trunc, block_size=1,
            accumulate=True, output_starting_point=True, increment_input=True,
        )
        return tuple(
            jnp.broadcast_to(bnd[k + 1], batch_shape + bnd[k + 1].shape[-2:])
            for k in range(out_trunc)
        )

    def _expand(coarse, fine_nodes, order):
        if not coarse or order == 0:
            return coarse
        nc = coarse[0].shape[-2]
        idx = jnp.minimum(jnp.arange(fine_nodes) >> order, nc - 1)
        return tuple(lev[..., idx, :] for lev in coarse)

    # f_col0: f-boundary at column 0, nodes i = 0..S_fine  (n levels)
    # g_row0: g-boundary at row 0, nodes j = 0..T_fine     (m levels)
    f_col0 = _expand(_bdev(dx, out_trunc=n), S_fine + 1, dyadic_order_x)
    g_row0 = _expand(_bdev(dy, out_trunc=m), T_fine + 1, dyadic_order_y)

    # Index-ready: move node axis to 0 → (nodes,) + batch + (width,)
    f_col0_i = tuple(jnp.moveaxis(lev, -2, 0) for lev in f_col0)
    g_row0_i = tuple(jnp.moveaxis(lev, -2, 0) for lev in g_row0)

    W = min(S_fine, T_fine) + 1
    max_int_w = max(min(S_fine, T_fine), 1)

    # --- Diagonal buffers ---

    def _zeros_buf():
        """Fresh buffer: u = 1 (boundary default), f = 0, g = 0."""
        u = jnp.ones((W,) + batch_shape, dtype=dtype)
        f = tuple(
            jnp.zeros((W,) + batch_shape + (dy[k].shape[-1],), dtype=dtype)
            for k in range(n)
        )
        g = tuple(
            jnp.zeros((W,) + batch_shape + (dx[k].shape[-1],), dtype=dtype)
            for k in range(m)
        )
        return u, f, g

    def _inject_boundary(buf, d):
        """Write non-trivial boundary values into a fresh diagonal buffer.

        On diagonal ``d`` (i + j = d) at most two boundary positions exist:
        * **top** (i = 0, j = d): g comes from g_row0.
        * **left** (i = d, j = 0): f comes from f_col0.
        """
        u_b, f_b, g_b = buf
        ext_lo = jnp.maximum(0, d - T_fine)

        # Top boundary: node (0, d) at buffer position 0
        if g_row0_i:
            has_top = (ext_lo == 0) & (d <= T_fine)
            j_top = jnp.clip(d, 0, T_fine)
            g_b = tuple(
                lev.at[0].set(jnp.where(has_top, g_row0_i[k][j_top], lev[0]))
                for k, lev in enumerate(g_b)
            )

        # Left boundary: node (d, 0) at buffer position d - ext_lo
        if f_col0_i:
            has_left = d <= S_fine
            pos_left = jnp.clip(d - ext_lo, 0, W - 1)
            i_left = jnp.clip(d, 0, S_fine)
            f_b = tuple(
                lev.at[pos_left].set(jnp.where(has_left, f_col0_i[k][i_left], lev[pos_left]))
                for k, lev in enumerate(f_b)
            )

        return u_b, f_b, g_b

    buf_d0 = _inject_boundary(_zeros_buf(), 0)
    buf_d1 = _inject_boundary(_zeros_buf(), 1)

    n_diags = S_fine + T_fine - 1
    if n_diags <= 0:
        out_u = jnp.ones(batch_shape + (s_nodes, t_nodes), dtype=dtype)
        if return_fg:
            out_f = tuple(
                jnp.zeros(batch_shape + (s_nodes, t_nodes, dy[k].shape[-1],), dtype=dtype)
                for k in range(n)
            )
            out_g = tuple(
                jnp.zeros(batch_shape + (s_nodes, t_nodes, dx[k].shape[-1],), dtype=dtype)
                for k in range(m)
            )
            return out_u, out_f, out_g
        return out_u

    d_arr = jnp.arange(2, S_fine + T_fine + 1, dtype=jnp.int32)

    def _idx_x(fine_idx):
        ci = jnp.minimum(fine_idx >> dyadic_order_x, S_coarse - 1)
        return tuple(lev[ci] for lev in dx_steps)

    def _idx_y(fine_idx):
        ci = jnp.minimum(fine_idx >> dyadic_order_y, T_coarse - 1)
        return tuple(lev[ci] for lev in dy_steps)

    # --- Diagonal scan ---

    def diag_step(carry, d):
        buf_prev, buf_prev2 = carry
        u_p, f_p, g_p = buf_prev
        u_p2, f_p2, g_p2 = buf_prev2

        ext_lo = jnp.maximum(0, d - T_fine)
        ext_lo_p = jnp.maximum(0, d - 1 - T_fine)
        ext_lo_p2 = jnp.maximum(0, d - 2 - T_fine)

        int_lo = jnp.maximum(1, d - T_fine)
        int_hi = jnp.minimum(S_fine, d - 1)
        int_w = jnp.maximum(int_hi - int_lo + 1, 0)

        offsets = jnp.arange(max_int_w, dtype=jnp.int32)
        valid = offsets < int_w
        i_int = jnp.clip(int_lo + offsets, 1, S_fine)
        j_int = jnp.clip(d - i_int, 1, T_fine)
        im1, jm1 = i_int - 1, j_int - 1
        i_safe = jnp.where(valid, i_int, 0)
        j_safe = jnp.where(valid, j_int, 0)

        # Buffer positions for the three-corner stencil
        k_n = jnp.clip(im1 - ext_lo_p, 0, W - 1)
        k_w = jnp.clip(i_int - ext_lo_p, 0, W - 1)
        k_nw = jnp.clip(im1 - ext_lo_p2, 0, W - 1)

        # Gather stencil values
        u_n_v, u_w_v, u_nw_v = u_p[k_n], u_p[k_w], u_p2[k_nw]
        f_n_v = tuple(lev[k_n] for lev in f_p)
        f_w_v = tuple(lev[k_w] for lev in f_p)
        f_nw_v = tuple(lev[k_nw] for lev in f_p2)
        g_n_v = tuple(lev[k_n] for lev in g_p)
        g_w_v = tuple(lev[k_w] for lev in g_p)
        g_nw_v = tuple(lev[k_nw] for lev in g_p2)

        # Driving increments (coarse-indexed)
        dx_v = _idx_x(im1)
        dy_v = _idx_y(jm1)

        # Parallel cell step over the anti-diagonal
        cells = FreeCellData(
            dx_i=dx_v, dy_j=dy_v,
            u_nw=u_nw_v, u_n=u_n_v, u_w=u_w_v,
            f_nw=f_nw_v, f_n=f_n_v, f_w=f_w_v,
            g_nw=g_nw_v, g_n=g_n_v, g_w=g_w_v,
        )
        u_new, f_new, g_new = jax.vmap(
            lambda c: _free_cell_step(
                c, core=core, M=M, N=N, m=m, n=n, P=P, P_f=P_f, P_g=P_g,
            )
        )(cells)

        # Fresh buffer with boundary values, then overwrite interior
        new_u, new_f, new_g = _inject_boundary(_zeros_buf(), d)

        int_pos = jnp.clip(int_lo - ext_lo + offsets, 0, W - 1)
        vm = valid.reshape(valid.shape + (1,) * len(batch_shape))
        vm_t = valid.reshape(valid.shape + (1,) * (len(batch_shape) + 1))

        new_u = new_u.at[int_pos].set(jnp.where(vm, u_new, new_u[int_pos]))
        new_f = tuple(
            lev.at[int_pos].set(jnp.where(vm_t, f_new[k], lev[int_pos]))
            for k, lev in enumerate(new_f)
        )
        new_g = tuple(
            lev.at[int_pos].set(jnp.where(vm_t, g_new[k], lev[int_pos]))
            for k, lev in enumerate(new_g)
        )

        new_buf = (new_u, new_f, new_g)
        out = None if terminal_only else (i_safe, j_safe, valid, u_new, f_new, g_new)
        return (new_buf, buf_prev), out

    (last_buf, _), all_out = lax.scan(diag_step, (buf_d1, buf_d0), d_arr)

    if terminal_only:
        u_last, f_last, g_last = last_buf
        u_term = u_last[0]
        if return_fg:
            return u_term, tuple(lev[0] for lev in f_last), tuple(lev[0] for lev in g_last)
        return u_term

    # --- Reconstruct full grid ---
    i_all, j_all, valid_all, u_all, f_all, g_all = all_out

    i_flat = i_all.reshape(-1)
    j_flat = j_all.reshape(-1)
    v_flat = valid_all.reshape(-1)
    flat_idx = i_flat * t_nodes + j_flat  # 1-D index into (s_nodes * t_nodes)

    n_batch = len(batch_shape)
    # u_all: (n_diags, max_int_w) + batch_shape → batch_shape + (n_flat,)
    u_flat_b = jnp.moveaxis(u_all.reshape(-1, *batch_shape), 0, n_batch)
    vm_u = v_flat.reshape((1,) * n_batch + (-1,))

    u_grid = jnp.ones(batch_shape + (s_nodes * t_nodes,), dtype=dtype)
    u_grid = u_grid.at[..., flat_idx].set(
        jnp.where(vm_u, u_flat_b, u_grid[..., flat_idx])
    )
    u_grid = u_grid.reshape(batch_shape + (s_nodes, t_nodes))

    if not return_fg:
        return u_grid

    vm_fg = v_flat.reshape((1,) * n_batch + (-1,) + (1,))

    def _scatter_tuple(all_levels, n_levels, grid_width_fn, boundary_row, boundary_col):
        grids = []
        for k in range(n_levels):
            wk = grid_width_fn(k)
            g = jnp.zeros(batch_shape + (s_nodes * t_nodes, wk), dtype=dtype)
            flat = jnp.moveaxis(all_levels[k].reshape(-1, *batch_shape, wk), 0, n_batch)
            g = g.at[..., flat_idx, :].set(
                jnp.where(vm_fg, flat, g[..., flat_idx, :])
            )
            g = g.reshape(batch_shape + (s_nodes, t_nodes, wk))
            # Inject boundary row (i=0) and column (j=0)
            if boundary_row:
                g = g.at[..., 0, :, :].set(boundary_row[k])
            if boundary_col:
                g = g.at[..., :, 0, :].set(boundary_col[k])
            grids.append(g)
        return tuple(grids)

    f_grid = _scatter_tuple(
        f_all, n,
        lambda k: dy[k].shape[-1],
        boundary_row=tuple(
            jnp.zeros(batch_shape + (t_nodes, dy[k].shape[-1]), dtype=dtype)
            for k in range(n)
        ),
        boundary_col=f_col0,
    )
    g_grid = _scatter_tuple(
        g_all, m,
        lambda k: dx[k].shape[-1],
        boundary_row=g_row0,
        boundary_col=tuple(
            jnp.zeros(batch_shape + (s_nodes, dx[k].shape[-1]), dtype=dtype)
            for k in range(m)
        ),
    )

    return u_grid, f_grid, g_grid


@dataclass(frozen=True)
class FreeKernel(BaseKernel):
    """
    Kernel induced by truncated free developments of tensor-valued paths.

    This class evaluates the free kernel on tensor paths given in packed
    positive-level form and provides empirical kernel statistics such as
    batchwise kernel values, Gram matrices, MMD, and scoring rules.

    Input convention
    ----------------
    Inputs are given as one tensor level or as a tuple/list of tensor levels.

    - A single array is interpreted as a level-1 tensor path.
    - A tuple/list is interpreted as packed positive tensor levels.
    - A missing leading sample axis is promoted to batch size ``1``.

    Depending on ``increment_in``, tensor inputs are interpreted either as
    path values or as interval increments.
    """
    backend: Literal["scan", "wavefront"] = "scan"
    dyadic_order: DyadicOrder = 0
    increment_in: bool = False
    core: object = None
    num_devices: int = 1

    def __call__(
            self,
            X,
            Y,
            *,
            evaluate: str = "terminal",
            return_fg: bool = False,
            pairwise: bool = False,
    ):
        # ...existing code...
        return free_kernel(
            X,
            Y,
            evaluate=evaluate,
            return_fg=return_fg,
            pairwise=pairwise,
            backend=self.backend,
            dyadic_order=self.dyadic_order,
            core=self.core,
            increment_in=self.increment_in,
            num_devices=self.num_devices,
        )

    def _as_sample_batch(self, X):
        """
        Normalize tensor-path input to packed positive-level form with a leading
        empirical sample axis.

        Parameters
        ----------
        X :
            Either a single tensor level or a tuple/list of tensor levels. Each
            level is expected to have shape ``(batch, steps, width)`` or
            ``(steps, width)`` for a single sample.

        Returns
        -------
        tuple
            Tuple of arrays, each carrying the empirical sample axis on axis ``0``.

        Raises
        ------
        ValueError
            If no tensor levels are provided, if levels inconsistently include a
            sample axis, or if leading sample sizes disagree.
        """
        levels = (jnp.asarray(X),) if not isinstance(X, (tuple, list)) else tuple(jnp.asarray(z) for z in X)
        if not levels:
            raise ValueError("Expected at least one positive tensor level.")

        if all(level.ndim == 2 for level in levels):
            levels = tuple(level[None, ...] for level in levels)
        elif any(level.ndim == 2 for level in levels):
            raise ValueError("All levels must either all have a sample axis or all omit it.")

        return levels
