# TODO: Abstract this to general cores using tensor_abra etc...

from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Optional, Union

import jax.numpy as jnp
from jax import lax

from tensordev import Jax
from tensordev.core.universal import DenseElemFirstOn
from tensordev.kernel.base_kernel import BaseKernel

JaxCore = Jax()
Array = jnp.ndarray

from dataclasses import dataclass
from typing import Literal, Optional

import jax.numpy as jnp


def free_kernel(
        x: Union[DenseElemFirstOn, tuple[Callable[[Array], DenseElemFirstOn], Array]],
        y: Union[DenseElemFirstOn, tuple[Callable[[Array], DenseElemFirstOn], Array]],
        *,
        evaluate: Literal["terminal", "grid"] = "terminal",
        return_fg: bool = False,
        pairwise: bool = False,
        chunk_size_x: Optional[int] = None,
        chunk_size_y: Optional[int] = None,
        backend: Literal["scan", "wavefront"] = "scan",
        dyadic_order: int = 0,
        quadrature: Literal["left", "midpoint", "trapezoid"] = "trapezoid",
        core=None,
        increment_in: bool = False):
    """
    Compute the truncated kernel of free developments.

    Parameters
    ----------
    x, y :
        Each input is either

        - a ``DenseElemFirstOn``, interpreted as path values if
          ``increment_in=False`` and as interval increments if
          ``increment_in=True``, or
        - a pair ``(callable, grid)``, where the callable is interpreted as the
          characteristic velocity and converted to increments on the given grid.

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

    chunk_size_x, chunk_size_y : int, optional
        Optional chunk sizes used in pairwise mode to reduce memory usage.

    backend : {"scan", "wavefront"}, default="scan"
        Discrete solver backend.

    dyadic_order : int, default=0
        Dyadic refinement level applied before solving.

    quadrature : {"left", "midpoint", "trapezoid"}, default="trapezoid"
        Quadrature rule used when callable characteristic velocities are converted
        to increments.

    core : optional
        Tensor algebra backend. If ``None``, the default ``Jax()`` backend is used.

    increment_in : bool, default=False
        Whether tensor-valued inputs ``x`` and ``y`` are already given as interval
        increments. If ``False``, they are interpreted as path values and converted
        to increments along the interval axis.

    Returns
    -------
    Depending on ``evaluate`` and ``return_fg``, returns either ``w`` or
    ``(w, f, g)``, either at the terminal point or on the full discrete grid.
    """
    core = JaxCore if core is None else core

    dx = _normalize_input(
        x,
        quadrature=quadrature,
        dyadic_order=dyadic_order,
        increment_in=increment_in,
    )
    dy = _normalize_input(
        y,
        quadrature=quadrature,
        dyadic_order=dyadic_order,
        increment_in=increment_in,
    )

    if pairwise:
        dx, dy = _broadcast_pairwise(dx, dy)

    if chunk_size_x is not None or chunk_size_y is not None:
        return _solve_chunked(
            dx,
            dy,
            evaluate=evaluate,
            return_fg=return_fg,
            backend=backend,
            chunk_size_x=chunk_size_x,
            chunk_size_y=chunk_size_y,
            core=core,
        )

    if backend == "scan":
        return _solve_scan(dx, dy, evaluate=evaluate, return_fg=return_fg, core=core)

    if backend == "wavefront":
        return _solve_wavefront(dx, dy, evaluate=evaluate, return_fg=return_fg, core=core)

    raise ValueError(f"Unknown backend={backend!r}.")


def _normalize_input(
        arg: Union[DenseElemFirstOn, tuple[Callable[[Array], DenseElemFirstOn], Array]],
        *,
        quadrature: Literal["left", "midpoint", "trapezoid"],
        dyadic_order: int,
        increment_in: bool,
) -> DenseElemFirstOn:
    """
    Normalize one input into interval increments.

    Cases
    -----
    - arg = (callable, grid): the callable is interpreted as a characteristic
      velocity and converted to interval increments on the given node grid.
    - arg = levels:
        * if ``increment_in=False``, the levels are interpreted as path values and
          converted to interval increments by differencing along the interval axis;
        * if ``increment_in=True``, the levels are interpreted directly as interval
          increments.

    Refinement convention
    ---------------------
    - callable input: refine the node grid first, then compute increments on the
      refined grid;
    - tensor input with ``increment_in=False``: first convert path values to
      increments, then refine;
    - tensor input with ``increment_in=True``: refine the already-given increments.
    """
    if dyadic_order < 0:
        raise ValueError("dyadic_order must be non-negative.")

    if isinstance(arg, tuple) and len(arg) == 2 and callable(arg[0]):
        velocity, grid = arg
        grid = jnp.asarray(grid)

        if grid.ndim != 1:
            raise ValueError("The node grid must be one-dimensional.")
        if grid.shape[0] < 2:
            raise ValueError("The node grid must contain at least two nodes.")

        if dyadic_order > 0:
            for _ in range(dyadic_order):
                mids = 0.5 * (grid[:-1] + grid[1:])
                grid = jnp.sort(jnp.concatenate([grid, mids], axis=0))

        return _velocity_to_increments(
            velocity,
            grid,
            quadrature=quadrature,
        )

    levels = tuple(arg)
    if not levels:
        raise ValueError("Expected at least one positive tensor level.")

    if increment_in:
        normalized_levels = levels
    else:
        for k, lvl in enumerate(levels, start=1):
            if lvl.shape[-2] < 2:
                raise ValueError(
                    f"Path-valued tensor input at level {k} must have at least "
                    f"two nodes along the interval axis, got {lvl.shape[-2]}."
                )
        normalized_levels = tuple(jnp.diff(lvl, axis=-2) for lvl in levels)

    S = int(normalized_levels[0].shape[-2])
    for k, lvl in enumerate(normalized_levels, start=1):
        if lvl.shape[-2] != S:
            raise ValueError(
                f"All tensor levels must have the same interval axis length. "
                f"Level 1 has {S}, level {k} has {lvl.shape[-2]}."
            )

    if dyadic_order > 0:
        normalized_levels = _refine_increments(
            normalized_levels,
            dyadic_order=dyadic_order,
        )

    return normalized_levels


def _velocity_to_increments(
        velocity: Callable[[Array], DenseElemFirstOn],
        grid: Array,
        *,
        quadrature: Literal["left", "midpoint", "trapezoid"],
) -> DenseElemFirstOn:
    """
    Convert a characteristic velocity into interval increments on the node grid.

    For each interval [t_{i-1}, t_i], compute
        Δx_i = ∫_{t_{i-1}}^{t_i} x(r) dr
    by the chosen quadrature rule.
    """
    t0 = grid[:-1]
    t1 = grid[1:]
    dt = t1 - t0

    if quadrature == "left":
        vals = velocity(t0)
        return tuple(v * jnp.expand_dims(dt, axis=-1) for v in vals)

    if quadrature == "midpoint":
        tm = 0.5 * (t0 + t1)
        vals = velocity(tm)
        return tuple(v * jnp.expand_dims(dt, axis=-1) for v in vals)

    if quadrature == "trapezoid":
        v0 = velocity(t0)
        v1 = velocity(t1)
        return tuple(0.5 * (a + b) * jnp.expand_dims(dt, axis=-1) for a, b in zip(v0, v1))

    raise ValueError(f"Unknown quadrature={quadrature!r}.")


def _refine_increments(
        levels: DenseElemFirstOn,
        *,
        dyadic_order: int,
) -> DenseElemFirstOn:
    """
    Dyadically refine interval increments by equal splitting.

    If the original interval increments are
        Δx_1, ..., Δx_S,
    then after dyadic refinement of order λ each increment Δx_i is replaced by
        2**λ copies of Δx_i / 2**λ
    along the interval axis ``-2``.

    Parameters
    ----------
    levels :
        Positive tensor levels interpreted as interval increments. The k-th level
        is assumed to have shape ``batch + (S, d**k)``.

    dyadic_order : int
        Dyadic refinement order. ``0`` means no refinement.

    Returns
    -------
    DenseElemFirstOn
        Refined interval increments with interval axis length multiplied by
        ``2**dyadic_order``.
    """
    if dyadic_order < 0:
        raise ValueError("dyadic_order must be non-negative.")
    if dyadic_order == 0:
        return levels

    r = 2 ** dyadic_order
    return tuple(jnp.repeat(level / r, repeats=r, axis=-2) for level in levels)


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
        core,
):
    """
    Build the exact boundary data for the rowwise scan solver.

    Returns
    -------
    tuple
        (w_row_0, f_row_0, g_row_0, w_col_0, g_col_0, dx_steps, dy_steps, w_boundary_steps, f_boundary_steps)
    """
    M, N = len(dx), len(dy)
    m, n = M - 1, N - 1

    S = dx[0].shape[-2]
    T = dy[0].shape[-2]

    batch_shape = jnp.broadcast_shapes(dx[0].shape[:-2], dy[0].shape[:-2])
    dtype = jnp.result_type(dx[0], dy[0])
    one = jnp.ones(batch_shape, dtype=dtype)

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

    f_boundary = _boundary_development(
        dx,
        out_trunc=n,
        nodes=S + 1,
        template=dy,
    )
    g_boundary = _boundary_development(
        dy,
        out_trunc=m,
        nodes=T + 1,
        template=dx,
    )

    # south / west scalar boundary for w
    w_row_0 = jnp.broadcast_to(one[..., None], batch_shape + (T + 1,))
    w_col_0 = jnp.broadcast_to(one[..., None], batch_shape + (S + 1,))

    # south boundary for f is zero
    f_row_0 = tuple(
        jnp.zeros(batch_shape + (T + 1, dy[k].shape[-1]), dtype=dtype)
        for k in range(n)
    )

    # south boundary for g is the boundary development in y
    g_row_0 = g_boundary

    # west boundary for g is zero
    g_col_0 = tuple(
        jnp.zeros(batch_shape + (dx[k].shape[-1],), dtype=dtype)
        for k in range(m)
    )

    dx_steps = core.tensor_moveaxis(dx, source=-2, destination=0)
    dy_steps = core.tensor_moveaxis(dy, source=-2, destination=0)

    # row scan needs west boundary values for rows i = 1, ..., S
    w_boundary_steps = jnp.moveaxis(w_col_0[..., 1:], -1, 0)

    # row scan needs west f-boundary values for rows i = 1, ..., S
    f_boundary_steps = core.tensor_moveaxis(
        tuple(level[..., 1:, :] for level in f_boundary),
        source=-2,
        destination=0,
    )

    return (
        w_row_0,
        f_row_0,
        g_row_0,
        w_col_0,
        g_col_0,
        dx_steps,
        dy_steps,
        w_boundary_steps,
        f_boundary_steps,
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


def _solve_scan(
        dx: DenseElemFirstOn,
        dy: DenseElemFirstOn,
        *,
        evaluate: Literal["terminal", "grid"],
        return_fg: bool,
        core,
):
    """
    Solve the truncated free-kernel system by a rowwise scan, using symmetry to
    choose the more economical orientation.
    """
    S, T = dx[0].shape[-2], dy[0].shape[-2]

    swapped = T > S
    if swapped:
        dx, dy = dy, dx
        S, T = T, S

    M, N = len(dx), len(dy)
    P = min(M, N)
    m, n = M - 1, N - 1
    P_f, P_g = min(M, n), min(N, m)

    adj_left = partial(
        core.tensor_adjoint_product,
        side="left",
        w_first_on=True,
        y_first_on=True,
        first_on_out=True,
    )
    adj_right = partial(
        core.tensor_adjoint_product,
        side="right",
        w_first_on=True,
        y_first_on=True,
        first_on_out=True,
    )
    t_prod = partial(
        core.tensor_product,
        a_first_on=True,
        b_first_on=True,
    )
    t_sum = core.tensor_summation
    t_scal = core.tensor_scalar_multiply

    (
        u_row_0,
        f_row_0,
        g_row_0,
        _u_col_0,
        g_w_0,
        dx_steps,
        dy_steps,
        u_w_steps,
        f_w_steps,
    ) = _build_scan_boundaries(dx, dy, core=core)

    def split_rows(some_row):
        some_s = core.tensor_moveaxis(
            tuple(level[..., 1:, :] for level in some_row),
            source=-2,
            destination=0,
        )
        some_sw = core.tensor_moveaxis(
            tuple(level[..., :-1, :] for level in some_row),
            source=-2,
            destination=0,
        )
        return some_s, some_sw

    def prepend_rows(first, scanned, axis):
        return jnp.concatenate(
            (jnp.expand_dims(first, axis), jnp.moveaxis(scanned, 0, axis)),
            axis=axis,
        )

    def row_step(carry, data):
        u_s_row, f_s_row, g_s_row = carry
        dx_i, u_w0, f_w0 = data

        u_s = jnp.moveaxis(u_s_row[..., 1:], -1, 0)
        u_sw = jnp.moveaxis(u_s_row[..., :-1], -1, 0)

        f_s, f_sw = split_rows(f_s_row)
        g_s, g_sw = split_rows(g_s_row)

        def cell_step(carry_cell, data_cell):
            u_w, f_w, g_w = carry_cell
            dy_j, u_sj, u_swj, f_sj, f_swj, g_sj, g_swj = data_cell

            f_ij = t_sum(
                t_sum(
                    f_sj,
                    t_scal(dx_i[:P_f], u_sj),
                ),
                t_sum(
                    t_prod(f_sj, dx_i[:P_f], trunc=n),
                    adj_left(g_sj, dx_i, trunc=n),
                ),
            )

            g_ij = t_sum(
                t_sum(
                    g_w,
                    t_scal(dy_j[:P_g], u_w),
                ),
                t_sum(
                    t_prod(g_w, dy_j[:P_g], trunc=m),
                    adj_left(f_w, dy_j, trunc=m),
                ),
            )

            G_ij = core.tensor_inner_product(dx_i[:P], dy_j[:P])

            if P == 1:
                kap = 1.0 / 12.0
                G2_ij = G_ij * G_ij
                u_ij = (u_sj + u_w) * (1.0 + 0.5 * G_ij + kap * G2_ij) - u_swj * (1.0 - kap * G2_ij)
            else:
                dx_adj_dy = adj_right(dx_i, dy_j, trunc=n)
                dy_adj_dx = adj_right(dy_j, dx_i, trunc=m)

                def forcing(u_val, f_val, g_val):
                    return (
                            u_val * G_ij
                            + core.tensor_inner_product(f_val, dx_adj_dy)
                            + core.tensor_inner_product(g_val, dy_adj_dx)
                    )

                F_sw = forcing(u_swj, f_swj, g_swj)
                F_s = forcing(u_sj, f_sj, g_sj)
                F_w = forcing(u_w, f_w, g_w)

                u_p = u_sj + u_w - u_swj + F_sw
                F_p = forcing(u_p, f_ij, g_ij)
                u_ij = u_sj + u_w - u_swj + 0.25 * (F_sw + F_s + F_w + F_p)

            return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

        (_, _, _), (u_cells, f_cells, g_cells) = lax.scan(
            cell_step,
            (u_w0, f_w0, g_w_0),
            (dy_steps, u_s, u_sw, f_s, f_sw, g_s, g_sw),
        )

        u_row = jnp.concatenate(
            [u_w0[..., None], jnp.moveaxis(u_cells, 0, -1)],
            axis=-1,
        )
        f_row = tuple(
            jnp.concatenate(
                [f_w0[k][..., None, :], jnp.moveaxis(f_cells[k], 0, -2)],
                axis=-2,
            )
            for k in range(n)
        )
        g_row = tuple(
            jnp.concatenate(
                [g_w_0[k][..., None, :], jnp.moveaxis(g_cells[k], 0, -2)],
                axis=-2,
            )
            for k in range(m)
        )

        return (u_row, f_row, g_row), (u_row, f_row, g_row)

    if evaluate == "terminal":
        def row_step_terminal(carry, data):
            carry, _ = row_step(carry, data)
            return carry, None

        (u_last_row, f_last_row, g_last_row), _ = lax.scan(
            row_step_terminal,
            (u_row_0, f_row_0, g_row_0),
            (dx_steps, u_w_steps, f_w_steps),
        )

        out = u_last_row[..., -1]
        if return_fg:
            out = (
                out,
                tuple(level[..., -1, :] for level in f_last_row),
                tuple(level[..., -1, :] for level in g_last_row),
            )

    elif evaluate == "grid":
        if return_fg:
            (_, _, _), (u_rows, f_rows, g_rows) = lax.scan(
                row_step,
                (u_row_0, f_row_0, g_row_0),
                (dx_steps, u_w_steps, f_w_steps),
            )

            u = prepend_rows(u_row_0, u_rows, -2)
            f = tuple(prepend_rows(f0, fr, -3) for f0, fr in zip(f_row_0, f_rows))
            g = tuple(prepend_rows(g0, gr, -3) for g0, gr in zip(g_row_0, g_rows))
            out = (u, f, g)
        else:
            def row_step_uonly(carry, data):
                carry, (u_row, _, _) = row_step(carry, data)
                return carry, u_row

            (_, _, _), u_rows = lax.scan(
                row_step_uonly,
                (u_row_0, f_row_0, g_row_0),
                (dx_steps, u_w_steps, f_w_steps),
            )

            out = prepend_rows(u_row_0, u_rows, -2)

    else:
        raise ValueError(f"Unknown evaluate={evaluate!r}.")

    return _swap_scan_output(out, evaluate=evaluate, return_fg=return_fg) if swapped else out


def _solve_wavefront(
        dx: DenseElemFirstOn,
        dy: DenseElemFirstOn,
        *,
        evaluate: Literal["terminal", "grid"],
        return_fg: bool,
        core,
):
    """Optional anti-diagonal backend."""
    raise NotImplementedError


def _solve_chunked(
        dx: DenseElemFirstOn,
        dy: DenseElemFirstOn,
        *,
        evaluate: Literal["terminal", "grid"],
        return_fg: bool,
        backend: Literal["scan", "wavefront"],
        chunk_size_x: Optional[int],
        chunk_size_y: Optional[int],
        core,
):
    """Chunked pairwise evaluation."""
    raise NotImplementedError


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
    dyadic_order: int = 0
    quadrature: Literal["left", "midpoint", "trapezoid"] = "trapezoid"
    increment_in: bool = False
    core: object = None

    def __call__(
            self,
            X,
            Y,
            *,
            evaluate: str = "terminal",
            return_fg: bool = False,
            pairwise: bool = False,
    ):
        """
        Evaluate the configured free kernel.

        Parameters
        ----------
        X, Y :
            Normalized tensor-path inputs in packed positive-level form, with
            empirical sample axis on axis ``0``.
        evaluate : {"terminal", "grid"}, default="terminal"
            Whether to return only the terminal kernel values or the full discrete
            solution.
        return_fg : bool, default=False
            Whether to additionally return the tensor-valued components ``f`` and
            ``g``.
        pairwise : bool, default=False
            Whether to evaluate batchwise or pairwise over the empirical samples.

        Returns
        -------
        Array or tuple
            Output of ``free_kernel`` with the stored hyperparameters.
        """
        return free_kernel(
            X,
            Y,
            evaluate=evaluate,
            return_fg=return_fg,
            pairwise=pairwise,
            backend=self.backend,
            dyadic_order=self.dyadic_order,
            quadrature=self.quadrature,
            core=self.core,
            increment_in=self.increment_in,
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
