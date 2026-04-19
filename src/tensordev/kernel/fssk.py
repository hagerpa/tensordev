from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import jax
import jax.numpy as jnp

from tensordev.kernel.base_kernel import BaseKernel
from tensordev.kernel.util import DyadicOrder, normalize_dyadic_order
from tensordev.volterra.fssk.kernels import FSSKKernel

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FSSKSigKernel(BaseKernel):
    """Configured finite-state-space Volterra signature kernel.

    Wraps an :class:`FSSKKernel` together with discretization parameters into a
    callable :class:`BaseKernel` that can evaluate terminal values, full grids,
    Gram matrices, and MMD statistics via the inherited empirical helpers.

    Parameters
    ----------
    kernel : FSSKKernel
        Algebraic specification of the kernel (decay matrix Λ, projection
        tensors A, mixing weights b).
    dt_x, dt_y : float or Array
        Time-step size(s) for the *X* and *Y* paths respectively. A scalar
        means uniform spacing; a 1-D array provides per-interval widths.
    backend : {"scan", "wavefront"}, default="scan"
        Traversal strategy for the 2-D PDE grid.
    scheme : {"etd1", "heun"}, default="heun"
        Numerical cell-update scheme. ``"etd1"`` is first-order exponential
        time differencing; ``"heun"`` adds a predictor–corrector step for
        second-order convergence.
    precompute_propagators : bool, default=False
        If ``True``, matrix exponentials and φ-functions are precomputed once
        per time-step size and reused across all cells. Faster when dt is
        uniform but uses more memory.
    dyadic_order : int, default=0
        Dyadic refinement level. Each coarse interval is split into ``2^k``
        sub-intervals; gamma is stored on the coarse grid and indexed lazily.
    increment_in : bool, default=False
        If ``True``, *X* and *Y* are already provided as increments
        ``(batch, intervals, dim)`` rather than as path values
        ``(batch, nodes, dim)``.
    """

    kernel: FSSKKernel
    dt_x: Array | float
    dt_y: Array | float
    backend: str = "scan"
    scheme: str = "heun"
    precompute_propagators: bool = True
    dyadic_order: DyadicOrder = 0
    increment_in: bool = False

    def __call__(self, X, Y, *, evaluate="terminal", return_fg=False, pairwise=False):
        return fssk_sigkernel(
            X, Y,
            kernel=self.kernel, dt_x=self.dt_x, dt_y=self.dt_y,
            evaluate=evaluate, return_fg=return_fg, pairwise=pairwise,
            backend=self.backend, scheme=self.scheme,
            precompute_propagators=self.precompute_propagators,
            dyadic_order=self.dyadic_order,
            increment_in=self.increment_in,
        )

    def _as_sample_batch(self, X):
        X = jnp.asarray(X)
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim != 3:
            raise ValueError("Expected shape (batch, length, dim) or (length, dim).")
        if X.shape[-2] < 2:
            raise ValueError("Each path must contain at least two time points.")
        if X.shape[-1] != self.kernel.path_dim:
            raise ValueError(
                f"Expected terminal feature dimension {self.kernel.path_dim}, got {X.shape[-1]}."
            )
        return X


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CellData:
    """Backend-independent geometric neighbourhood for one cell update."""

    hx: Array
    hy: Array

    K_w: Array
    K_n: Array
    K_nw: Array

    eta_w: Array
    eta_n: Array
    eta_nw: Array

    A_w: Array
    A_n: Array
    A_nw: Array

    B_w: Array
    B_n: Array
    B_nw: Array

    gamma_nw: Array
    gamma_n: Array
    gamma_w: Array
    gamma_se: Array

    @staticmethod
    def move_scan_axis(x: Array) -> Array:
        return jnp.moveaxis(x, 1, 0)

    @staticmethod
    def unmove_scan_axis(x: Array) -> Array:
        return jnp.moveaxis(x, 0, 1)

    @staticmethod
    def prepend_history(boundary: Array, hist: Array) -> Array:
        return jnp.concatenate([boundary[:, None, ...], CellData.unmove_scan_axis(hist)], axis=1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RowState:
    K: Array
    eta: Array
    A: Array
    B: Array

    @classmethod
    def zeros(cls, *, batch: int, t_nodes: int, r: int, dtype) -> "RowState":
        return cls(
            K=jnp.zeros((batch, t_nodes, r, r), dtype=dtype),
            eta=jnp.ones((batch, t_nodes), dtype=dtype),
            A=jnp.zeros((batch, t_nodes, r, r), dtype=dtype),
            B=jnp.zeros((batch, t_nodes, r, r), dtype=dtype),
        )

    @classmethod
    def from_scan_hist(cls, *, zero_node: Array, one_node: Array, hist) -> "RowState":
        K_hist, eta_hist, A_hist, B_hist = hist
        ph = CellData.prepend_history
        return cls(
            K=ph(zero_node, K_hist), eta=ph(one_node, eta_hist),
            A=ph(zero_node, A_hist), B=ph(zero_node, B_hist),
        )

    @classmethod
    def stack_history(cls, initial: "RowState", hist: "RowState") -> "WaveState":
        um = CellData.unmove_scan_axis
        return WaveState(
            eta=jnp.concatenate([initial.eta[:, None, :], um(hist.eta)], axis=1),
            K=jnp.concatenate([initial.K[:, None], um(hist.K)], axis=1),
            A=jnp.concatenate([initial.A[:, None], um(hist.A)], axis=1),
            B=jnp.concatenate([initial.B[:, None], um(hist.B)], axis=1),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WaveState:
    eta: Array
    K: Array
    A: Array
    B: Array



@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EmptyTransportParams:
    pass


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PrecomputedTransportParams:
    expm_x: Array
    phi1_x: Array
    phi2_x: Array
    expm_y_t: Array
    phi1_y_t: Array
    phi2_y_t: Array


def fssk_sigkernel(
        X, Y, *,
        kernel: FSSKKernel,
        dt_x: Array | float, dt_y: Array | float,
        evaluate: str = "terminal", return_fg: bool = False, pairwise: bool = False,
        backend: str = "scan", scheme: str = "heun",
        precompute_propagators: bool = False, dyadic_order: DyadicOrder = 0,
        increment_in: bool = False,
):
    """Evaluate the beta PDE approximation of the FSSK signature kernel.

    Parameters
    ----------
    X, Y : array-like
        Path data of shape ``(batch, length, dim)`` or ``(length, dim)``.
        When ``increment_in=True`` the second axis indexes intervals (length
        equals the number of increments); otherwise it indexes nodes (length
        equals number of time-points ≥ 2).
    kernel : FSSKKernel
        Algebraic kernel specification.
    dt_x, dt_y : float or Array
        Time-step size(s). Scalar for uniform grids, 1-D array for
        non-uniform.
    evaluate : {"terminal", "grid"}, default="terminal"
        ``"terminal"`` returns only the value at the final grid corner;
        ``"grid"`` returns the full 2-D solution arrays.
    return_fg : bool, default=False
        If ``True``, also return the auxiliary matrices *K*, *A*, *B*
        alongside η.
    pairwise : bool, default=False
        If ``False``, evaluate batchwise ``k(Xᵢ, Yᵢ)``.
        If ``True``, evaluate the full pairwise block ``k(Xᵢ, Yⱼ)``.
    backend : {"scan", "wavefront"}, default="scan"
        Traversal strategy.
    scheme : {"etd1", "heun"}, default="heun"
        Cell-update scheme.
    precompute_propagators : bool, default=False
        Precompute and reuse matrix exponentials / φ-functions.
    dyadic_order : int, default=0
        Number of dyadic refinement levels.
    increment_in : bool, default=False
        If ``True``, *X* and *Y* are path increments, not path values.

    Returns
    -------
    Array or tuple
        * ``evaluate="terminal", return_fg=False`` — scalar η values,
          shape ``(batch,)`` or ``(bx, by)``.
        * ``evaluate="terminal", return_fg=True`` — tuple
          ``(η, K, A, B)`` at the terminal cell.
        * ``evaluate="grid", return_fg=False`` — η grid of shape
          ``(…, s_nodes, t_nodes)``.
        * ``evaluate="grid", return_fg=True`` — tuple
          ``(η, K, A, B)`` grids.
    """
    if backend not in {"scan", "wavefront"}:
        raise ValueError("backend must be either 'scan' or 'wavefront'.")
    if evaluate not in {"terminal", "grid"}:
        raise ValueError("evaluate must be either 'terminal' or 'grid'.")
    if scheme not in _SCHEME_REGISTRY:
        raise ValueError(f"Unknown scheme {scheme!r}. Supported: {', '.join(sorted(_SCHEME_REGISTRY))}.")
    dyadic_order_x, dyadic_order_y = normalize_dyadic_order(dyadic_order)

    dim = kernel.path_dim
    dX = _to_increments(X, path_dim=dim, name="X", increment_in=increment_in)
    dY = _to_increments(Y, path_dim=dim, name="Y", increment_in=increment_in)

    dtype = jnp.result_type(dX.dtype, dY.dtype, kernel.A.dtype, kernel.b.dtype)
    dt_x_arr, dt_x_uniform = _prepare_dt(dt_x, int(dX.shape[-2]), name="dt_x", dtype=dtype)
    dt_y_arr, dt_y_uniform = _prepare_dt(dt_y, int(dY.shape[-2]), name="dt_y", dtype=dtype)

    # Build gamma on the coarse grid; solvers index it via i >> dyadic_order.
    proj = _state_projection_tensor(kernel, dtype=dtype)
    gamma_coarse, batch_shape = _build_gamma_grid(dX, dY, proj=proj, pairwise=pairwise)

    # Dyadic refinement: scale gamma (sub-cell increments are 1/2^k of coarse)
    # and build fine-grid dt. Coarse gamma shape is unchanged.
    n_x = 1 << dyadic_order_x
    n_y = 1 << dyadic_order_y
    if dyadic_order_x > 0 or dyadic_order_y > 0:
        gamma_coarse = gamma_coarse * dtype.type(0.5 ** (dyadic_order_x + dyadic_order_y))
    if dyadic_order_x > 0:
        dt_x_arr = jnp.repeat(dt_x_arr * dtype.type(0.5 ** dyadic_order_x), n_x, axis=0)
    if dyadic_order_y > 0:
        dt_y_arr = jnp.repeat(dt_y_arr * dtype.type(0.5 ** dyadic_order_y), n_y, axis=0)
        # uniformity is preserved: all refined dt's are identical

    transport_params = _build_transport_params(
        kernel.Lambda, dt_x_arr, dt_y_arr,
        dt_x_uniform=dt_x_uniform, dt_y_uniform=dt_y_uniform,
        dtype=dtype, precompute_propagators=precompute_propagators,
    )

    terminal_only = evaluate != "grid"
    solver = _get_solver(backend=backend, scheme=scheme, precompute_propagators=precompute_propagators,
                         dyadic_order=(dyadic_order_x, dyadic_order_y), terminal_only=terminal_only)
    result = solver(
        gamma_coarse, dt_x_arr, dt_y_arr, lambda_op=kernel.Lambda,
        transport_params=transport_params,
    )

    if terminal_only:
        # result is (eta_terminal, K_terminal, A_terminal, B_terminal) — just the terminal cell
        eta_t, K_t, A_t, B_t = result
        eta_t = eta_t.reshape(batch_shape)
        if not return_fg:
            return eta_t
        K_t = K_t.reshape(batch_shape + K_t.shape[-2:])
        A_t = A_t.reshape(batch_shape + A_t.shape[-2:])
        B_t = B_t.reshape(batch_shape + B_t.shape[-2:])
        return eta_t, K_t, A_t, B_t

    eta_grid, K_grid, A_grid, B_grid = result
    eta_grid = eta_grid.reshape(batch_shape + eta_grid.shape[-2:])
    K_grid = K_grid.reshape(batch_shape + K_grid.shape[-4:])
    A_grid = A_grid.reshape(batch_shape + A_grid.shape[-4:])
    B_grid = B_grid.reshape(batch_shape + B_grid.shape[-4:])
    return (eta_grid, K_grid, A_grid, B_grid) if return_fg else eta_grid


# ---------------------------------------------------------------------------
# Shared local algebra
# ---------------------------------------------------------------------------


def _source(gamma: Array, eta: Array) -> Array:
    return gamma * eta[..., None, None]


# ---------------------------------------------------------------------------
# Shared transport helpers
# ---------------------------------------------------------------------------


def _build_transport_params(lambda_op, dt_x, dt_y, *, dt_x_uniform, dt_y_uniform, dtype, precompute_propagators):
    if not precompute_propagators:
        return EmptyTransportParams()

    def build_ops(dt, uniform):
        if uniform:
            e = lambda_op.expm(dt[0], dtype=dtype)
            p1 = lambda_op.phi1(dt[0], dtype=dtype)
            p2 = lambda_op.phi2(dt[0], dtype=dtype)
            n = dt.shape[0]
            return (jnp.broadcast_to(e, (n,) + e.shape),
                    jnp.broadcast_to(p1, (n,) + p1.shape),
                    jnp.broadcast_to(p2, (n,) + p2.shape))
        e = lambda_op.expm(dt, dtype=dtype)
        return e, lambda_op.phi1(dt, dtype=dtype), lambda_op.phi2(dt, dtype=dtype)

    expm_x, phi1_x, phi2_x = build_ops(dt_x, dt_x_uniform)
    expm_y, phi1_y, phi2_y = build_ops(dt_y, dt_y_uniform)
    T = lambda a: jnp.swapaxes(a, -1, -2)
    return PrecomputedTransportParams(
        expm_x=expm_x, phi1_x=phi1_x, phi2_x=phi2_x,
        expm_y_t=T(expm_y), phi1_y_t=T(phi1_y), phi2_y_t=T(phi2_y),
    )


def _scale_like(h: Array, x: Array) -> Array:
    return h.reshape(h.shape + (1,) * (x.ndim - h.ndim))


def _left_apply(op: Array, X: Array) -> Array:
    if op.ndim == 2:
        return jnp.einsum("ab,...bc->...ac", op, X)
    return jnp.einsum("wab,w...bc->w...ac", op, X)


def _right_apply(X: Array, op: Array) -> Array:
    if op.ndim == 2:
        return jnp.einsum("...ab,bc->...ac", X, op)
    return jnp.einsum("w...ab,wbc->w...ac", X, op)


def _exact_step_action(lambda_op, h, prev, src, *, side, dtype):
    """Exact exponential + phi1 transport step (action-based)."""
    if side == "left":
        return lambda_op.expm_multiply_left(h, prev, dtype=dtype) + lambda_op.phi1_multiply_left(h, src, dtype=dtype)
    return lambda_op.expm_multiply_right(h, prev, dtype=dtype) + lambda_op.phi1_multiply_right(h, src, dtype=dtype)


def _exact_step_precomputed(op, phi1, prev, src, *, side):
    """Precomputed matrix transport step (left or right)."""
    if side == "left":
        return _left_apply(op, prev) + _left_apply(phi1, src)
    return _right_apply(prev, op) + _right_apply(src, phi1)


def _etd2_step_action(lambda_op, h, prev, src0, dsrc, *, side, dtype):
    if side == "left":
        return (lambda_op.expm_multiply_left(h, prev, dtype=dtype)
                + lambda_op.phi1_multiply_left(h, src0, dtype=dtype)
                + lambda_op.phi2_multiply_left(h, dsrc, dtype=dtype))
    return (lambda_op.expm_multiply_right(h, prev, dtype=dtype)
            + lambda_op.phi1_multiply_right(h, src0, dtype=dtype)
            + lambda_op.phi2_multiply_right(h, dsrc, dtype=dtype))


def _etd2_step_precomputed(op, phi1, phi2, prev, src0, dsrc, *, side):
    if side == "left":
        return _left_apply(op, prev) + _left_apply(phi1, src0) + _left_apply(phi2, dsrc)
    return _right_apply(prev, op) + _right_apply(src0, phi1) + _right_apply(dsrc, phi2)


def _transport_step_etd1(lambda_op, h, prev, src, *,
                         side, transport_params, precompute_propagators, step_index, dtype):
    """ETD1 (exact + phi1) transport, action or precomputed."""
    if not precompute_propagators:
        return _exact_step_action(lambda_op, h, prev, src, side=side, dtype=dtype)
    ops = _get_precomputed_ops(transport_params, side, step_index)
    return _exact_step_precomputed(ops[0], ops[1], prev, src, side=side)


def _transport_step_etd2(lambda_op, h, prev, src0, dsrc, *,
                         side, transport_params, precompute_propagators, step_index, dtype):
    """ETD2 (exact + phi1 + phi2) transport, action or precomputed."""
    if not precompute_propagators:
        return _etd2_step_action(lambda_op, h, prev, src0, dsrc, side=side, dtype=dtype)
    ops = _get_precomputed_ops(transport_params, side, step_index)
    return _etd2_step_precomputed(ops[0], ops[1], ops[2], prev, src0, dsrc, side=side)


def _get_precomputed_ops(tp, side, idx):
    """Return (expm, phi1, phi2) for the given side and step index."""
    if side == "left":
        return tp.expm_x[idx], tp.phi1_x[idx], tp.phi2_x[idx]
    return tp.expm_y_t[idx], tp.phi1_y_t[idx], tp.phi2_y_t[idx]


# ---------------------------------------------------------------------------
# Cell steps
# ---------------------------------------------------------------------------


def _etd1_cell_step(lambda_op, cell, transport_params, *,
                    precompute_propagators, ix, iy, dtype):
    gij, ds, dt = cell.gamma_nw, cell.hx, cell.hy
    A_prev, B_curr = cell.A_n, cell.B_w

    kw = dict(transport_params=transport_params, precompute_propagators=precompute_propagators, dtype=dtype)
    A_next = _transport_step_etd1(lambda_op, ds, A_prev, _source(gij, cell.eta_n),
                                  side="left", step_index=ix, **kw)
    B_next = _transport_step_etd1(lambda_op, dt, B_curr, _source(gij, cell.eta_w),
                                  side="right", step_index=iy, **kw)

    dA = A_next - A_prev
    dB = B_next - B_curr

    L2K_w = _lambda_left_right(lambda_op, cell.K_w, dtype=dtype)
    L2K_n = _lambda_left_right(lambda_op, cell.K_n, dtype=dtype)
    dsdt = _scale_like(ds * dt, L2K_w)
    area = 0.5 * dsdt * (L2K_w + L2K_n) - 0.5 * _source(gij, cell.eta_w + cell.eta_n)
    K_se = cell.K_w + cell.K_n - cell.K_nw + area + dA + dB
    eta_se = 1.0 + jnp.sum(K_se, axis=(-2, -1))
    return K_se, eta_se, A_next, B_next


def _heun_cell_step(lambda_op, cell, transport_params, *,
                         precompute_propagators, ix, iy, dtype):
    gij, ds, dt = cell.gamma_nw, cell.hx, cell.hy
    A_prev, B_curr = cell.A_n, cell.B_w

    if precompute_propagators:
        E_s, P_s, Q_s = _get_precomputed_ops(transport_params, "left", ix)
        E_t, P_t, Q_t = _get_precomputed_ops(transport_params, "right", iy)
        _eL = lambda X: _left_apply(E_s, X)
        _p1L = lambda X: _left_apply(P_s, X)
        _p2L = lambda X: _left_apply(Q_s, X)
        _eR = lambda X: _right_apply(X, E_t)
        _p1R = lambda X: _right_apply(X, P_t)
        _p2R = lambda X: _right_apply(X, Q_t)
    else:
        _eL = lambda X: lambda_op.expm_multiply_left(ds, X, dtype=dtype)
        _p1L = lambda X: lambda_op.phi1_multiply_left(ds, X, dtype=dtype)
        _p2L = lambda X: lambda_op.phi2_multiply_left(ds, X, dtype=dtype)
        _eR = lambda X: lambda_op.expm_multiply_right(dt, X, dtype=dtype)
        _p1R = lambda X: lambda_op.phi1_multiply_right(dt, X, dtype=dtype)
        _p2R = lambda X: lambda_op.phi2_multiply_right(dt, X, dtype=dtype)

    # Predictor strip sources (increment gamma — no dt factors needed)
    H0 = _source(gij, 0.5 * (cell.eta_nw + cell.eta_n))
    J0 = _source(gij, 0.5 * (cell.eta_nw + cell.eta_w))
    dA_pred = (_eL(A_prev) - A_prev) + _p1L(H0)
    dB_pred = (_eR(B_curr) - B_curr) + _p1R(J0)

    L2K_nw = _lambda_left_right(lambda_op, cell.K_nw, dtype=dtype)
    L2K_w = _lambda_left_right(lambda_op, cell.K_w, dtype=dtype)
    L2K_n = _lambda_left_right(lambda_op, cell.K_n, dtype=dtype)

    base = cell.K_w + cell.K_n - cell.K_nw
    dsdt = _scale_like(ds * dt, L2K_w)
    K_pred = (base + 0.5 * dsdt * (L2K_w + L2K_n)
              - 0.5 * _source(gij, cell.eta_w + cell.eta_n)
              + dA_pred + dB_pred)
    eta_pred = 1.0 + jnp.sum(K_pred, axis=(-2, -1))

    # Corrected strip sources
    H1 = _source(gij, 0.5 * (cell.eta_w + eta_pred))
    J1 = _source(gij, 0.5 * (cell.eta_n + eta_pred))
    A_next = _eL(A_prev) + _p1L(H0) + _p2L(H1 - H0)
    B_next = _eR(B_curr) + _p1R(J0) + _p2R(J1 - J0)

    dA = A_next - A_prev
    dB = B_next - B_curr

    L2K_pred = _lambda_left_right(lambda_op, K_pred, dtype=dtype)
    K_se = (base + 0.25 * dsdt * (L2K_nw + L2K_w + L2K_n + L2K_pred)
            - 0.25 * _source(gij, cell.eta_nw + cell.eta_w + cell.eta_n + eta_pred)
            + dA + dB)
    eta_se = 1.0 + jnp.sum(K_se, axis=(-2, -1))
    return K_se, eta_se, A_next, B_next


_SCHEME_REGISTRY = {
    "etd1": _etd1_cell_step,
    "heun": _heun_cell_step,
}


# ---------------------------------------------------------------------------
# Solver factories
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _get_solver(*, backend, scheme, precompute_propagators, dyadic_order, terminal_only):
    cell_step = _SCHEME_REGISTRY[scheme]
    if backend == "scan":
        return _make_scan_solver(cell_step, precompute_propagators, dyadic_order, terminal_only)
    if backend == "wavefront":
        return _make_wavefront_solver(cell_step, precompute_propagators, dyadic_order, terminal_only)
    raise ValueError(f"Unknown backend {backend!r}.")


def _make_scan_solver(cell_step, precompute_propagators, dyadic_order, terminal_only):

    dyadic_order_x, dyadic_order_y = dyadic_order

    @jax.jit
    def solver(gamma, dt_x, dt_y, *, lambda_op, transport_params):
        batch, s_coarse, t_coarse, r, _ = gamma.shape
        n_x = 1 << dyadic_order_x
        n_y = 1 << dyadic_order_y
        s_steps = (s_coarse - 1) * n_x
        t_steps = (t_coarse - 1) * n_y
        t_nodes = t_steps + 1
        dtype = gamma.dtype

        zero_node = jnp.zeros((batch, r, r), dtype=dtype)
        one_node = jnp.ones((batch,), dtype=dtype)
        initial_row = RowState.zeros(batch=batch, t_nodes=t_nodes, r=r, dtype=dtype)

        def _g(i, j):
            """Index coarse gamma at fine-grid node (i, j)."""
            return gamma[:, jnp.minimum(i >> dyadic_order_x, s_coarse - 1),
                            jnp.minimum(j >> dyadic_order_y, t_coarse - 1)]

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
                    gamma_nw=_g(ix, iy), gamma_n=_g(ix, iy + 1),
                    gamma_w=_g(ix + 1, iy), gamma_se=_g(ix + 1, iy + 1),
                )
                out = cell_step(
                    lambda_op, cell, transport_params,
                    precompute_propagators=precompute_propagators,
                    ix=ix, iy=iy, dtype=dtype,
                )
                return out, out

            _, cell_hist = jax.lax.scan(
                cell_scan,
                (zero_node, one_node, zero_node, zero_node),
                jnp.arange(t_steps, dtype=jnp.int32),
            )
            south = RowState.from_scan_hist(zero_node=zero_node, one_node=one_node, hist=cell_hist)
            return south, None if terminal_only else south

        last_row, row_hist = jax.lax.scan(
            row_step, initial_row,
            (dt_x, jnp.arange(s_steps, dtype=jnp.int32)),
        )
        if terminal_only:
            return last_row.eta[:, -1], last_row.K[:, -1], last_row.A[:, -1], last_row.B[:, -1]
        grid = RowState.stack_history(initial_row, row_hist)
        return grid.eta, grid.K, grid.A, grid.B

    return solver


def _make_wavefront_solver(cell_step, precompute_propagators, dyadic_order, terminal_only):
    """Wavefront (anti-diagonal) solver using lightweight diagonal buffers."""

    dyadic_order_x, dyadic_order_y = dyadic_order

    @jax.jit
    def solver(gamma, dt_x, dt_y, *, lambda_op, transport_params):
        batch, s_coarse, t_coarse, r, _ = gamma.shape
        n_x = 1 << dyadic_order_x
        n_y = 1 << dyadic_order_y
        s_steps = (s_coarse - 1) * n_x
        t_steps = (t_coarse - 1) * n_y
        s_nodes, t_nodes = s_steps + 1, t_steps + 1
        dtype = gamma.dtype
        W = min(s_steps, t_steps) + 1
        max_int_w = max(min(s_steps, t_steps), 1)

        def _zeros_buf():
            return (
                jnp.ones((W, batch), dtype=dtype),
                jnp.zeros((W, batch, r, r), dtype=dtype),
                jnp.zeros((W, batch, r, r), dtype=dtype),
                jnp.zeros((W, batch, r, r), dtype=dtype),
            )

        buf_d0 = _zeros_buf()
        buf_d1 = _zeros_buf()

        n_diags = s_steps + t_steps - 1
        if n_diags <= 0:
            eta_grid = jnp.ones((batch, s_nodes, t_nodes), dtype=dtype)
            z = jnp.zeros((batch, s_nodes, t_nodes, r, r), dtype=dtype)
            return eta_grid, z, z.copy(), z.copy()

        d_arr = jnp.arange(2, s_steps + t_steps + 1, dtype=jnp.int32)

        def diag_step(carry, d):
            buf_prev, buf_prev2 = carry
            eta_p, K_p, A_p, B_p = buf_prev
            eta_p2, K_p2, A_p2, B_p2 = buf_prev2

            ext_lo = jnp.maximum(0, d - t_steps)
            ext_lo_p = jnp.maximum(0, d - 1 - t_steps)
            ext_lo_p2 = jnp.maximum(0, d - 2 - t_steps)

            int_lo = jnp.maximum(1, d - t_steps)
            int_hi = jnp.minimum(s_steps, d - 1)
            int_w = jnp.maximum(int_hi - int_lo + 1, 0)

            offsets = jnp.arange(max_int_w, dtype=jnp.int32)
            valid = offsets < int_w
            i_int = jnp.clip(int_lo + offsets, 1, s_steps)
            j_int = jnp.clip(d - i_int, 1, t_steps)
            im1, jm1 = i_int - 1, j_int - 1
            i_safe = jnp.where(valid, i_int, 0)
            j_safe = jnp.where(valid, j_int, 0)

            k_n = jnp.clip(im1 - ext_lo_p, 0, W - 1)
            k_w = jnp.clip(i_int - ext_lo_p, 0, W - 1)
            k_nw = jnp.clip(im1 - ext_lo_p2, 0, W - 1)
            sw = lambda a: jnp.swapaxes(a, 0, 1)
            # Index coarse gamma at fine-grid positions
            _gc = lambda i, j: gamma[:, jnp.minimum(i >> dyadic_order_x, s_coarse - 1),
                                        jnp.minimum(j >> dyadic_order_y, t_coarse - 1)]

            cells = CellData(
                hx=dt_x[im1], hy=dt_y[jm1],
                K_w=K_p[k_w], K_n=K_p[k_n], K_nw=K_p2[k_nw],
                eta_w=eta_p[k_w], eta_n=eta_p[k_n], eta_nw=eta_p2[k_nw],
                A_w=A_p[k_w], A_n=A_p[k_n], A_nw=A_p2[k_nw],
                B_w=B_p[k_w], B_n=B_p[k_n], B_nw=B_p2[k_nw],
                gamma_nw=sw(_gc(im1, jm1)), gamma_n=sw(_gc(im1, j_int)),
                gamma_w=sw(_gc(i_int, jm1)), gamma_se=sw(_gc(i_int, j_int)),
            )

            K_new, eta_new, A_new, B_new = jax.vmap(
                lambda c, ix, iy: cell_step(
                    lambda_op, c, transport_params,
                    precompute_propagators=precompute_propagators, ix=ix, iy=iy, dtype=dtype,
                ), in_axes=(0, 0, 0),
            )(cells, im1, jm1)

            new_eta, new_K, new_A, new_B = _zeros_buf()

            int_pos = jnp.clip(int_lo - ext_lo + offsets, 0, W - 1)
            vmask, vmask_rr = valid[:, None], valid[:, None, None, None]
            new_eta = new_eta.at[int_pos].set(jnp.where(vmask, eta_new, new_eta[int_pos]))
            new_K = new_K.at[int_pos].set(jnp.where(vmask_rr, K_new, new_K[int_pos]))
            new_A = new_A.at[int_pos].set(jnp.where(vmask_rr, A_new, new_A[int_pos]))
            new_B = new_B.at[int_pos].set(jnp.where(vmask_rr, B_new, new_B[int_pos]))

            return ((new_eta, new_K, new_A, new_B), buf_prev), \
                None if terminal_only else (i_safe, j_safe, valid, eta_new, K_new, A_new, B_new)

        (last_buf, _), all_out = jax.lax.scan(diag_step, (buf_d1, buf_d0), d_arr)

        if terminal_only:
            # Terminal cell (s_steps, t_steps) is always the last element of the last diagonal buffer
            eta_last, K_last, A_last, B_last = last_buf
            # The terminal cell is at position 0 in the last diagonal's buffer
            # (last diagonal has width 1 containing only the terminal cell)
            return eta_last[0], K_last[0], A_last[0], B_last[0]

        i_all, j_all, valid_all, eta_all, K_all, A_all, B_all = all_out

        # Reconstruct full grid
        eta_grid = jnp.ones((batch, s_nodes, t_nodes), dtype=dtype)
        z = jnp.zeros((batch, s_nodes, t_nodes, r, r), dtype=dtype)
        K_grid, A_grid, B_grid = z, z.copy(), z.copy()

        i_flat, j_flat, v_flat = i_all.reshape(-1), j_all.reshape(-1), valid_all.reshape(-1)
        mv = lambda a: jnp.moveaxis(a.reshape(-1, *a.shape[2:]), 0, 1)
        eta_flat, K_flat, A_flat, B_flat = mv(eta_all), mv(K_all), mv(A_all), mv(B_all)

        vmask, vmask_rr = v_flat[None, :], v_flat[None, :, None, None]
        eta_grid = eta_grid.at[:, i_flat, j_flat].set(jnp.where(vmask, eta_flat, eta_grid[:, i_flat, j_flat]))
        K_grid = K_grid.at[:, i_flat, j_flat].set(jnp.where(vmask_rr, K_flat, K_grid[:, i_flat, j_flat]))
        A_grid = A_grid.at[:, i_flat, j_flat].set(jnp.where(vmask_rr, A_flat, A_grid[:, i_flat, j_flat]))
        B_grid = B_grid.at[:, i_flat, j_flat].set(jnp.where(vmask_rr, B_flat, B_grid[:, i_flat, j_flat]))

        return eta_grid, K_grid, A_grid, B_grid

    return solver


# ---------------------------------------------------------------------------
# Shared path/gamma helpers
# ---------------------------------------------------------------------------


def _build_gamma_grid(dX, dY, *, proj, pairwise):
    """Build the increment-based gamma grid from path increments."""
    sx = _project_increments(dX, proj=proj)
    sy = _project_increments(dY, proj=proj)

    if pairwise:
        bx, by = int(dX.shape[0]), int(dY.shape[0])
        gamma = jnp.einsum("biRm,cjSm->bcijRS", sx, sy)
        return gamma.reshape((bx * by,) + gamma.shape[-4:]), (bx, by)

    bx, by = int(dX.shape[0]), int(dY.shape[0])
    if bx != by:
        raise ValueError(f"Batchwise evaluation requires matching batch sizes; got {bx} and {by}.")
    return jnp.einsum("biRm,bjSm->bijRS", sx, sy), (bx,)


def _state_projection_tensor(kernel, *, dtype):
    return jnp.einsum("qR,qmd->Rmd", kernel.b.astype(dtype), kernel.A.astype(dtype))


def _project_increments(dX, *, proj):
    """Project increments and repeat last to get node-aligned values."""
    nodes = jnp.concatenate([dX, dX[:, -1:, :]], axis=-2)
    return jnp.einsum("Rmd,bid->biRm", proj, nodes)


def _to_increments(X, *, path_dim, name, increment_in):
    """Normalize input to (batch, intervals, dim) increments."""
    X = jnp.asarray(X)
    if X.ndim == 2:
        X = X[None, ...]
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (batch, length, dim) or (length, dim).")
    if X.shape[-1] != path_dim:
        raise ValueError(f"{name} must have terminal feature dimension {path_dim}, got {X.shape[-1]}.")
    if increment_in:
        if X.shape[-2] < 1:
            raise ValueError(f"{name} must contain at least one interval.")
        return X
    if X.shape[-2] < 2:
        raise ValueError(f"{name} must contain at least two time points.")
    return jnp.diff(X, axis=-2)




def _prepare_dt(dt, steps, *, name, dtype):
    dt_arr = jnp.asarray(dt, dtype=dtype)
    if dt_arr.ndim == 0:
        return jnp.full((steps,), dt_arr, dtype=dtype), True
    if dt_arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array, got shape {tuple(dt_arr.shape)}.")
    if int(dt_arr.shape[0]) != steps:
        raise ValueError(f"{name} must have length {steps}, got {dt_arr.shape[0]}.")
    return dt_arr, False



def _lambda_left_right(lambda_op, X, *, dtype):
    return lambda_op.lambda_multiply_right(lambda_op.lambda_multiply_left(X, dtype=dtype), dtype=dtype)


__all__ = ["fssk_sigkernel", "FSSKSigKernel", "CellData", "RowState", "WaveState"]
