"""BL2 finite-state-space approximations of fractional Volterra kernels.

This module constructs an FSSK approximation of the fractional kernel

    K_beta(t) = t ** (beta - 1) / Gamma(beta),

for beta in (1/2, 1), using the positive-Hurst BL2/european approximation
path from RoughKernel.quadrature_rule.

The relation to the rough-kernel notation is

    H = beta - 1/2,

so that

    K_H(t) = t ** (H - 1/2) / Gamma(H + 1/2)
           = t ** (beta - 1) / Gamma(beta).

Courtesy
--------
The BL2 approximation logic is adapted from the public implementation

    https://github.com/SimonBreneis/approximations_to_fractional_stochastic_volterra_equations/

in particular the positive-Hurst european/BL2 branch of RoughKernel.quadrature_rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.special import gamma, gammainc

from tensordev.sss.kernel import FSSK

Array = jax.Array


def fractional_fssk(
    *,
    beta: float,
    R: int,
    A: Array,
    T: float = 1.0,
    coef_quad_order: int = 32,
    dtype: jnp.dtype | None = None,
) -> FSSK:
    """Build an FSSK approximation of the fractional kernel.

    Parameters
    ----------
    beta:
        Fractional exponent parameter in ``(1/2, 1)``. The target kernel is

            K_beta(t) = t ** (beta - 1) / Gamma(beta).

    R:
        Number of exponential factors in the BL2 approximation. This is also
        the state-space dimension of the returned FSSK.
    A:
        Kernel matrices with shape ``(n, m, d)``.
    T:
        Approximation horizon. Default is ``1.0``.
    coef_quad_order:
        Contour quadrature order used later by ``FSSK.coef``. This is distinct
        from ``R``.
    dtype:
        Optional dtype for the returned FSSK arrays.

    Returns
    -------
    FSSK
        Finite-state-space kernel with diagonal state matrix such that

            1^T exp(-Lambda t) b[p] ~= K_beta(t)

        for every component ``p``.
    """
    beta = float(beta)
    R = int(R)
    T = float(T)

    _validate_beta_R_T(beta=beta, R=R, T=T)

    A_arr = jnp.asarray(A)
    if A_arr.ndim != 3:
        raise ValueError(
            "A must have shape (n, m, d); "
            f"got shape {tuple(A_arr.shape)}."
        )

    real_dtype = jnp.dtype(dtype or A_arr.dtype)
    A_arr = A_arr.astype(real_dtype)

    nodes, weights = _bl2_quadrature_rule(beta=beta, R=R, T=T)
    nodes_arr = jnp.asarray(nodes, dtype=real_dtype)
    weights_arr = jnp.asarray(weights, dtype=real_dtype)

    q = int(A_arr.shape[0])
    b = jnp.broadcast_to(weights_arr[None, :], (q, R))

    return FSSK.from_jordan(
        real_rates=nodes_arr,
        real_sizes=(1,) * R,
        A=A_arr,
        b=b,
        quad_order=int(coef_quad_order),
    )


def _bl2_quadrature_rule(
    *,
    beta: float,
    R: int,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return BL2/european exponential nodes and weights.

    This is the minimal positive-Hurst path corresponding to

        RoughKernel.quadrature_rule(H, R, T, mode="european")

    with ``H = beta - 1/2``.
    """
    H = float(beta) - 0.5
    nodes, weights = _european_rule(H=H, R=int(R), T=float(T))
    weights[np.logical_and(nodes < 1.0, np.abs(weights) > 100.0)] = 0.0
    return _sort_rule(nodes, weights)


def _european_rule(
    *,
    H: float,
    R: int,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive-Hurst european/BL2 rule from RoughKernel.py."""
    H = float(H)
    R = int(R)
    T = float(T)

    last_nodes: np.ndarray

    def optimizing_func(R_: int, tol_: float, bound_: float | None):
        if R_ == 1:
            nod = np.array([1.0 / T], dtype=np.float64)
        elif len(last_nodes) == R_:
            nod = last_nodes
        else:
            nod = np.empty(R_, dtype=np.float64)
            nod[:-1] = last_nodes
            nod[-1] = float(bound_)

        nod = nod / 1.03 ** np.fmin(np.arange(1, R_ + 1) ** 2, 100)
        return _optimize_error_l2(
            H=H,
            R=R_,
            T=T,
            tol=tol_,
            bound=bound_,
            init_nodes=nod,
        )

    _, nodes, weights = optimizing_func(R_=1, tol_=1e-6, bound_=None)
    if R == 1:
        return nodes, weights

    L_step = 1.15
    bound = float(np.amax(nodes) / L_step)
    current_R = 1
    last_nodes = nodes

    while current_R < R:
        increase_R = 0
        L_step = 1.15

        while increase_R < 2:
            bound = bound * L_step
            error_, nodes, weights = optimizing_func(
                R_=current_R + 1,
                tol_=1e-7 / current_R,
                bound_=bound,
            )

            p = np.argsort(nodes)
            nodes = nodes[p]
            weights = weights[p]

            if (
                np.amin(nodes[1:] / nodes[:-1]) < 1.4
                or np.abs(np.amin(weights)) < 1e-2
                or np.abs(np.amin(weights[1:] / weights[:-1])) < 0.4
            ):
                increase_R = 0
                L_step = 1.15
            elif error_ < optimizing_func(
                R_=current_R,
                tol_=1e-7 / current_R,
                bound_=bound,
            )[0]:
                increase_R += 1
                if L_step > 1.06:
                    L_step = 1.05
                    bound = bound / 1.15
            else:
                increase_R = 0
                L_step = 1.15

        current_R += 1
        last_nodes = nodes

    if R >= 4:
        return nodes, weights

    if R == 2:
        L_4 = bound * 2.0
        L_5 = bound * 3.0
        L_6 = bound * 4.0
    else:
        L_4 = bound
        L_5 = bound * 1.25
        L_6 = bound * 1.5

    error_4, nodes_4, weights_4 = optimizing_func(R_=R, tol_=1e-8, bound_=L_4)
    error_5, nodes_5, weights_5 = optimizing_func(R_=R, tol_=1e-8, bound_=L_5)
    error_6, nodes_6, weights_6 = optimizing_func(R_=R, tol_=1e-8, bound_=L_6)

    if error_4 <= error_5 and error_4 <= error_6:
        return nodes_4, weights_4
    if error_5 <= error_6:
        return nodes_5, weights_5
    return nodes_6, weights_6


def _optimize_error_l2(
    *,
    H: float,
    R: int,
    T: float,
    tol: float = 1e-8,
    bound: float | None = None,
    init_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize the L2 error over nodes, using optimal weights."""
    H = float(H)
    R = int(R)
    T = float(T)

    if bound is None:
        bound = 1e100

    nodes = np.asarray(init_nodes, dtype=np.float64)
    if nodes.shape != (R,):
        raise ValueError(
            f"init_nodes must have shape ({R},), got {nodes.shape}."
        )

    lower_bound = 1.0 / (10.0 * R * T) * ((0.5 - H) / 0.4) ** 2
    nodes = np.fmin(np.fmax(nodes, lower_bound), bound)

    bounds = ((np.log(lower_bound), np.log(bound)),) * R
    original_error, original_weights = _error_l2_optimal_weights(
        H=H,
        T=T,
        nodes=nodes,
        output="error",
    )
    original_nodes = nodes.copy()

    def func(x):
        err, grad, _ = _error_l2_optimal_weights(
            H=H,
            T=T,
            nodes=np.exp(x),
            output="gradient",
        )
        return err, np.exp(x) * grad

    res = minimize(
        func,
        np.log(nodes),
        tol=tol**2,
        bounds=bounds,
        jac=True,
    )

    nodes = np.exp(res.x)
    err, weights = _error_l2_optimal_weights(
        H=H,
        T=T,
        nodes=nodes,
        output="error",
    )

    if err > 2.0 * np.fmax(original_error, 1e-9):
        return (
            np.array([np.sqrt(np.fmax(original_error, 0.0))]),
            original_nodes,
            original_weights,
        )

    return np.array([np.sqrt(np.fmax(err, 0.0))]), nodes, weights


def _error_l2_optimal_weights(
    *,
    H: float,
    T: float,
    nodes: np.ndarray,
    output: str = "error",
):
    """L2 error and optimal weights for fixed nodes.

    This is the positive-Hurst scalar-T branch needed by BL2/european.
    """
    H = float(H)
    T = float(T)
    nodes = np.asarray(nodes, dtype=np.float64)

    if len(nodes) == 1:
        node = np.fmax(1e-4, nodes[0])
        gamma_1 = gamma(H + 0.5)

        nT = node * T
        gamma_ints = gammainc(H + 0.5, nT)
        exp_node_matrix = _exp_underflow(2.0 * nT)
        exp_node_vec = _exp_underflow(nT)

        A = (1.0 - exp_node_matrix) / (2.0 * node)
        b = -2.0 * gamma_ints / node ** (H + 0.5)
        c = T ** (2.0 * H) / (2.0 * H * gamma_1**2)

        v = b / A
        err = c - 0.25 * b * v
        opt_weight = np.array([-0.5 * v])

        if output in {"error", "err"}:
            return err, opt_weight

        A_grad = (-1.0 + (1.0 + 2.0 * nT) * exp_node_matrix) / (
            4.0 * node**2
        )
        b_grad = (
            -2.0
            * (
                nT ** (H + 0.5) * exp_node_vec / gamma_1
                - (H + 0.5) * gamma_ints
            )
            / node ** (H + 1.5)
        )
        grad = 0.5 * (A_grad * v - b_grad) * v

        if output in {"gradient", "grad"}:
            return err, grad, opt_weight

        raise NotImplementedError(f"Unsupported output={output!r}.")

    nodes = _regularize_nodes(nodes)

    node_matrix = nodes[:, None] + nodes[None, :]
    gamma_1 = gamma(H + 0.5)

    nT = nodes * T
    nmT = node_matrix * T
    gamma_ints = gammainc(H + 0.5, nT)
    exp_node_matrix = _exp_underflow(nmT)

    A = (1.0 - exp_node_matrix) / node_matrix
    b = -2.0 * gamma_ints / nodes ** (H + 0.5)
    c = T ** (2.0 * H) / (2.0 * H * gamma_1**2)

    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(A, b, rcond=None)[0]

    if np.amax(v) > 0.0:
        v = lsq_linear(A, b).x

    err = 0.25 * v @ A @ v - 0.5 * np.dot(b, v) + c
    opt_weights = -0.5 * v

    if output in {"error", "err"}:
        return err, opt_weights

    exp_node_vec = _exp_underflow(nT)
    A_grad = (-1.0 + (1.0 + nmT) * exp_node_matrix) / node_matrix**2
    b_grad = (
        -2.0
        * (
            nT ** (H + 0.5) * exp_node_vec / gamma_1
            - (H + 0.5) * gamma_ints
        )
        / nodes ** (H + 1.5)
    )
    grad = 0.5 * v * (A_grad @ v) - 0.5 * b_grad * v

    if output in {"gradient", "grad"}:
        return err, grad, opt_weights

    raise NotImplementedError(f"Unsupported output={output!r}.")


def _regularize_nodes(nodes: np.ndarray) -> np.ndarray:
    """Sort-copy regularization used by RoughKernel's L2 optimizer."""
    nodes = np.asarray(nodes, dtype=np.float64)

    perm = np.argsort(nodes)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)

    sorted_nodes = nodes[perm].copy()
    sorted_nodes[0] = np.fmax(1e-4, sorted_nodes[0])

    for i in range(len(sorted_nodes) - 1):
        if 1.01 * sorted_nodes[i] > sorted_nodes[i + 1]:
            sorted_nodes[i + 1] = sorted_nodes[i] * 1.01

    return sorted_nodes[inv]


def _exp_underflow(x):
    """Compute exp(-x) with large-x underflow protection."""
    if isinstance(x, np.ndarray):
        if x.dtype == int:
            x = x.astype(float)
        eps = np.finfo(x.dtype).tiny
    else:
        if isinstance(x, int):
            x = float(x)
        eps = np.finfo(x.__class__).tiny

    log_eps = -np.log(eps) / 2.0
    result = np.exp(-np.fmin(x, log_eps))
    return np.where(x > log_eps, 0.0, result)


def _sort_rule(
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort nodes and weights jointly by increasing node."""
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if nodes.ndim != 1:
        raise ValueError(f"nodes must be one-dimensional, got {nodes.shape}.")
    if weights.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got {weights.shape}.")
    if nodes.shape != weights.shape:
        raise ValueError(
            "nodes and weights must have matching shapes; "
            f"got nodes.shape={nodes.shape}, weights.shape={weights.shape}."
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if np.any(nodes < 0.0):
        raise ValueError("nodes must be non-negative.")

    perm = np.argsort(nodes)
    return nodes[perm], weights[perm]


def _validate_beta_R_T(*, beta: float, R: int, T: float) -> None:
    if not (0.5 < float(beta) < 1.0):
        raise ValueError(
            "beta must lie in (1/2, 1) for the positive-Hurst BL2 rule; "
            f"got beta={beta}."
        )
    if int(R) <= 0:
        raise ValueError(f"R must be positive, got {R}.")
    if float(T) <= 0.0:
        raise ValueError(f"T must be positive, got {T}.")


__all__ = ["fractional_fssk"]