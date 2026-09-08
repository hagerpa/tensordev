"""Finite-state-space approximations of fractional Volterra kernels.

For ``beta`` in ``(1/2, 1)``, the fractional kernel has the Laplace
formula

    t ** (beta - 1) / Gamma(beta)
        = sin(pi * beta) / pi
          * integral(exp(-x * t) * x ** (-beta), x=0..infinity).

Replacing the measure by positive point masses gives a finite exponential
sum and therefore a finite-dimensional state-space kernel. The implementation
below selects logarithmically spaced decay rates and fits nonnegative weights
on a logarithmic time grid.

SciPy is an optional dependency because constructing the approximation is a
one-time host-side optimization. Install it with ``tensordev[rough]``.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
    """Build an FSSK approximation of a fractional kernel.

    Parameters
    ----------
    beta:
        Fractional exponent parameter in ``(1/2, 1)``. The target kernel is
        ``t ** (beta - 1) / Gamma(beta)``.
    R:
        Number of exponential factors and state-space dimension.
    A:
        Kernel matrices with shape ``(q, m, d)``.
    T:
        Approximation horizon. Default is ``1.0``.
    coef_quad_order:
        Contour quadrature order used later by :meth:`FSSK.coef`.
    dtype:
        Optional dtype for the returned FSSK arrays.

    Returns
    -------
    FSSK
        Finite-state-space kernel with diagonal state matrix satisfying
        ``1^T exp(-Lambda t) b[p] ~= K_beta(t)`` for every component ``p``.
    """
    beta = float(beta)
    R = int(R)
    T = float(T)
    _validate_beta_R_T(beta=beta, R=R, T=T)

    A_arr = jnp.asarray(A)
    if A_arr.ndim != 3:
        raise ValueError(
            "A must have shape (q, m, d); "
            f"got shape {tuple(A_arr.shape)}."
        )

    real_dtype = jnp.dtype(dtype or A_arr.dtype)
    A_arr = A_arr.astype(real_dtype)
    nodes, weights = _fractional_exponential_rule(
        beta=beta,
        R=R,
        T=T,
    )
    nodes_arr = jnp.asarray(nodes, dtype=real_dtype)
    weights_arr = jnp.asarray(weights, dtype=real_dtype)
    q = int(A_arr.shape[0])

    return FSSK.from_jordan(
        real_rates=nodes_arr,
        real_sizes=(1,) * R,
        A=A_arr,
        b=jnp.broadcast_to(weights_arr[None, :], (q, R)),
        quad_order=int(coef_quad_order),
    )


def _fractional_exponential_rule(
    *,
    beta: float,
    R: int,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a positive ``R``-factor approximation on ``[0.001 T, T]``.

    The dimensionless decay rates cover the time scales resolved by the fit.
    Dividing the rates by ``T`` and multiplying the weights by
    ``T ** (beta - 1)`` then gives the rule for an arbitrary horizon.
    """
    beta = float(beta)
    R = int(R)
    T = float(T)
    _validate_beta_R_T(beta=beta, R=R, T=T)
    scipy_optimize = _require_scipy_optimize()

    unit_nodes = np.geomspace(0.1, 1000.0, R, dtype=np.float64)
    sample_count = max(256, 64 * R)
    unit_times = np.geomspace(1e-3, 1.0, sample_count, dtype=np.float64)
    target = unit_times ** (beta - 1.0) / math.gamma(beta)

    design = np.exp(-np.outer(unit_times, unit_nodes))
    relative_design = design / target[:, None]
    result = scipy_optimize.lsq_linear(
        relative_design,
        np.ones(sample_count, dtype=np.float64),
        bounds=(0.0, np.inf),
        tol=1e-12,
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(
            "The fractional-kernel nonnegative least-squares fit failed: "
            f"{result.message}"
        )

    nodes = unit_nodes / T
    weights = np.asarray(result.x) * T ** (beta - 1.0)
    return nodes, weights


def _require_scipy_optimize() -> Any:
    """Import the optional host-side optimizer with an actionable error."""
    try:
        from scipy import optimize as scipy_optimize
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "fractional_fssk requires SciPy; install tensordev[rough]."
        ) from error
    return scipy_optimize


def _validate_beta_R_T(*, beta: float, R: int, T: float) -> None:
    if not (0.5 < float(beta) < 1.0):
        raise ValueError(
            "beta must lie in (1/2, 1); "
            f"got beta={beta}."
        )
    if int(R) <= 0:
        raise ValueError(f"R must be positive, got {R}.")
    if float(T) <= 0.0:
        raise ValueError(f"T must be positive, got {T}.")


__all__ = ["fractional_fssk"]
