"""Volterra kernels and packed coefficient builders.

The public names :class:`FractionalKernel` and :class:`GammaKernel` both
inherit from the abstract base :class:`ConvolutionKernel`.  The base class holds
the shared kernel matrix ``A`` and implements :meth:`coef`, :meth:`coef_grid`,
and :meth:`lag_weights`; concrete subclasses implement :meth:`alpha`.

The coefficient methods return :class:`VolterraCoefficients`, whose
``alpha[..., p, ell_idx]`` stores

.. math::
    \\mathcal{K}_{s,t}^{w(\\ell)\\,p,\\,\\tau} \\,/\\, (t-s)^{|\\ell|+1}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln

from tensordev.util.combinatorics import MultiIndexLayout, build_multiindex_layout
from tensordev.volterra.coeffs import VolterraCoefficients


Array = jax.Array


@dataclass(frozen=True, slots=True)
class ConvolutionKernel:
    r"""Abstract base class for Volterra convolution kernels.

    A Volterra kernel is represented as

    .. math::
        K(t,s) = \sum_{p=1}^n k_p(t-s)\, A_p,

    where :math:`A_p \in \mathbb{R}^{m \times d}` and :math:`\beta_p > 0` is
    the fractional order of :math:`k_p` (:math:`\beta_p = 1` for smooth kernels).
    Concrete subclasses implement :meth:`alpha`.
    """

    A: Array
    beta: Array

    def __post_init__(self) -> None:
        A = jnp.asarray(self.A)
        if A.ndim != 3:
            raise ValueError(
                "A must have shape (n, m, d); "
                f"got {tuple(A.shape)}."
            )
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "beta", jnp.asarray(self.beta))

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def fractional(cls, *, beta: Array, A: Array) -> "FractionalKernel":
        r"""Construct a multivariate fractional Volterra kernel.

        The scalar kernels are

        .. math::
            k_p(t,s) = \frac{(t-s)^{\beta_p - 1}}{\Gamma(\beta_p)},
            \qquad p = 1,\ldots,n,

        where :math:`\beta_p > 0`.

        Parameters
        ----------
        beta : Array
            Exponent vector of shape ``(n,)`` with all entries positive.
        A : Array
            Kernel matrices of shape ``(n, m, d)``.
        """
        return FractionalKernel(beta=beta, A=A)

    @classmethod
    def gamma(
        cls,
        *,
        beta: Array,
        A: Array,
        scale: Array = 1.0,
        rate: Array = 1.0,
        quad_order: int = 32,
    ) -> "GammaKernel":
        r"""Construct a scalar Gamma Volterra kernel (:math:`n = 1`).

        The kernel is

        .. math::
            k(t,s) = \mathrm{scale} \cdot e^{-\mathrm{rate}(t-s)}
                     \cdot \frac{(t-s)^{\beta-1}}{\Gamma(\beta)},

        where :math:`\beta > 0`, :math:`\mathrm{scale} > 0` and
        :math:`\mathrm{rate} \geq 0`.

        Parameters
        ----------
        beta : Array
            Shape parameter, scalar or shape ``(1,)``.  Must be positive.
        A : Array
            Kernel matrix of shape ``(1, m, d)``.
        scale : Array, default=1.0
            Positive scale factor.
        rate : Array, default=1.0
            Non-negative exponential decay rate.
        quad_order : int, default=32
            Number of Gauss-Legendre nodes used when building coefficients.
        """
        return GammaKernel(beta=beta, A=A, scale=scale, rate=rate, quad_order=quad_order)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q(self) -> int:
        """Number of scalar kernel components."""
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        """Latent output dimension of each ``A_p``."""
        return int(self.A.shape[1])

    @property
    def path_dim(self) -> int:
        """Input path dimension ``d``."""
        return int(self.A.shape[2])

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def alpha(
        self,
        layout: MultiIndexLayout,
        *,
        rho: float | Array = 0.0,
        dtype: jnp.dtype,
        s: Array,
        t: Array,
        tau: Array,
    ) -> tuple[Array, Array]:
        """Evaluate normalized kernel coefficients for time triples ``(s, t, tau)``.

        Returns ``(vals, valid)`` where ``vals`` has shape ``leading + (n, M)``
        and ``valid`` has shape ``leading``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement alpha.")

    # ------------------------------------------------------------------
    # Shared coefficient builders
    # ------------------------------------------------------------------

    def coef(
        self,
        s: Array,
        t: Array,
        tau: Array,
        *,
        trunc: int,
        rho: float | Array = 0.0,
        dtype: Optional[jnp.dtype] = None,
    ) -> VolterraCoefficients:
        r"""Build packed coefficients for broadcasted triples ``(s, t, tau)``.

        ``s``, ``t`` and ``tau`` are broadcast to a common leading shape.  The
        returned ``alpha`` has shape ``leading + (n, M)``.  Triples outside
        ``s < t <= tau`` are marked invalid and their coefficients are zero.

        Parameters
        ----------
        rho:
            Basis exponent for higher-order schemes.  ``rho=0`` (default)
            recovers the standard Euler coefficients.
        """
        if trunc <= 0:
            raise ValueError(f"trunc must be positive, got {trunc}.")

        real_dtype = jnp.dtype(dtype or self.A.dtype)
        s_arr, t_arr, tau_arr = jnp.broadcast_arrays(
            jnp.asarray(s, dtype=real_dtype),
            jnp.asarray(t, dtype=real_dtype),
            jnp.asarray(tau, dtype=real_dtype),
        )
        layout = build_multiindex_layout(self.q, trunc - 1)

        alpha, valid = self.alpha(layout, rho=rho, dtype=real_dtype, s=s_arr, t=t_arr, tau=tau_arr)

        return VolterraCoefficients(
            layout=layout,
            trunc=trunc,
            m=self.m,
            q=self.q,
            alpha=alpha.astype(real_dtype),
            valid=valid,
        )

    def coef_grid(
        self,
        times: Array,
        *,
        trunc: int,
        rho: float | Array = 0.0,
        tau: Optional[Array] = None,
        dtype: Optional[jnp.dtype] = None,
    ) -> VolterraCoefficients:
        r"""Build source/readout-grid coefficients.

        Parameters
        ----------
        times:
            Path grid of shape ``(S + 1,)``.  Source interval ``i`` is
            ``[times[i], times[i+1]]``.
        rho:
            Basis exponent for higher-order schemes.  ``rho=0`` (default)
            recovers the standard Euler coefficients.
        tau:
            Optional readout times of shape ``(R,)``.  If omitted,
            ``tau = times[1:]`` and the leading coefficient shape is ``(S, S)``.
        trunc:
            Tensor truncation level.

        Returns
        -------
        VolterraCoefficients
            Coefficients with leading shape ``(S, R)``.
        """
        if trunc <= 0:
            raise ValueError(f"trunc must be positive, got {trunc}.")

        real_dtype = jnp.dtype(dtype or self.A.dtype)
        times_arr = jnp.asarray(times, dtype=real_dtype)
        if times_arr.ndim != 1:
            raise ValueError(f"times must be one-dimensional, got shape {tuple(times_arr.shape)}.")
        if times_arr.shape[0] < 2:
            raise ValueError("times must contain at least two nodes.")

        s = times_arr[:-1]
        t = times_arr[1:]
        tau_arr = times_arr[1:] if tau is None else jnp.asarray(tau, dtype=real_dtype)
        if tau_arr.ndim != 1:
            raise ValueError(f"tau must be one-dimensional when provided, got {tuple(tau_arr.shape)}.")

        return self.coef(
            s[:, None],
            t[:, None],
            tau_arr[None, :],
            trunc=trunc,
            rho=rho,
            dtype=real_dtype,
        )

    def lag_weights(
        self,
        *,
        out_len: int,
        h: Array,
        theta: Array,
        n: int,
        rho: float | Array = 0.0,
        dtype: jnp.dtype,
    ) -> Array:
        r"""Lag-k causal convolution weights for ``k = 0, ..., out_len-1``.

        Evaluates the degree-``(n-1)`` Volterra coefficients at the time triples
        ``s=0, t=h, tau=(k+θ)·h``, which are the causal convolution weights for
        the order-``n`` term of the discrete signature kernel.  Lag 0 is always
        zero (strict causality, ``tau < t``).

        Returns an array of shape ``(out_len, self.q, M_{n-1})``, where
        ``M_{n-1}`` is the number of multi-indices of degree ``n-1`` for
        ``self.n`` components.
        """
        dtype_ = jnp.dtype(dtype)
        lag = jnp.arange(out_len, dtype=dtype_)
        layout = build_multiindex_layout(self.q, n - 1)
        vals, _ = self.alpha(
            layout,
            rho=rho,
            dtype=dtype_,
            s=jnp.zeros(out_len, dtype=dtype_),
            t=jnp.full(out_len, h, dtype=dtype_),
            tau=(lag + theta) * h,
        )
        start = int(layout.offsets[n - 1])
        result = vals[..., start:].astype(dtype_)
        # Strict causality: lag 0 is zero regardless of theta.  alpha uses
        # tau >= t which admits tau = t when theta = 1, but that is not a
        # past lag and must be zeroed out explicitly.
        return result.at[0].set(jnp.zeros_like(result[0]))


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class FractionalKernel(ConvolutionKernel):
    r"""Multivariate fractional Volterra kernel.

    The scalar kernels are

    .. math::
        k_p(t,s) = \frac{(t-s)^{\beta_p - 1}}{\Gamma(\beta_p)},
        \qquad p = 1,\ldots,n,

    where :math:`\beta_p > 0` is the fractional order of :math:`k_p`
    (in particular :math:`\beta_p = 1` gives the flat kernel
    :math:`k_p \equiv 1`).  The full kernel is
    :math:`K(t,s) = \sum_p k_p(t,s)\,A_p`.

    Parameters
    ----------
    beta : Array
        Fractional-order vector of shape ``(n,)`` with all entries positive
        (:math:`\beta_p = 1` recovers the flat/classical case).
    A : Array
        Kernel matrices of shape ``(n, m, d)``.
    """

    def __post_init__(self) -> None:
        ConvolutionKernel.__post_init__(self)
        if self.beta.shape != (self.q,):
            raise ValueError(
                "fractional beta must have shape (n,), matching A.shape[0]; "
                f"got beta.shape={tuple(self.beta.shape)} and n={self.q}."
            )
        if bool(jnp.any(self.beta <= 0)):
            raise ValueError("fractional beta entries must be positive.")

    def alpha(
        self,
        layout: MultiIndexLayout,
        *,
        rho: float | Array = 0.0,
        dtype: jnp.dtype,
        s: Array,
        t: Array,
        tau: Array,
    ) -> tuple[Array, Array]:
        dtype_ = jnp.dtype(dtype)
        return _fractional_alpha(
            s.astype(dtype_),
            t.astype(dtype_),
            tau.astype(dtype_),
            self.beta.astype(dtype_),
            layout.ell.astype(dtype_),
            layout.degree.astype(dtype_),
            rho,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class GammaKernel(ConvolutionKernel):
    r"""Scalar Gamma Volterra kernel (:math:`n = 1`).

    The kernel is

    .. math::
        k(t,s) = \mathrm{scale} \cdot e^{-\mathrm{rate}(t-s)}
                 \cdot \frac{(t-s)^{\beta-1}}{\Gamma(\beta)},

    where :math:`\beta > 0`, :math:`\mathrm{scale} > 0` and
    :math:`\mathrm{rate} \geq 0`.

    Parameters
    ----------
    beta : Array
        Shape parameter, scalar or shape ``(1,)``.  Must be positive.
    scale : Array
        Positive scale factor.
    rate : Array
        Non-negative exponential decay rate.
    A : Array
        Kernel matrix of shape ``(1, m, d)``.
    quad_order : int, default=32
        Number of Gauss-Legendre nodes used when building coefficients.
    """

    scale: Array
    rate: Array
    quad_order: int = field(default=32, metadata={"static": True})

    def __post_init__(self) -> None:
        ConvolutionKernel.__post_init__(self)
        scale = jnp.asarray(self.scale)
        rate = jnp.asarray(self.rate)
        if self.q != 1:
            raise ValueError(
                "gamma kernels are scalar in this implementation, so A.shape[0] must be 1."
            )
        if self.beta.shape not in [(), (1,)]:
            raise ValueError(f"gamma beta must be scalar or shape (1,), got {tuple(self.beta.shape)}.")
        if scale.shape not in [(), (1,)]:
            raise ValueError(f"gamma scale must be scalar or shape (1,), got {tuple(scale.shape)}.")
        if rate.shape not in [(), (1,)]:
            raise ValueError(f"gamma rate must be scalar or shape (1,), got {tuple(rate.shape)}.")
        if bool(jnp.any(self.beta <= 0)):
            raise ValueError("gamma beta must be positive.")
        if bool(jnp.any(scale <= 0)):
            raise ValueError("gamma scale must be positive.")
        if bool(jnp.any(rate < 0)):
            raise ValueError("gamma rate must be non-negative.")
        if self.quad_order <= 0:
            raise ValueError(f"quad_order must be positive, got {self.quad_order}.")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rate", rate)

    def alpha(
        self,
        layout: MultiIndexLayout,
        *,
        rho: float | Array = 0.0,
        dtype: jnp.dtype,
        s: Array,
        t: Array,
        tau: Array,
    ) -> tuple[Array, Array]:
        dtype_ = jnp.dtype(dtype)
        nodes_np, weights_np = np.polynomial.legendre.leggauss(int(self.quad_order))
        return _gamma_alpha(
            s.astype(dtype_),
            t.astype(dtype_),
            tau.astype(dtype_),
            self.beta.reshape(()).astype(dtype_),
            self.scale.reshape(()).astype(dtype_),
            self.rate.reshape(()).astype(dtype_),
            layout.degree.astype(dtype_),
            jnp.asarray(nodes_np, dtype=dtype_),
            jnp.asarray(weights_np, dtype=dtype_),
            rho,
        )

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln


@jax.jit
def _fractional_alpha(
    s: Array,
    t: Array,
    tau: Array,
    beta: Array,
    ell: Array,
    degree: Array,
    rho: float | Array,
) -> tuple[Array, Array]:
    r"""Evaluate normalized multivariate fractional coefficients.

    For each multi-index ``m`` (row of ``ell``) and component ``j``, computes

    .. math::

        \alpha_{m,j}(s, t, \tau) =
            \frac{\Gamma(\rho+1)}{\Gamma\!\left(\rho + \ell_m\!\cdot\!\beta + \beta_j + 1\right)}
            \cdot
            \frac{(\tau - s)^{\rho + \ell_m\cdot\beta + \beta_j}}{(t - s)^{|\ell_m| + 1}}
            \cdot
            I_{(t-s)/(\tau-s)}\!\left(\rho + \ell_m\!\cdot\!\beta + 1,\; \beta_j\right),

    where :math:`I_z(a, b)` is the regularized incomplete beta function.
    All inputs are assumed pre-cast to a common floating-point dtype by the
    calling :meth:`FractionalKernel.alpha`.  The result is zero whenever
    :math:`s < t \le \tau` is violated.

    Returns ``(vals, valid)`` with ``vals`` of shape ``leading + (n, M)``.
    """
    h = t - s
    tau_s = tau - s
    valid = (h > 0) & (tau >= t)
    h_safe = jnp.where(valid, h, 1.0)
    tau_s_safe = jnp.where(valid, tau_s, 1.0)
    z = jnp.clip(h_safe / tau_s_safe, 0.0, 1.0)

    prefix = ell @ beta                      # (M,)
    total = prefix[:, None] + beta[None, :]  # (M, n)
    a = rho + prefix[:, None] + 1.0
    b = beta[None, :]

    # leading + (M, n)
    log_scale = (
        (rho + total) * jnp.log(tau_s_safe[..., None, None])
        - (degree[:, None] + 1.0) * jnp.log(h_safe[..., None, None])
        + gammaln(rho + 1.0)
        - gammaln(rho + total + 1.0)
    )
    vals = jnp.exp(log_scale) * betainc(a, b, z[..., None, None])
    vals = jnp.where(valid[..., None, None], vals, 0.0)
    return jnp.swapaxes(vals, -1, -2), valid  # leading + (n, M)


@jax.jit
def _gamma_alpha(
    s: Array,
    t: Array,
    tau: Array,
    beta: Array,
    scale: Array,
    rate: Array,
    degree: Array,
    nodes: Array,
    weights: Array,
    rho: float | Array,
) -> tuple[Array, Array]:
    r"""Evaluate normalized scalar Gamma coefficients by Gauss-Legendre quadrature.

    Computes

    .. math::

        \alpha_m(s, t, \tau;\, \rho) =
            \frac{\Gamma(\rho+1)}{(t-s)^{n}}
            \int_s^t \!\left(\frac{u-s}{t-s}\right)^{\!\rho}
            \dot{\kappa}(u;\, t,\tau,\, n)\,\mathrm{d}u,

    where :math:`n = |\ell_m| + 1` and :math:`\dot{\kappa}` is the closed-form
    dot-kappa of the scalar Gamma kernel.  The weight
    :math:`\bigl((u-s)/(t-s)\bigr)^\rho` reduces to :math:`1` at
    :math:`\rho = 0`, recovering the standard coefficient.  For the
    Gauss-Legendre nodes :math:`\xi_k` on :math:`[-1,1]`, the weight at node
    :math:`k` is :math:`((\xi_k+1)/2)^\rho`.

    All inputs are pre-cast to a common floating-point dtype and the quadrature
    nodes/weights pre-computed by the calling :meth:`GammaKernel.alpha`.

    Returns ``(vals, valid)`` with ``vals`` of shape ``leading + (1, M)``.
    """
    h = t - s
    valid = (h > 0) & (tau >= t)
    h_safe = jnp.where(valid, h, 1.0)

    # u has shape leading + (Q,); (u - s) / (t - s) = (nodes + 1) / 2
    u = s[..., None] + 0.5 * h_safe[..., None] * (nodes + 1.0)
    # fold rho weight into quadrature weights: leading + (Q,)
    effective_w = 0.5 * h_safe[..., None] * weights * (((nodes + 1.0) * 0.5) ** rho)

    n = degree + 1.0  # (M,)

    dot = _gamma_dot_kappa(
        u[..., None],
        t[..., None, None],
        tau[..., None, None],
        n,
        beta,
        scale,
        rate,
    )  # leading + (Q, M)
    kappa = jnp.sum(effective_w[..., None] * dot, axis=-2)  # leading + (M,)
    alpha = jnp.exp(gammaln(rho + 1.0)) * kappa / (h_safe[..., None] ** n)
    alpha = jnp.where(valid[..., None], alpha, 0.0)
    return alpha[..., None, :], valid


def _gamma_dot_kappa(
    u: Array,
    t: Array,
    tau: Array,
    n: Array,
    beta: Array,
    scale: Array,
    rate: Array,
) -> Array:
    """Closed-form dot-kappa integrand for the scalar Gamma kernel."""
    tau_u = tau - u
    x = jnp.clip((t - u) / tau_u, 0.0, 1.0)
    nbeta = n * beta
    exp_decay = jnp.exp(-rate * tau_u)

    dot_n1 = (
        scale
        * exp_decay
        * (tau_u ** (beta - 1.0))
        / jnp.exp(gammaln(beta))
    )
    dot_ngt1 = (
        (scale ** n)
        * exp_decay
        * (tau_u ** (nbeta - 1.0))
        * betainc((n - 1.0) * beta, beta, x)
        / jnp.exp(gammaln(nbeta))
    )
    return jnp.where(n == 1.0, dot_n1, dot_ngt1)


__all__ = ["ConvolutionKernel", "FractionalKernel", "GammaKernel"]