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

import functools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from tensordev.core.utils.xp import xp
from tensordev.util.combinatorics import MultiIndexLayout, build_multiindex_layout
from tensordev.util.special import betainc
from tensordev.volterra.coeffs import VolterraCoefficients
from tensordev.sss.lambdas import Lambda
from tensordev.sss.kernel import FSSK, prepare_coef, eval_psi


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
    def fractional(cls, *, beta: Array, A: Array) -> FractionalKernel:
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
        quad_order: int = 64,
    ) -> GammaKernel:
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
        quad_order : int, default=64
            Number of tanh-sinh nodes for coefficients with
            ``rate * (t - s)`` above the threshold of the truncated series
            (13.85 in float64 and 18.93 in float32); see
            :class:`GammaKernel`.
        """
        return GammaKernel(beta=beta, A=A, scale=scale, rate=rate, quad_order=quad_order)

    @classmethod
    def fssk(cls, fssk: FSSK) -> FSSKConvolutionKernel:
        r"""Construct a finite-state-space Volterra kernel adapter.

        Wraps an :class:`~tensordev.sss.kernel.FSSK` instance so that it
        satisfies the :class:`ConvolutionKernel` interface.  The scalar
        kernels are smooth exponential mixtures (:math:`\beta = 1`).

        Parameters
        ----------
        fssk : FSSK
            Finite-state-space kernel to wrap.
        """
        return FSSKConvolutionKernel.from_fssk(fssk)


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

    With :math:`\lambda = \mathrm{rate}`, :math:`h = t-s`, :math:`n = |\ell|+1`
    and the same unnormalised basis weight :math:`(u-s)^\rho` as
    :class:`FractionalKernel`, the coefficients are
    :math:`\alpha_n(s,t,\tau;\rho) = h^{-n}\int_s^t (u-s)^\rho\,
    \dot\kappa_n(u)\,\mathrm{d}u`.  Expanding
    :math:`e^{-\lambda(\tau-u)} = e^{-\lambda(\tau-s)}\sum_j \lambda^j(u-s)^j/j!`
    gives the exact positive series

    .. math::
        \alpha_n(\rho) = \mathrm{scale}^n e^{-\lambda(\tau-s)}
            \sum_{j\ge 0} \frac{\lambda^j}{j!}\,\alpha^{\mathrm{frac}}_n(\rho+j),

    where :math:`\alpha^{\mathrm{frac}}_n` is the closed form of
    :class:`FractionalKernel`.  It is truncated after 64 terms, which bounds
    the relative error by :math:`\sum_{j\ge 64} x^j/j!` with
    :math:`x = \lambda h`; this is below half the machine epsilon for
    :math:`x \le 13.8` in float64 and :math:`x \le 18.9` in float32.  Larger
    :math:`x` substitute :math:`(t-u)/h = -\log(1-cy)/x`,
    :math:`c = 1-e^{-x}`, so that
    :math:`e^{-xv}\,\mathrm{d}v = (c/x)\,\mathrm{d}y` for :math:`v = (t-u)/h`, and apply
    the tanh-sinh rule on :math:`y \in [0, 1]` with ``quad_order`` nodes
    :math:`y_k = (1+e^{-\pi\sinh t_k})^{-1}` at equispaced :math:`t_k`.  For
    :math:`\beta \ge 0.15` its float64 relative error is about ``1e-9`` for
    :math:`n\beta < 3` and ``1e-7`` for :math:`n\beta < 6` with 64 nodes and
    about ``1e-11`` for :math:`n\beta < 6` with 128 nodes; it grows with the
    exponent of the integrand factor :math:`(\tau-u)^{n\beta-1}`.  In
    float32 it is about ``1e-5``.

    At ``tau == t`` the one-sided partial derivative in ``tau`` is infinite
    for :math:`\beta < 1`.  There :math:`(t-s)/(\tau-s)` and :math:`\tau-t`
    are held constant in differentiation, so the returned derivatives in
    ``t`` and ``tau`` are not the partial derivatives, but their sum, the
    derivative of :math:`\alpha_n(s, t, t; \rho)` in :math:`t`, is exact.

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
    quad_order : int, default=64
        Number of tanh-sinh nodes for coefficients with ``rate * (t - s)``
        above the series threshold (13.85 in float64, 18.93 in float32).  When
        any coefficient of a call is above it, every coefficient of that call
        costs ``quad_order``, rounded up to a multiple of 16, incomplete-beta
        evaluations.
    """

    scale: Array
    rate: Array
    quad_order: int = field(default=64, metadata={"static": True})

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
        return _gamma_alpha(
            s.astype(dtype_),
            t.astype(dtype_),
            tau.astype(dtype_),
            self.beta.reshape(()).astype(dtype_),
            self.scale.reshape(()).astype(dtype_),
            self.rate.reshape(()).astype(dtype_),
            layout.degree.astype(dtype_),
            rho,
            num_terms=_GAMMA_SERIES_TERMS,
            num_nodes=int(self.quad_order),
        )

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
    valid, h_safe, tau_s_safe = _safe_intervals(s, t, tau)
    z = jnp.clip(h_safe / tau_s_safe, 0.0, 1.0)

    prefix = xp.matmul(ell, beta)            # (M,)
    total = prefix[:, None] + beta[None, :]  # (M, n)
    a = rho + prefix[:, None] + 1.0
    b = beta[None, :]

    # leading + (M, n)
    log_scale = _fractional_log_scale(
        rho,
        total,
        degree[:, None] + 1.0,
        tau_s_safe[..., None, None],
        h_safe[..., None, None],
    )
    vals = jnp.exp(log_scale) * betainc(a, b, z[..., None, None])
    vals = jnp.where(valid[..., None, None], vals, 0.0)
    return jnp.swapaxes(vals, -1, -2), valid  # leading + (n, M)


def _safe_intervals(s: Array, t: Array, tau: Array) -> tuple[Array, Array, Array]:
    """Mask ``s < t <= tau`` and ``(t - s, tau - s)``, set to 1 where it is false."""
    h = t - s
    valid = (h > 0) & (tau >= t)
    return valid, jnp.where(valid, h, 1.0), jnp.where(valid, tau - s, 1.0)


def _fractional_log_scale(
    rho: float | Array, total: Array, n: Array, tau_s: Array, h: Array
) -> Array:
    r"""Logarithm of the prefactor of the fractional coefficients,

    .. math::
        \log\frac{\Gamma(\rho+1)\,T^{\rho+\gamma}}{\Gamma(\rho+\gamma+1)\,h^n},

    with :math:`T` = ``tau_s`` and :math:`\gamma` = ``total``.
    """
    return (
        (rho + total) * jnp.log(tau_s)
        - n * jnp.log(h)
        + gammaln(rho + 1.0)
        - gammaln(rho + total + 1.0)
    )


# Terms of the gamma series; see GammaKernel for the resulting accuracy range.
_GAMMA_SERIES_TERMS = 64
# Tanh-sinh grid extent towards u = t and towards u = s; offsets from the
# endpoints reach exp(-pi sinh(extent)).  The u = t side carries the
# (t - u)^(n beta - 1) singularity and needs the longer reach; it is capped
# per dtype so the offsets stay normal numbers.
_TANH_SINH_SPAN = (5.0, 3.5)
# Nodes per scan step of the quadrature, bounding its working memory.
_TANH_SINH_CHUNK = 16


@functools.lru_cache(maxsize=None)
def _gamma_series_x_max(num_terms: int, dtype_name: str) -> float:
    r"""Largest ``x`` with :math:`\sum_{j \ge J} x^j/j! \le \epsilon/2`, ``J = num_terms``.

    This bounds the relative truncation error of the ``J``-term gamma series
    at ``x = rate * (t - s)``.
    """
    tol = 0.5 * float(np.finfo(np.dtype(dtype_name)).eps)

    def tail(x: float) -> float:
        term = math.exp(num_terms * math.log(x) - math.lgamma(num_terms + 1.0))
        total, j = 0.0, num_terms
        while term > 1e-20 * total:
            total += term
            j += 1
            term *= x / j
        return total

    lo, hi = 0.0, float(num_terms)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tail(mid) <= tol else (lo, mid)
    return lo


@functools.lru_cache(maxsize=None)
def _tanh_sinh_rule(num_nodes: int, dtype_name: str) -> tuple[np.ndarray, ...]:
    r"""Tanh-sinh rule on ``[0, 1]`` as ``(y, 1 - y, log y, log(1 - y), log w)``.

    .. math::
        y_k = \frac{1}{1 + e^{-\pi\sinh t_k}}, \qquad
        w_k = \Delta\,\pi\cosh(t_k)\,y_k(1-y_k), \qquad
        t_k = -L + (k + \tfrac12)\Delta, \qquad
        \Delta = \frac{L+R}{N},

    for :math:`k < N` = ``num_nodes``, where
    :math:`L = \min(L_0, \operatorname{asinh}(-\log(2^{10}\varepsilon_0)/\pi))`,
    :math:`(L_0, R)` = ``_TANH_SINH_SPAN`` and :math:`\varepsilon_0` is the
    smallest normal number of the dtype.  :math:`1-y` is stored separately,
    so it stays positive where :math:`y` rounds to 1.  Arrays have shape
    ``(chunks, _TANH_SINH_CHUNK)``; padding nodes carry ``log w = -inf``.
    """
    reach = -math.log(float(np.finfo(np.dtype(dtype_name)).tiny)) - 10.0 * math.log(2.0)
    left = min(_TANH_SINH_SPAN[0], math.asinh(reach / math.pi))
    step = (left + _TANH_SINH_SPAN[1]) / num_nodes
    nodes = -left + (np.arange(num_nodes) + 0.5) * step
    arg = np.pi * np.sinh(nodes)
    log_y = -np.logaddexp(0.0, -arg)
    log_yc = -np.logaddexp(0.0, arg)
    log_w = np.log(step * np.pi * np.cosh(nodes)) + log_y + log_yc
    pad = (0, -num_nodes % _TANH_SINH_CHUNK)
    half = math.log(0.5)
    columns = (
        (np.exp(log_y), 0.5),
        (np.exp(log_yc), 0.5),
        (log_y, half),
        (log_yc, half),
        (log_w, -np.inf),
    )
    return tuple(
        np.pad(v, pad, constant_values=fill).reshape(-1, _TANH_SINH_CHUNK)
        for v, fill in columns
    )


@functools.partial(jax.jit, static_argnames=("num_terms", "num_nodes"))
def _gamma_alpha(
    s: Array,
    t: Array,
    tau: Array,
    beta: Array,
    scale: Array,
    rate: Array,
    degree: Array,
    rho: float | Array,
    *,
    num_terms: int,
    num_nodes: int,
) -> tuple[Array, Array]:
    r"""Evaluate normalized scalar Gamma coefficients.

    Computes, for :math:`n = |\ell_m| + 1`, :math:`h = t-s` and
    :math:`\lambda = \mathrm{rate}`,

    .. math::

        \alpha_m(s, t, \tau;\, \rho)
            = h^{-n} \int_s^t (u-s)^{\rho}\, \dot{\kappa}_n(u)\,\mathrm{d}u,
        \qquad
        \dot{\kappa}_n(u) = \mathrm{scale}^n e^{-\lambda(\tau-u)}
            \frac{(\tau-u)^{n\beta-1}}{\Gamma(n\beta)}
            I_{\frac{t-u}{\tau-u}}\bigl((n-1)\beta,\, \beta\bigr),

    with :math:`I_x(0, \beta) = 1`.  For :math:`x = \lambda h` up to
    :func:`_gamma_series_x_max` this is the ``num_terms``-term series of
    :class:`GammaKernel`, and otherwise the ``num_nodes``-node tanh-sinh
    quadrature :func:`_gamma_alpha_quad`.  All inputs are pre-cast to a common
    floating-point dtype by :meth:`GammaKernel.alpha`.  The result is zero
    whenever :math:`s < t \le \tau` is violated.

    Returns ``(vals, valid)`` with ``vals`` of shape ``leading + (1, M)``.
    """
    valid, h_safe, tau_s_safe = _safe_intervals(s, t, tau)
    # z and tau - t are held constant at tau == t, where the derivative in
    # tau alone is infinite for beta < 1 (see GammaKernel).
    delta = jnp.where(valid & (tau > t), tau - t, 0.0)
    z = h_safe / tau_s_safe
    z = jnp.where(z < 1.0, z, 1.0)
    x = rate * h_safe
    n = degree + 1.0

    x_max = _gamma_series_x_max(num_terms, s.dtype.name)
    use_quad = valid & (x > x_max)
    alpha = _gamma_alpha_series(
        z, h_safe, tau_s_safe, jnp.where(use_quad, x_max, x),
        beta, scale, rate, n, rho, num_terms,
    )

    def with_quad() -> Array:
        quad = _gamma_alpha_quad(
            h_safe, delta, jnp.where(use_quad, x, x_max),
            beta, scale, rate, n, rho, num_nodes,
        )
        return jnp.where(use_quad[..., None], quad, alpha)

    alpha = jax.lax.cond(jnp.any(use_quad), with_quad, lambda: alpha)
    alpha = jnp.where(valid[..., None], alpha, 0.0)
    return alpha[..., None, :], valid


def _gamma_alpha_series(
    z: Array,
    h: Array,
    tau_s: Array,
    x: Array,
    beta: Array,
    scale: Array,
    rate: Array,
    n: Array,
    rho: float | Array,
    num_terms: int,
) -> Array:
    r"""Gamma coefficients by the series truncated after ``J = num_terms`` terms,

    .. math::
        \mathrm{scale}^n e^{-\lambda T}\sum_{j<J}
            \frac{\lambda^j}{j!}\,\alpha^{\mathrm{frac}}_n(\rho+j).

    With :math:`T = \tau - s`, :math:`a = \rho + (n-1)\beta + 1` and
    :math:`I_j = I_z(a+j, \beta)`,

    .. math::
        \frac{\lambda^j}{j!}\alpha^{\mathrm{frac}}_n(\rho+j)
            = \frac{\Gamma(\rho+1)\, T^{\rho+n\beta}}{\Gamma(\rho+n\beta+1)\, h^n}
              \Bigl(\prod_{k=1}^{j} e_k\Bigr) \frac{I_j}{z^j},
        \qquad e_k = \frac{x(\rho+k)}{k(\rho+k+n\beta)}.

    Only :math:`I_{J-1}` calls ``betainc``; the others follow from the
    additive recurrence :math:`I_j = I_{j+1} + d_j`,
    :math:`d_j = z^{a+j}(1-z)^\beta\,\Gamma(a+j+\beta)/(\Gamma(a+j+1)\Gamma(\beta))`.
    It runs on :math:`K_j = I_j z^{-j} / S` with
    :math:`S = \max(I_{J-1} z^{1-J}, d_0)` per entry, so every term stays in
    range in float32.  Inputs ``z, h, tau_s, x`` have the leading shape; the
    result has shape ``leading + (M,)``.
    """
    a = rho + (n - 1.0) * beta + 1.0  # (M,)
    nb = n * beta
    k = jnp.arange(num_terms, dtype=z.dtype)[:, None]  # (J, 1)
    # P[j] = d_j / (d_0 z^j), shape (J - 1, M) for j < J - 1
    P = jnp.exp(
        gammaln(a + k[:-1] + beta) - gammaln(a + beta)
        - gammaln(a + k[:-1] + 1.0) + gammaln(a + 1.0)
    )
    # q[j - 1] = e_j / x
    q = (rho + k[1:]) / (k[1:] * (rho + k[1:] + nb))  # (J - 1, M)

    zz = z[..., None]
    log_z = jnp.log(zz)
    has_tail = zz < 1.0
    log_1mz = jnp.log(jnp.where(has_tail, 1.0 - zz, 1.0))
    log_d0 = jnp.where(
        has_tail,
        a * log_z + beta * log_1mz + gammaln(a + beta) - gammaln(a + 1.0) - gammaln(beta),
        -jnp.inf,
    )
    i_top = betainc(a + (num_terms - 1), beta, zz)
    top_pos = i_top > 0
    log_top = jnp.where(
        top_pos,
        jnp.log(jnp.where(top_pos, i_top, 1.0)) - (num_terms - 1) * log_z,
        -jnp.inf,
    )
    # The result does not depend on S, so no gradient is needed through it.
    log_S = jax.lax.stop_gradient(jnp.maximum(log_top, log_d0))
    K = jnp.exp(log_top - log_S)
    D = jnp.exp(log_d0 - log_S)

    xx = x[..., None]

    def step(carry: tuple[Array, Array], coeffs: tuple[Array, Array]):
        K_next, acc = carry
        P_j, q_next = coeffs
        K_j = zz * K_next + D * P_j
        return (K_j, K_j + (xx * q_next) * acc), None

    (_, acc), _ = jax.lax.scan(step, (K, K), (P, q), reverse=True)

    log_pref = (
        n * jnp.log(scale)
        - rate * tau_s[..., None]
        + _fractional_log_scale(rho, nb, n, tau_s[..., None], h[..., None])
        + log_S
    )
    return jnp.exp(log_pref) * acc


def _gamma_alpha_quad(
    h: Array,
    delta: Array,
    x: Array,
    beta: Array,
    scale: Array,
    rate: Array,
    n: Array,
    rho: float | Array,
    num_nodes: int,
) -> Array:
    r"""Tanh-sinh quadrature of the Gamma coefficients for large ``x = rate * h``.

    Substitutes :math:`v = (t-u)/h = -\log(1 - c y)/x`, :math:`c = 1 - e^{-x}`,
    so that :math:`e^{-x v}\,\mathrm{d}v = (c/x)\,\mathrm{d}y`, and sums over
    the nodes :math:`(y_k, w_k)` of :func:`_tanh_sinh_rule`:

    .. math::
        \alpha_n \approx \frac{\mathrm{scale}^n e^{-\lambda\delta}\, c}
            {\Gamma(n\beta)\, h^{n-1} x}
            \sum_k w_k\, \bigl(h(1-v_k)\bigr)^\rho (\delta + h v_k)^{n\beta-1}
            I_{\frac{h v_k}{\delta + h v_k}}\bigl((n-1)\beta, \beta\bigr),

    with :math:`\delta = \tau - t`, :math:`x v_k = -\log(1 - c y_k)` and
    :math:`x(1 - v_k) = \log(y_k + (1-y_k) e^x)`, both formed from the stored
    :math:`y_k` and :math:`1 - y_k` without cancellation.  Nodes are processed
    in chunks by a checkpointed scan accumulating the sum in log space.  The
    result has shape ``leading + (M,)``.
    """
    dtype = h.dtype
    nodes = tuple(
        jnp.asarray(v, dtype=dtype) for v in _tanh_sinh_rule(num_nodes, dtype.name)
    )
    xe = x[..., None]
    c = -jnp.expm1(-xe)
    he = h[..., None]
    de = delta[..., None]
    use_inc = (n > 1.0)[:, None] & (de > 0)[..., None, :]  # leading + (M, 1)
    a_inc = jnp.where(n > 1.0, (n - 1.0) * beta, 1.0)[:, None]
    nb = n * beta

    def log_f(y: Array, yc: Array, log_y: Array, log_yc: Array, log_w: Array) -> Array:
        # x v = -log(1 - c y) and x (1 - v) = log(y + (1 - y) e^x).  The inner
        # where keeps log1p away from -1 where y rounds to 1 and c to 1, whose
        # infinite derivative would turn the reverse-mode gradient into NaN.
        lower = y <= 0.5
        xv = jnp.where(lower, -jnp.log1p(-c * jnp.where(lower, y, 0.0)), -jnp.log(yc + y * jnp.exp(-xe)))
        xw = jnp.logaddexp(log_y, log_yc + xe)
        t_u = he * xv / xe
        tau_u = de + t_u
        # leading + (M, chunk)
        x_inc = (t_u / tau_u)[..., None, :]
        inc_ok = use_inc & (x_inc > jnp.finfo(dtype).tiny)
        inc = betainc(a_inc, beta, jnp.where(inc_ok, x_inc, 0.5))
        inc_ok = inc_ok & (inc > 0)
        log_inc = jnp.where(
            inc_ok,
            jnp.log(jnp.where(inc_ok, inc, 1.0)),
            jnp.where(use_inc, -jnp.inf, 0.0),
        )
        return (
            log_w
            + rho * jnp.log(he * xw / xe)[..., None, :]
            + (nb[:, None] - 1.0) * jnp.log(tau_u)[..., None, :]
            + log_inc
        )

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def step(carry: tuple[Array, Array], chunk: tuple[Array, ...]):
        m, total = carry
        lf = log_f(*chunk)
        m_new = jax.lax.stop_gradient(jnp.maximum(m, jnp.max(lf, axis=-1)))
        shift = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
        total = total * jnp.exp(m - shift) + jnp.sum(jnp.exp(lf - shift[..., None]), axis=-1)
        return (m_new, total), None

    shape = x.shape + n.shape
    init = (jnp.full(shape, -jnp.inf, dtype=dtype), jnp.zeros(shape, dtype=dtype))
    (m, total), _ = jax.lax.scan(step, init, nodes)
    pos = total > 0
    log_sum = jnp.where(pos, m + jnp.log(jnp.where(pos, total, 1.0)), -jnp.inf)
    log_pref = (
        n * jnp.log(scale)
        - rate * de
        + (1.0 - n) * jnp.log(he)
        - gammaln(nb)
        + jnp.log(c / xe)
    )
    return jnp.exp(log_pref + log_sum)


def _rho_to_static_int(rho: float | Array) -> int:
    """Convert rho to a static integer for FSSK eval_psi.

    For FSSK, beta is one, so the deduplicated higher-order basis only
    produces rho in {0, 1, 2}.
    """
    rho_val = float(jax.device_get(jnp.asarray(rho)))
    rho_int = int(round(rho_val))

    if abs(rho_val - rho_int) > 1e-8 or rho_int not in (0, 1, 2):
        raise NotImplementedError(
            f"FSSKConvolutionKernel only supports rho in {{0, 1, 2}}, "
            f"got rho={rho_val}."
        )

    return rho_int


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class FSSKConvolutionKernel(ConvolutionKernel):
    """Expose an FSSK kernel through the generic ConvolutionKernel interface.

    This lets the generic Volterra iteration code call:

        kernel.coef(..., rho=...)

    while the coefficient values are computed using the FSSK psi coefficients.

    Notes
    -----
    FSSK is smooth/exponential, so beta is fixed to ones(q).
    """

    lam: Lambda
    b: Array
    quad_order: int = field(default=32, metadata={"static": True})

    @classmethod
    def from_fssk(cls, fssk: FSSK) -> FSSKConvolutionKernel:
        """Build a generic ConvolutionKernel adapter from an existing FSSK."""

        A = jnp.asarray(fssk.A)
        beta = jnp.ones((int(A.shape[0]),), dtype=A.dtype)
        return cls(
            A=A,
            beta=beta,
            lam=fssk.Lambda,
            b=jnp.asarray(fssk.b),
            quad_order=int(fssk.quad_order),
        )

    def __post_init__(self) -> None:
        ConvolutionKernel.__post_init__(self)

        b = jnp.asarray(self.b)
        object.__setattr__(self, "b", b)

        if self.beta.shape != (self.q,):
            raise ValueError(
                f"FSSKConvolutionKernel beta must have shape ({self.q},), "
                f"got {self.beta.shape}."
            )

        if bool(jnp.any(self.beta != 1)):
            raise ValueError("FSSKConvolutionKernel requires beta == 1 for all components.")

        if b.ndim != 2:
            raise ValueError(f"b must have shape (q, R), got {b.shape}.")

        if b.shape[0] != self.q:
            raise ValueError(
                f"b.shape[0] must match q={self.q}, got b.shape={b.shape}."
            )

        if self.quad_order <= 0:
            raise ValueError(f"quad_order must be positive, got {self.quad_order}.")

    @property
    def state_dim(self) -> int:
        """FSSK state dimension R."""
        return int(self.b.shape[1])

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
        """Evaluate packed FSSK Volterra coefficients.

        Returns
        -------
        vals:
            Array of shape ``leading_shape + (q, M)``, where
            ``M = layout.size``.
        valid:
            Boolean array of shape ``leading_shape``.

        The coefficient is

            vals[..., p, ell] =
                psi_rho[..., ell, R] · exp(-Lambda * (tau - t)) b_p

        in the normalized coefficient convention expected by VolterraCoefficients.
        """
        real_dtype = jnp.dtype(dtype or self.A.dtype)

        s_arr, t_arr, tau_arr = jnp.broadcast_arrays(
            jnp.asarray(s, dtype=real_dtype),
            jnp.asarray(t, dtype=real_dtype),
            jnp.asarray(tau, dtype=real_dtype),
        )

        h = t_arr - s_arr
        lookahead = tau_arr - t_arr
        valid = (h > 0) & (tau_arr >= t_arr)

        # Avoid invalid values entering matrix exponentials / shifted solves.
        h_safe = jnp.where(valid, h, 1.0)
        lookahead_safe = jnp.where(valid, lookahead, 0.0)

        leading_shape = h_safe.shape
        h_flat = h_safe.reshape((-1,))
        lookahead_flat = lookahead_safe.reshape((-1,))

        rho_int = _rho_to_static_int(rho)

        b = self.b.astype(real_dtype)

        zeta_c, slope, gamma, r, _u = prepare_coef(
            self.lam,
            b,
            h_flat,
            layout.ell,
            quad_order=int(self.quad_order),
            dtype=real_dtype,
        )

        psi_flat = eval_psi(
            self.lam,
            h_flat,
            zeta_c,
            slope,
            gamma,
            r,
            rho=rho_int,
            dtype=real_dtype,
        )  # (B, M, R)

        E_flat = self.lam.expm(
            lookahead_flat,
            dtype=real_dtype,
        )  # (B, R, R)

        # readout_flat[b, p, r] = (E_b b_p)_r
        readout_flat = xp.einsum("brs,ps->bpr", E_flat, b)  # (B, q, R)

        # vals_flat[b, p, ell] = sum_r psi[b, ell, r] * readout[b, p, r]
        vals_flat = xp.einsum("bmR,bpR->bpm", psi_flat, readout_flat)

        M = int(layout.ell.shape[0])
        vals = vals_flat.reshape(leading_shape + (self.q, M))
        vals = jnp.where(valid[..., None, None], vals, 0.0)

        return vals.astype(real_dtype), valid

    def as_convolution_kernel(self) -> FSSKConvolutionKernel:
        """For API symmetry; this object is already a ConvolutionKernel."""
        return self

__all__ = ["ConvolutionKernel", "FractionalKernel", "GammaKernel", "FSSKConvolutionKernel"]
