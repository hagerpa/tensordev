"""Volterra kernels and packed coefficient builders.

The public :class:`VolterraKernel` name is intentionally generic.  Internally,
the implemented coefficient builders assume the symmetry hypothesis of Part II:
for a word ``w p``, the coefficient depends on the prefix ``w`` only through its
multi-index of letter counts.  This covers convolution kernels, including the
multivariate fractional family, and piecewise constant kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln

from tensordev.util.combinatorics import build_multiindex_layout
from tensordev.volterra.coeffs import VolterraCoefficients


Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class VolterraKernel:
    r"""
    Volterra kernel data and coefficient builders.

    The kernel is represented as

    .. math::
        K(t,s) = \sum_{p=1}^q k_p(t,s)\, A_p,

    where :math:`A_p \in \mathbb{R}^{m \times d}`.  The currently implemented
    families are:

    - :meth:`fractional`: multivariate fractional convolution kernels
      :math:`k_p(t,s) = \Gamma(\beta_p)^{-1}(t-s)^{\beta_p - 1}`.
    - :meth:`gamma`: scalar Gamma kernel (:math:`q = 1`) with
      :math:`k(t,s) = \mathrm{scale}\cdot e^{-\mathrm{rate}(t-s)}\cdot\Gamma(\beta)^{-1}(t-s)^{\beta-1}`.
    - :meth:`piecewise_constant`: kernels constant on source/readout cells with
      coefficients ``B[p, source, readout]``.

    The coefficient methods return :class:`VolterraCoefficients`, whose
    ``alpha[..., p, ell_idx]`` stores

    .. math::
        \mathcal{K}_{s,t}^{w(\ell)\,p,\,\tau} \,/\, (t-s)^{|\ell|+1}.

    This normalization follows the definition of :math:`\mathcal{E}` in Part II,
    where a word of length :math:`|\ell|+1` is divided by :math:`(t-s)^{|\ell|+1}`.
    """

    kind: str = field(metadata={"static": True})
    A: Array
    beta: Array
    B: Array
    scale: Array
    rate: Array
    quad_order: int = field(default=32, metadata={"static": True})

    def __post_init__(self) -> None:
        """Validate and normalize kernel arrays.

        All array fields are converted to JAX arrays.  Shape and positivity
        constraints are enforced according to ``kind``.
        """
        A = jnp.asarray(self.A)
        beta = jnp.asarray(self.beta)
        B = jnp.asarray(self.B)
        scale = jnp.asarray(self.scale)
        rate = jnp.asarray(self.rate)

        if A.ndim != 3:
            raise ValueError(
                "A must have shape (q, m, d); "
                f"got {tuple(A.shape)}."
            )
        if self.kind not in {"fractional", "gamma", "piecewise_constant"}:
            raise ValueError(
                "kind must be one of 'fractional', 'gamma', 'piecewise_constant'; "
                f"got {self.kind!r}."
            )
        if self.quad_order <= 0:
            raise ValueError(f"quad_order must be positive, got {self.quad_order}.")

        q = int(A.shape[0])
        if self.kind == "fractional":
            if beta.shape != (q,):
                raise ValueError(
                    "fractional beta must have shape (q,), matching A.shape[0]; "
                    f"got beta.shape={tuple(beta.shape)} and q={q}."
                )
            if bool(jnp.any(beta <= 0)):
                raise ValueError("fractional beta entries must be positive.")
        elif self.kind == "gamma":
            if q != 1:
                raise ValueError("gamma kernels are scalar in this implementation, so A.shape[0] must be 1.")
            if beta.shape not in [(), (1,)]:
                raise ValueError(f"gamma beta must be scalar or shape (1,), got {tuple(beta.shape)}.")
            if scale.shape not in [(), (1,)]:
                raise ValueError(f"gamma scale must be scalar or shape (1,), got {tuple(scale.shape)}.")
            if rate.shape not in [(), (1,)]:
                raise ValueError(f"gamma rate must be scalar or shape (1,), got {tuple(rate.shape)}.")
            if bool(jnp.any(beta <= 0)):
                raise ValueError("gamma beta must be positive.")
            if bool(jnp.any(scale <= 0)):
                raise ValueError("gamma scale must be positive.")
            if bool(jnp.any(rate < 0)):
                raise ValueError("gamma rate must be non-negative.")
        else:
            if B.ndim != 3:
                raise ValueError(
                    "piecewise constant B must have shape (q, S, R); "
                    f"got {tuple(B.shape)}."
                )
            if B.shape[0] != q:
                raise ValueError(
                    "piecewise constant B and A must have the same component axis q; "
                    f"got B.shape[0]={B.shape[0]} and q={q}."
                )
            if B.shape[1] <= 0 or B.shape[2] <= 0:
                raise ValueError(f"piecewise constant B has invalid shape {tuple(B.shape)}.")

        object.__setattr__(self, "A", A)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rate", rate)

    @classmethod
    def fractional(cls, *, beta: Array, A: Array) -> "VolterraKernel":
        r"""Construct a multivariate fractional Volterra kernel.

        The scalar kernels are

        .. math::
            k_p(t,s) = \frac{(t-s)^{\beta_p - 1}}{\Gamma(\beta_p)},
            \qquad p = 1,\ldots,q,

        where :math:`\beta_p > 0`.  The full kernel is
        :math:`K(t,s) = \sum_p k_p(t,s)\,A_p`.

        Parameters
        ----------
        beta : Array
            Exponent vector of shape ``(q,)`` with all entries positive.
        A : Array
            Kernel matrices of shape ``(q, m, d)``.

        Returns
        -------
        VolterraKernel
            Fractional kernel instance.
        """
        return cls(
            kind="fractional",
            A=A,
            beta=beta,
            B=jnp.empty((0, 0, 0)),
            scale=jnp.asarray(1.0),
            rate=jnp.asarray(0.0),
            quad_order=1,
        )

    @classmethod
    def gamma(
        cls,
        *,
        beta: Array,
        A: Array,
        scale: Array = 1.0,
        rate: Array = 1.0,
        quad_order: int = 32,
    ) -> "VolterraKernel":
        r"""Construct a scalar Gamma Volterra kernel (:math:`q = 1`).

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

        Returns
        -------
        VolterraKernel
            Gamma kernel instance.
        """
        return cls(
            kind="gamma",
            A=A,
            beta=beta,
            B=jnp.empty((0, 0, 0)),
            scale=scale,
            rate=rate,
            quad_order=quad_order,
        )

    @classmethod
    def piecewise_constant(cls, *, B: Array, A: Array) -> "VolterraKernel":
        r"""Construct a piecewise constant Volterra kernel.

        The scalar coefficient for component ``p`` when the source variable
        lies in cell ``i`` and the readout variable lies in cell ``j`` is
        ``B[p, i, j]``.  The path grid supplied to :meth:`coef_grid` must
        match the cell structure of ``B``.

        Parameters
        ----------
        B : Array
            Coefficient tensor of shape ``(q, S, R)``, where ``S`` is the
            number of source cells and ``R`` is the number of readout cells.
        A : Array
            Kernel matrices of shape ``(q, m, d)``.

        Returns
        -------
        VolterraKernel
            Piecewise constant kernel instance.
        """
        return cls(
            kind="piecewise_constant",
            A=A,
            beta=jnp.empty((0,)),
            B=B,
            scale=jnp.asarray(1.0),
            rate=jnp.asarray(0.0),
            quad_order=1,
        )

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

    def coef(
        self,
        s: Array,
        t: Array,
        tau: Array,
        *,
        trunc: int,
        dtype: Optional[jnp.dtype] = None,
    ) -> VolterraCoefficients:
        r"""Build packed coefficients for broadcasted triples ``(s,t,tau)``.

        ``s``, ``t`` and ``tau`` are broadcast to a common leading shape.  The
        returned ``alpha`` has shape ``leading + (q, M)``.  Triples outside
        ``s < t <= tau`` are marked invalid and their coefficients are zero.

        Piecewise constant kernels should usually use :meth:`coef_grid`, since
        their coefficients are indexed by source/readout cells rather than by
        analytic time triples.
        """
        if trunc <= 0:
            raise ValueError(f"trunc must be positive, got {trunc}.")
        if self.kind == "piecewise_constant":
            raise ValueError("piecewise_constant coefficients are cell-indexed; use coef_grid or coef_from_indices.")

        real_dtype = jnp.dtype(dtype or self.A.dtype)
        s_arr, t_arr, tau_arr = jnp.broadcast_arrays(
            jnp.asarray(s, dtype=real_dtype),
            jnp.asarray(t, dtype=real_dtype),
            jnp.asarray(tau, dtype=real_dtype),
        )
        layout = build_multiindex_layout(self.q, trunc - 1)

        if self.kind == "fractional":
            alpha, valid = _fractional_alpha(
                s_arr,
                t_arr,
                tau_arr,
                self.beta.astype(real_dtype),
                layout.ell,
                layout.degree,
            )
        else:
            alpha, valid = _gamma_alpha(
                s_arr,
                t_arr,
                tau_arr,
                self.beta.reshape(()).astype(real_dtype),
                self.scale.reshape(()).astype(real_dtype),
                self.rate.reshape(()).astype(real_dtype),
                layout.degree,
                quad_order=self.quad_order,
                dtype=real_dtype,
            )

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
        tau: Optional[Array] = None,
        dtype: Optional[jnp.dtype] = None,
    ) -> VolterraCoefficients:
        r"""Build source/readout-grid coefficients.

        Parameters
        ----------
        times:
            Path grid of shape ``(S + 1,)``.  Source interval ``i`` is
            ``[times[i], times[i+1]]``.
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

        if self.kind == "piecewise_constant":
            if tau is not None:
                raise ValueError(
                    "piecewise_constant coef_grid currently expects tau=None, "
                    "so readout cells are the same grid cells as the path grid."
                )
            S = int(times_arr.shape[0] - 1)
            if self.B.shape[1:] != (S, S):
                raise ValueError(
                    "For piecewise_constant coef_grid, B.shape must be (q, S, S) "
                    "for the supplied path grid; "
                    f"got B.shape={tuple(self.B.shape)} and S={S}."
                )
            return self.coef_from_indices(
                jnp.arange(S, dtype=jnp.int32)[:, None],
                jnp.arange(S, dtype=jnp.int32)[None, :],
                trunc=trunc,
                dtype=real_dtype,
            )

        return self.coef(
            s[:, None],
            t[:, None],
            tau_arr[None, :],
            trunc=trunc,
            dtype=real_dtype,
        )

    def coef_from_indices(
        self,
        source: Array,
        readout: Array,
        *,
        trunc: int,
        dtype: Optional[jnp.dtype] = None,
    ) -> VolterraCoefficients:
        r"""Build piecewise constant coefficients from source/readout cells."""
        if self.kind != "piecewise_constant":
            raise ValueError("coef_from_indices is only defined for piecewise_constant kernels.")
        if trunc <= 0:
            raise ValueError(f"trunc must be positive, got {trunc}.")

        source_arr, readout_arr = jnp.broadcast_arrays(
            jnp.asarray(source, dtype=jnp.int32),
            jnp.asarray(readout, dtype=jnp.int32),
        )
        S, R = int(self.B.shape[1]), int(self.B.shape[2])
        valid = (source_arr >= 0) & (source_arr < S) & (readout_arr >= 0) & (readout_arr < R) & (source_arr <= readout_arr)
        source_safe = jnp.where(valid, source_arr, 0)
        readout_safe = jnp.where(valid, readout_arr, 0)

        real_dtype = jnp.dtype(dtype or self.A.dtype)
        layout = build_multiindex_layout(self.q, trunc - 1)
        alpha = _piecewise_constant_alpha(
            self.B.astype(real_dtype),
            source_safe,
            readout_safe,
            layout.ell,
            layout.degree,
            valid,
        )
        return VolterraCoefficients(
            layout=layout,
            trunc=trunc,
            m=self.m,
            q=self.q,
            alpha=alpha.astype(real_dtype),
            valid=valid,
        )


@jax.jit
def _fractional_alpha(
    s: Array,
    t: Array,
    tau: Array,
    beta: Array,
    ell: Array,
    degree: Array,
) -> tuple[Array, Array]:
    """Evaluate normalized multivariate fractional coefficients."""
    dtype = jnp.result_type(s, t, tau, beta)
    s = s.astype(dtype)
    t = t.astype(dtype)
    tau = tau.astype(dtype)
    beta = beta.astype(dtype)
    ell = ell.astype(dtype)
    degree = degree.astype(dtype)

    h = t - s
    tau_s = tau - s
    valid = (h > 0) & (tau >= t)
    h_safe = jnp.where(valid, h, jnp.ones_like(h))
    tau_s_safe = jnp.where(valid, tau_s, jnp.ones_like(tau_s))
    z = jnp.clip(h_safe / tau_s_safe, 0.0, 1.0)

    prefix = ell @ beta                      # (M,)
    total = prefix[:, None] + beta[None, :]  # (M, q)
    a = prefix[:, None] + 1.0
    b = beta[None, :]

    # leading + (M, q)
    log_scale = (
        total * jnp.log(tau_s_safe[..., None, None])
        - (degree[:, None] + 1.0) * jnp.log(h_safe[..., None, None])
        - gammaln(total + 1.0)
    )
    vals = jnp.exp(log_scale) * betainc(a, b, z[..., None, None])
    vals = jnp.where(valid[..., None, None], vals, jnp.zeros_like(vals))
    return jnp.swapaxes(vals, -1, -2), valid  # leading + (q, M)


@partial(jax.jit, static_argnames=("quad_order", "dtype"))
def _gamma_alpha(
    s: Array,
    t: Array,
    tau: Array,
    beta: Array,
    scale: Array,
    rate: Array,
    degree: Array,
    *,
    quad_order: int,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Evaluate normalized scalar Gamma coefficients by Gauss-Legendre quadrature."""
    dtype = jnp.dtype(dtype)
    s = s.astype(dtype)
    t = t.astype(dtype)
    tau = tau.astype(dtype)
    beta = beta.astype(dtype)
    scale = scale.astype(dtype)
    rate = rate.astype(dtype)
    degree = degree.astype(dtype)

    nodes_np, weights_np = np.polynomial.legendre.leggauss(int(quad_order))
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    weights = jnp.asarray(weights_np, dtype=dtype)

    h = t - s
    valid = (h > 0) & (tau >= t)
    h_safe = jnp.where(valid, h, jnp.ones_like(h))

    # u has shape leading + (Q,)
    u = s[..., None] + 0.5 * h_safe[..., None] * (nodes + 1.0)
    quad_w = 0.5 * h_safe[..., None] * weights

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
    kappa = jnp.sum(quad_w[..., None] * dot, axis=-2)  # leading + (M,)
    alpha = kappa / (h_safe[..., None] ** n)
    alpha = jnp.where(valid[..., None], alpha, jnp.zeros_like(alpha))
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
    """Closed-form dot-kappa for the scalar Gamma kernel."""
    tau_u = tau - u
    x = jnp.clip((t - u) / tau_u, 0.0, 1.0)
    nbeta = n * beta

    dot_n1 = (
        scale
        * jnp.exp(-rate * tau_u)
        * (tau_u ** (beta - 1.0))
        / jnp.exp(gammaln(beta))
    )
    dot_ngt1 = (
        (scale ** n)
        * jnp.exp(-rate * tau_u)
        * (tau_u ** (nbeta - 1.0))
        * betainc((n - 1.0) * beta, beta, x)
        / jnp.exp(gammaln(nbeta))
    )
    return jnp.where(n == 1.0, dot_n1, dot_ngt1)


@jax.jit
def _piecewise_constant_alpha(
    B: Array,
    source: Array,
    readout: Array,
    ell: Array,
    degree: Array,
    valid: Array,
) -> Array:
    """Evaluate normalized piecewise constant coefficients from cell indices."""
    dtype = B.dtype
    ell_f = ell.astype(dtype)
    degree_f = degree.astype(dtype)

    # B_diag: leading + (q,), B_out: leading + (q,)
    diag = jnp.moveaxis(B[:, source, source], 0, -1)
    out = jnp.moveaxis(B[:, source, readout], 0, -1)

    # product_r diag_r ** ell_r, shape leading + (M,)
    prefix = jnp.prod(diag[..., None, :] ** ell_f, axis=-1)
    inv_fact = jnp.exp(-gammaln(degree_f + 2.0))  # 1 / (|ell| + 1)!
    vals = out[..., :, None] * prefix[..., None, :] * inv_fact
    return jnp.where(valid[..., None, None], vals, jnp.zeros_like(vals))


__all__ = ["VolterraKernel"]
