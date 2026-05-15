"""Finite-state-space Volterra kernels and coefficient builders.

This module contains :class:`FSSK`, the public kernel object for the
finite-state-space family, together with the shared coefficient assembly
routine used by both dense and Jordan state-space realizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from tensordev.util.combinatorics import build_multiindex_layout, num_multiindices_leq
from tensordev.sss.coeffs import FSSKCoefficients
from tensordev.sss.lambdas import DenseLambda, JordanLambda, Lambda

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class FSSK:
    r"""
    Finite-state-space Volterra kernel data.

    This class represents kernels of the form

    .. math::
        K_{A,b}^{\Lambda}(t, s)
        = \sum_{r=1}^n \big(\mathbf 1^\top e^{-\Lambda(t-s)} b_r\big) A_r,

    where ``A_r in R^{m x d}``, ``b_r in R^R`` and ``Lambda`` is a state-space
    operator providing ``expm`` and shifted linear solves.

    The method :meth:`coef` interprets ``dt`` as an arbitrary batch of time
    increments. If ``dt.shape == batch_shape``, then the returned coefficient
    arrays carry the same leading ``batch_shape``.
    """

    Lambda: Lambda
    A: Array
    b: Array
    quad_order: int = field(default=32, metadata={"static": True})

    def __post_init__(self) -> None:
        """Validate and normalize the kernel arrays.

        The kernel matrices ``A`` are stored with shape ``(n, m, d)`` and the
        state vectors ``b`` with shape ``(n, R)``. Both arrays are converted to
        JAX arrays so that the resulting kernel is a valid pytree and can be
        passed into jitted helper functions.
        """
        A = jnp.asarray(self.A)
        b = jnp.asarray(self.b)

        if A.ndim != 3:
            raise ValueError(
                "A must have shape (n, m, d); "
                f"got {tuple(A.shape)}."
            )
        if b.ndim != 2:
            raise ValueError(
                "b must have shape (n, R); "
                f"got {tuple(b.shape)}."
            )
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                "A and b must have the same leading component axis n; "
                f"got A.shape[0]={A.shape[0]} and b.shape[0]={b.shape[0]}."
            )
        if b.shape[1] != self.Lambda.state_dim:
            raise ValueError(
                "Lambda and b have incompatible state dimension R; "
                f"got Lambda.state_dim={self.Lambda.state_dim} and b.shape={tuple(b.shape)}."
            )
        if self.quad_order <= 0:
            raise ValueError(f"quad_order must be positive, got {self.quad_order}.")

        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)

    @classmethod
    def from_matrix(
            cls,
            *,
            Lambda: Array,
            A: Array,
            b: Array,
            quad_order: int = 32,
    ) -> "FSSK":
        r"""Construct a kernel from a dense matrix realization.

        This constructor uses the finite-state-space representation

        .. math::
            K_{A,b}^{\Lambda}(t,s)
            = \sum_{r=1}^n \big(\mathbf 1^\top e^{-\Lambda (t-s)} b_r\big) A_r,

        where:

        - :math:`\Lambda \in \mathbb{R}^{R \times R}` is given directly as a
          dense matrix,
        - :math:`b_r \in \mathbb{R}^R` are the state vectors collected in
          ``b``,
        - :math:`A_r \in \mathbb{R}^{m \times d}` are the kernel matrices
          collected in ``A``.

        This is the most direct constructor. It is appropriate when the
        state-space realization is already available in matrix form and no
        additional structural assumptions on :math:`\Lambda` are to be used.

        Parameters
        ----------
        Lambda : Array
            Dense state matrix with shape ``(R, R)``.
        A : Array
            Kernel matrices with shape ``(n, m, d)``.
        b : Array
            State vectors with shape ``(n, R)``.
        quad_order : int, default=32
            Number of contour quadrature nodes used by :meth:`coef`.

        Returns
        -------
        FSSK
            Kernel using a :class:`DenseLambda` realization.
        """
        return cls(Lambda=DenseLambda(Lambda), A=A, b=b, quad_order=quad_order)

    @classmethod
    def from_prony(
            cls,
            *,
            A: Array,
            real_rates: Array = (),
            real_sizes: Array = (),
            osc_decays: Array = (),
            osc_freqs: Array = (),
            osc_sizes: Array = (),
            alpha: Optional[Array] = None,
            beta: Optional[Array] = None,
            delta: Optional[Array] = None,
            quad_order: int = 32,
    ) -> "FSSK":
        r"""Construct a kernel from Jordan data and Prony coefficients.

        This constructor first builds a :class:`JordanLambda` from the
        prescribed real and oscillatory Jordan blocks, and then constructs the
        state vectors ``b`` from Prony coefficients ``alpha``, ``beta`` and
        ``delta``.

        The resulting kernel has the form

        .. math::
            K_{A,b}^{\Lambda}(t,s)
            = \sum_{r=1}^n \big(\mathbf 1^\top e^{-\Lambda (t-s)} b_r\big) A_r.

        For each component :math:`r`, the vector :math:`b_r` is assembled
        blockwise in the Jordan basis.

        **Real Jordan blocks.**
        For a real block of size :math:`n` with coefficients

        .. math::
            \alpha^{(r)} = (\alpha^{(r)}_0, \ldots, \alpha^{(r)}_{n-1}),

        and the convention :math:`\alpha^{(r)}_n = 0`, define

        .. math::
            \Delta \alpha^{(r)}_k
            = \alpha^{(r)}_k - \alpha^{(r)}_{k+1},
            \qquad k = 0, \ldots, n-1.

        The contribution of that block to :math:`b_r` is

        .. math::
            b^{(r)}_{\mathrm{real}}
            =
            \big(
                \Delta\alpha^{(r)}_0,
                \ldots,
                \Delta\alpha^{(r)}_{n-1}
            \big).

        **Oscillatory Jordan block pairs.**
        For an oscillatory pair of size :math:`n` with coefficients

        .. math::
            \beta^{(r)} = (\beta^{(r)}_0, \ldots, \beta^{(r)}_{n-1}),
            \qquad
            \delta^{(r)} = (\delta^{(r)}_0, \ldots, \delta^{(r)}_{n-1}),

        and the conventions :math:`\beta^{(r)}_n = \delta^{(r)}_n = 0`, define

        .. math::
            \Delta \beta^{(r)}_k
            = \beta^{(r)}_k - \beta^{(r)}_{k+1},
            \qquad
            \Delta \delta^{(r)}_k
            = \delta^{(r)}_k - \delta^{(r)}_{k+1}.

        The corresponding length-:math:`2n` Jordan-basis contribution is

        .. math::
            b^{(r)}_{\mathrm{osc}}
            =
            \Big(
                \tfrac12(\Delta\beta^{(r)}_0 - \Delta\delta^{(r)}_0),
                \tfrac12(\Delta\beta^{(r)}_0 + \Delta\delta^{(r)}_0),
                \ldots,
                \tfrac12(\Delta\beta^{(r)}_{n-1} - \Delta\delta^{(r)}_{n-1}),
                \tfrac12(\Delta\beta^{(r)}_{n-1} + \Delta\delta^{(r)}_{n-1})
            \Big).

        The full vector :math:`b_r` is obtained by concatenating all real-block
        and oscillatory-block contributions in the same order as the blocks are
        specified to :class:`JordanLambda`.

        Parameters
        ----------
        A : Array
            Kernel matrices with shape ``(n, m, d)``.
        real_rates, real_sizes : array-like
            Rates and Jordan block sizes for the real scalar poles.
        osc_decays, osc_freqs, osc_sizes : array-like
            Decays, frequencies and Jordan block sizes for oscillatory pole
            pairs.
        alpha, beta, delta : Array, optional
            Prony coefficients used to construct ``b``.

            - ``alpha`` must have shape ``(n, sum(real_sizes))`` when real
              blocks are present.
            - ``beta`` and ``delta`` must both have shape
              ``(n, sum(osc_sizes))`` when oscillatory blocks are present.

        quad_order : int, default=32
            Number of contour quadrature nodes used by :meth:`coef`.

        Returns
        -------
        FSSK
            Kernel using a :class:`JordanLambda` realization with ``b``
            constructed from the supplied Prony coefficients.
        """
        lam = JordanLambda(
            real_rates=real_rates,
            real_sizes=real_sizes,
            osc_decays=osc_decays,
            osc_freqs=osc_freqs,
            osc_sizes=osc_sizes,
        )
        b = lam.b_from_prony(alpha=alpha, beta=beta, delta=delta)
        return cls(Lambda=lam, A=A, b=b, quad_order=quad_order)

    @classmethod
    def from_jordan(
            cls,
            *,
            A: Array,
            b: Array,
            real_rates: Array = (),
            real_sizes: Array = (),
            osc_decays: Array = (),
            osc_freqs: Array = (),
            osc_sizes: Array = (),
            quad_order: int = 32,
    ) -> "FSSK":
        r"""Construct a kernel from Jordan block data and explicit state vectors.

        This constructor builds a :class:`JordanLambda` from the prescribed
        real and oscillatory Jordan blocks, while the state vectors ``b`` are
        supplied directly.

        The resulting kernel has the finite-state-space form

        .. math::
            K_{A,b}^{\Lambda}(t,s)
            = \sum_{r=1}^n \big(\mathbf 1^\top e^{-\Lambda (t-s)} b_r\big) A_r.

        Here :math:`\Lambda` is not passed as a dense matrix. Instead it is
        specified through its Jordan structure:

        - real Jordan blocks with rates ``real_rates`` and sizes
          ``real_sizes``,
        - oscillatory Jordan block pairs with decays ``osc_decays``,
          frequencies ``osc_freqs`` and sizes ``osc_sizes``.

        The vectors ``b_r`` are assumed to already be expressed in the same
        Jordan basis determined by those block specifications.

        Parameters
        ----------
        A : Array
            Kernel matrices with shape ``(n, m, d)``.
        b : Array
            State vectors with shape ``(n, R)``, expressed in the Jordan basis
            induced by the supplied block data.
        real_rates, real_sizes : array-like
            Rates and Jordan block sizes for the real scalar poles.
        osc_decays, osc_freqs, osc_sizes : array-like
            Decays, frequencies and Jordan block sizes for oscillatory pole
            pairs.
        quad_order : int, default=32
            Number of contour quadrature nodes used by :meth:`coef`.

        Returns
        -------
        FSSK
            Kernel using a :class:`JordanLambda` realization with explicitly
            provided Jordan-basis vectors ``b``.
        """
        lam = JordanLambda(
            real_rates=real_rates,
            real_sizes=real_sizes,
            osc_decays=osc_decays,
            osc_freqs=osc_freqs,
            osc_sizes=osc_sizes,
        )
        return cls(Lambda=lam, A=A, b=b, quad_order=quad_order)

    @property
    def q(self) -> int:
        """Number of kernel components in the decomposition."""
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        """Latent Volterra state width of each component matrix ``A_r``."""
        return int(self.A.shape[1])

    @property
    def path_dim(self) -> int:
        """Path dimension ``d`` of the driving signal."""
        return int(self.A.shape[2])

    @property
    def state_dim(self) -> int:
        """State-space dimension ``R`` of the realization."""
        return self.Lambda.state_dim

    def coef(
            self,
            dt: Array,
            *,
            trunc: int,
            dtype: Optional[jnp.dtype] = None,
    ) -> FSSKCoefficients:
        """Build packed FSSK coefficients for a batch of time increments.

        Let ``batch_shape = dt.shape``. Then the returned coefficient arrays have
        shapes

        - ``E``:   ``batch_shape + (R, R)``
        - ``psi``: ``batch_shape + (M, R)``
        - ``phi``: ``batch_shape + (n, Mphi, R, R)``

        where ``M`` and ``Mphi`` are determined by the packed multi-index layout.

        Parameters
        ----------
        dt : Array
            Array of time increments. Its full shape is treated as a batch shape.
        trunc : int
            Tensor truncation level. The packed multi-index layout is built up
            to degree ``trunc - 1``.
        dtype : optional
            Real dtype used for the returned arrays. If omitted, the dtype is
            inherited from ``b``.

        Returns
        -------
        FSSKCoefficients
            Packed coefficient object containing the propagators ``E`` and the
            normalized coefficient arrays ``psi`` and ``phi`` over the batch
            induced by ``dt``.
        """
        if trunc <= 0:
            raise ValueError(f"trunc must be positive, got {trunc}.")

        dt_arr = jnp.asarray(dt)
        real_dtype = jnp.dtype(dtype or self.b.dtype)
        dt_arr = dt_arr.astype(real_dtype)
        b = self.b.astype(real_dtype)

        layout = build_multiindex_layout(self.q, trunc - 1)
        dt_flat = dt_arr.reshape(-1)
        zeta_c, slope, gamma, r, u = prepare_coef(
            self.Lambda,
            b,
            dt_flat,
            layout.ell,
            quad_order=self.quad_order,
            dtype=real_dtype,
        )
        psi_flat = eval_psi(self.Lambda, dt_flat, zeta_c, slope, gamma, r, dtype=real_dtype)
        E_flat, phi_flat = eval_phi(self.Lambda, dt_flat, zeta_c, slope, gamma, u, r, dtype=real_dtype)

        batch_shape = dt_arr.shape
        R = self.state_dim
        E = E_flat.reshape(batch_shape + (R, R))
        psi = psi_flat.reshape(batch_shape + psi_flat.shape[-2:]).astype(real_dtype)
        phi_full = phi_flat.reshape(batch_shape + phi_flat.shape[-4:]).astype(real_dtype)
        mphi = num_multiindices_leq(self.q, trunc - 2)
        phi = phi_full[..., :mphi, :, :]
        return FSSKCoefficients(
            layout=layout,
            trunc=trunc,
            m=self.m,
            q=self.q,
            R=self.state_dim,
            E=E,
            psi=psi,
            phi=phi,
        )


@partial(jax.jit, static_argnames=("quad_order", "dtype"))
def prepare_coef(
        lam: Lambda,
        b: Array,
        dt_flat: Array,
        ell: Array,
        *,
        quad_order: int,
        dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array, Array]:
    """Compute quadrature nodes/weights and shared intermediates.

    Returns ``(zeta_c, slope, gamma, r, u)`` all on the flat batch axis
    ``B = dt_flat.shape[0]``.
    """
    R = int(b.shape[1])
    real_dtype = jnp.dtype(dtype)
    complex_dtype = _complex_dtype_for(real_dtype)

    m = quad_order
    j = jnp.arange(1, m + 1, dtype=real_dtype)
    pi = jnp.asarray(jnp.pi, dtype=real_dtype)
    theta = (2.0 * j - 1.0) * pi / (2.0 * m)

    zeta = (2.0 * m) * (
            jnp.asarray(0.1309, dtype=real_dtype)
            - jnp.asarray(0.1194, dtype=real_dtype) * theta * theta
            + jnp.asarray(0.25j, dtype=complex_dtype) * theta.astype(complex_dtype)
    )
    slope = (
            jnp.asarray(0.2388j, dtype=complex_dtype) * theta.astype(complex_dtype)
            + jnp.asarray(0.25, dtype=complex_dtype)
    )
    zeta_c = zeta.astype(complex_dtype)

    b_c = b.astype(complex_dtype)
    ones = jnp.ones((R, 1), dtype=complex_dtype)

    r = jax.vmap(
        lambda z: lam.solve_shifted_transpose(z, dt_flat, ones, dtype=real_dtype)[..., 0],
        in_axes=0,
        out_axes=0,
    )(zeta)  # (m, B, R)
    u = jax.vmap(
        lambda z: jnp.swapaxes(
            lam.solve_shifted(z, dt_flat, b_c.T, dtype=real_dtype), -2, -1
        ),
        in_axes=0,
        out_axes=0,
    )(zeta)  # (m, B, n, R)

    beta = jnp.einsum("mbr,pr->mbp", r, b_c)
    gamma = jnp.prod(
        beta[None, :, :, :] ** ell.astype(complex_dtype)[:, None, None, :],
        axis=-1,
    )  # (M, m, B)

    return zeta_c, slope, gamma, r, u


@partial(jax.jit, static_argnames=("rho", "dtype"))
def eval_psi(
        lam: Lambda,
        dt_flat: Array,
        zeta_c: Array,
        slope: Array,
        gamma: Array,
        r: Array,
        *,
        rho: int = 0,
        dtype: jnp.dtype,
) -> Array:
    """Compute normalized rho-weighted psi coefficients.

    Returns shape ``(B, M, R)``.
    """
    ez = jnp.exp(zeta_c)
    inv_z = 1.0 / zeta_c

    if rho == 0:
        tilde_omega = ez * inv_z * slope
        h_scale = jnp.ones_like(dt_flat)

    elif rho == 1:
        # reverse first moment: (1 - theta)
        tilde_omega = ez * inv_z ** 2 * slope
        h_scale = dt_flat

    elif rho == 2:
        # reverse second moment: (1 - theta)^2
        tilde_omega = 2.0 * ez * inv_z ** 3 * slope
        h_scale = dt_flat ** 2

    else:
        raise NotImplementedError(
            f"FSSK eval_psi only supports rho=0,1,2; got rho={rho}."
        )

    if rho == 0:
        phi1_dt = lam.phi1(dt_flat, dtype=jnp.dtype(dtype))  # (B, R, R)
        psi_empty = jnp.sum(phi1_dt, axis=-2)  # (B, R)
    else:
        psi_empty = jnp.sum(
            2.0
            * jnp.real(
                tilde_omega[:, None, None]
                * gamma[0, :, :, None]
                * r[:, :, :]
            ),
            axis=0,
        )  # (B, R)

    # Apply absolute local-time scaling to both empty and non-empty coefficients.
    psi_empty = psi_empty * h_scale[:, None]

    psi_tail = jnp.sum(
        2.0
        * jnp.real(
            tilde_omega[None, :, None, None]
            * gamma[1:, :, :, None]
            * r[None, :, :, :]
        ),
        axis=1,
    )  # (M - 1, B, R)

    psi_tail = psi_tail * h_scale[None, :, None]
    psi_tail = jnp.transpose(psi_tail, (1, 0, 2))  # (B, M - 1, R)

    return jnp.concatenate(
        [psi_empty[:, None, :], psi_tail],
        axis=1,
    )  # (B, M, R)

@partial(jax.jit, static_argnames=("dtype",))
def eval_phi(
        lam: Lambda,
        dt_flat: Array,
        zeta_c: Array,
        slope: Array,
        gamma: Array,
        u: Array,
        r: Array,
        *,
        dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Compute the propagator E and normalized phi coefficients.

    Returns ``(E, phi)`` with shapes ``(B, R, R)`` and ``(B, n, M, R, R)``.
    """
    omega = jnp.exp(zeta_c) * slope
    E = lam.expm(dt_flat, dtype=jnp.dtype(dtype))  # (B, R, R)
    outer = u[:, :, :, :, None] * r[:, :, None, None, :]  # (m, B, n, R, R)
    phi = jnp.sum(
        2.0
        * jnp.real(
            omega[None, :, None, None, None, None]
            * gamma[:, :, :, None, None, None]
            * outer[None, :, :, :, :, :]
        ),
        axis=1,
    )  # (M, B, n, R, R)
    return E, jnp.transpose(phi, (1, 2, 0, 3, 4))  # (B, n, M, R, R)


def _complex_dtype_for(dtype: jnp.dtype) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float32:
        return jnp.dtype(jnp.complex64)
    return jnp.dtype(jnp.complex128)


__all__ = ["FSSK", "prepare_coef", "eval_psi", "eval_phi"]
