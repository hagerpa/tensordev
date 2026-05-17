"""State-space operator backends for finite-state-space Volterra kernels.

The public classes in this module implement the low-level linear-algebra
operations needed by finite-state-space kernels. Concrete realizations share a
common interface for

- propagators ``exp(-dt * Lambda)``,
- auxiliary operators ``phi_1(-dt * Lambda)``,
- auxiliary operators ``phi_2(-dt * Lambda)``,
- shifted linear solves,
- left and right operator actions used by the PDE scheme.

Conventions
-----------
- Public API accepts scalar ``dt`` or rank-1 batched ``dt``.
- Internal batched keranels use ``dt.shape == (m,)``.
- For paired batched actions, the operand leading axis must match ``dt.shape[0]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

Array = jax.Array

# ---------------------------------------------------------------------------
# Scalar / batched dt normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_dt_api(dt: Array | float, *, dtype: jnp.dtype) -> tuple[Array, bool]:
    """Promote scalar ``dt`` to shape ``(1,)`` and record whether it was scalar."""
    dt_arr = jnp.asarray(dt, dtype=dtype)
    if dt_arr.ndim == 0:
        return dt_arr[None], True
    if dt_arr.ndim == 1:
        return dt_arr, False
    raise ValueError(f"dt must be a scalar or shape (m,), got shape {tuple(dt_arr.shape)}.")


def _restore_dt_api(x: Array, was_scalar: bool) -> Array:
    """Remove the leading dt axis when the API input was scalar."""
    return x[0] if was_scalar else x


def _normalize_zeta_api(zeta: complex | Array, *, batch: int, dtype: jnp.dtype) -> Array:
    """Normalize ``zeta`` to shape ``(batch,)``."""
    zeta_arr = jnp.asarray(zeta, dtype=dtype)
    if zeta_arr.ndim == 0:
        return jnp.broadcast_to(zeta_arr, (batch,))
    if zeta_arr.ndim == 1:
        if zeta_arr.shape[0] == batch:
            return zeta_arr
        if zeta_arr.shape[0] == 1:
            return jnp.broadcast_to(zeta_arr, (batch,))
    raise ValueError(
        f"zeta must be a scalar or shape ({batch},), got shape {tuple(zeta_arr.shape)}."
    )


# ---------------------------------------------------------------------------
# Operand flattening / restoration for scalar and paired dt
# ---------------------------------------------------------------------------

_SIDE = Literal["left", "right"]


def _operand_shape(side: _SIDE) -> str:
    return "(R, k)" if side == "left" else "(k, R)"


def _flatten_scalar_dt_operand(
    x: Array,
    *,
    dtype: jnp.dtype,
    side: _SIDE,
    name: str,
) -> tuple[Array, tuple[int, ...]]:
    """Flatten arbitrary leading axes for scalar-``dt`` broadcasted actions."""
    x = jnp.asarray(x, dtype=dtype)
    if x.ndim < 2:
        raise ValueError(f"{name} must have shape (..., {_operand_shape(side)}).")
    if side == "right":
        x = jnp.swapaxes(x, -1, -2)
    prefix = tuple(int(s) for s in x.shape[:-2])
    return x.reshape((-1,) + x.shape[-2:]), prefix


def _restore_scalar_dt_operand(x: Array, prefix: tuple[int, ...], *, side: _SIDE) -> Array:
    """Undo :func:`_flatten_scalar_dt_operand`."""
    x = x.reshape(prefix + x.shape[-2:])
    return jnp.swapaxes(x, -1, -2) if side == "right" else x


def _flatten_paired_operand(
    dt: Array,
    x: Array,
    *,
    dtype: jnp.dtype,
    side: _SIDE,
    name: str,
) -> tuple[Array, bool, tuple[int, ...], int]:
    """Flatten a paired ``dt`` operand to shape ``(m, R, cols)``."""
    x = jnp.asarray(x, dtype=dtype)
    if side == "right":
        x = jnp.swapaxes(x, -1, -2)

    if dt.shape[0] == 1 and x.ndim == 2:
        x = x[None, ...]
        operand_was_unbatched = True
    elif x.ndim == 2 and dt.shape[0] > 1:
        # Broadcast unbatched operand across all dt values.
        x = jnp.broadcast_to(x[None, ...], (dt.shape[0],) + x.shape)
        operand_was_unbatched = True
    else:
        operand_was_unbatched = False
        if x.ndim < 3 or x.shape[0] != dt.shape[0]:
            raise ValueError(
                f"For batched dt, {name} must have leading axis {dt.shape[0]}; "
                f"got shape {tuple(x.shape)}."
            )

    extra = tuple(int(s) for s in x.shape[1:-2])
    cols = int(x.shape[-1])
    x = jnp.moveaxis(x, -2, 1)  # (m, R, *extra, k)
    flat = x.reshape((x.shape[0], x.shape[1], -1))
    return flat, operand_was_unbatched, extra, cols


def _restore_paired_operand(
    x: Array,
    *,
    side: _SIDE,
    extra: tuple[int, ...],
    cols: int,
    was_scalar_dt: bool,
    operand_was_unbatched: bool,
) -> Array:
    """Undo :func:`_flatten_paired_operand`."""
    x = x.reshape((x.shape[0], x.shape[1], *extra, cols))
    x = jnp.moveaxis(x, 1, -2)
    if side == "right":
        x = jnp.swapaxes(x, -1, -2)
    return x[0] if (was_scalar_dt and operand_was_unbatched) else x


# ---------------------------------------------------------------------------
# Unified dispatch: dt-dependent action / solve
# ---------------------------------------------------------------------------


def _apply_dt_action(
    dt: Array,
    was_scalar_dt: bool,
    x: Array,
    *,
    dtype: jnp.dtype,
    side: _SIDE,
    name: str,
    scalar_action: Callable[[Array], Array],
    batched_action: Callable[[Array, Array], Array],
) -> Array:
    """Apply a ``dt``-dependent linear action with shared shape handling."""
    if was_scalar_dt:
        x_flat, prefix = _flatten_scalar_dt_operand(x, dtype=dtype, side=side, name=name)
        out = scalar_action(x_flat)
        return _restore_scalar_dt_operand(out, prefix, side=side)

    x_flat, operand_was_unbatched, extra, cols = _flatten_paired_operand(
        dt, x, dtype=dtype, side=side, name=name,
    )
    out = batched_action(dt, x_flat)
    return _restore_paired_operand(
        out, side=side, extra=extra, cols=cols,
        was_scalar_dt=was_scalar_dt, operand_was_unbatched=operand_was_unbatched,
    )


def _apply_dt_solve(
    zeta: complex | Array,
    dt: Array,
    was_scalar_dt: bool,
    rhs: Array,
    *,
    dtype: jnp.dtype,
    scalar_solve: Callable[[Array, Array, Array], Array],
    batched_solve: Callable[[Array, Array, Array], Array],
) -> Array:
    """Apply a shifted solve with shared scalar/batched shape handling."""
    if was_scalar_dt:
        rhs_flat, prefix = _flatten_scalar_dt_operand(rhs, dtype=dtype, side="left", name="rhs")
        zeta_flat = _normalize_zeta_api(zeta, batch=rhs_flat.shape[0], dtype=dtype)
        dt_flat = jnp.full((rhs_flat.shape[0],), dt[0], dtype=dt.dtype)
        out = scalar_solve(zeta_flat, dt_flat, rhs_flat)
        return _restore_scalar_dt_operand(out, prefix, side="left")

    rhs_flat, rhs_was_unbatched, extra, cols = _flatten_paired_operand(
        dt, rhs, dtype=dtype, side="left", name="rhs",
    )
    zeta_vec = _normalize_zeta_api(zeta, batch=dt.shape[0], dtype=dtype)
    out = batched_solve(zeta_vec, dt, rhs_flat)
    return _restore_paired_operand(
        out, side="left", extra=extra, cols=cols,
        was_scalar_dt=was_scalar_dt, operand_was_unbatched=rhs_was_unbatched,
    )


# ---------------------------------------------------------------------------
# Small linear-algebra primitives
# ---------------------------------------------------------------------------


@jax.jit
def _left_multiply(m: Array, x: Array) -> Array:
    """Return ``m @ x`` on the second-to-last axis."""
    return jnp.einsum("ab,...bc->...ac", m, x)


@jax.jit
def _right_multiply(x: Array, m: Array) -> Array:
    """Return ``x @ m`` on the last axis."""
    return jnp.einsum("...ab,bc->...ac", x, m)


# ---------------------------------------------------------------------------
# Dense matrix operators
# ---------------------------------------------------------------------------


@jax.jit
def _dense_expm_batched(matrix: Array, dt: Array) -> Array:
    """Return batched ``exp(-dt M)``."""
    return jsp_linalg.expm(-dt[:, None, None] * matrix[None, :, :])


# ---------------------------------------------------------------------------
# Direct Padé + scaling-and-squaring for phi_1  (replaces augmented expm)
# ---------------------------------------------------------------------------

# [6/6] Padé coefficients for φ₁(z) = (e^z − 1)/z = Σ z^k/(k+1)!
# Computed at import time so the JIT function sees constant arrays.
def _compute_phi1_pade_coeffs() -> tuple[np.ndarray, np.ndarray]:
    """Return (numerator, denominator) coefficient vectors for the [6/6] Padé of φ₁."""
    from math import factorial as _fac
    p = 6
    c = np.array([1.0 / _fac(k + 1) for k in range(2 * p + 1)], dtype=np.float64)
    # Denominator d_1,…,d_p  (d_0 = 1)
    _A = np.zeros((p, p))
    _rhs = np.zeros(p)
    for _i in range(p):
        _k = p + 1 + _i
        _rhs[_i] = -c[_k]
        for _j in range(1, p + 1):
            _A[_i, _j - 1] = c[_k - _j]
    _d = np.linalg.solve(_A, _rhs)
    d = np.concatenate([[1.0], _d])
    # Numerator n_0,…,n_p
    n = np.zeros(p + 1)
    for _k in range(p + 1):
        for _j in range(min(_k, p) + 1):
            n[_k] += c[_k - _j] * d[_j]
    return n, d

_PHI1_PADE_N, _PHI1_PADE_D = _compute_phi1_pade_coeffs()
_PHI1_PADE_THETA = 1.0  # 1-norm threshold for [6/6] Padé (full float64 accuracy)


@jax.jit
def _phi1_pade_ss(A: Array) -> Array:
    r"""Compute :math:`\varphi_1(A) = (e^A - I) A^{-1}` via [6/6] Padé + scaling-and-squaring.

    Operates on a single ``(R, R)`` matrix.  Cost: ≈ 5 matrix multiplies
    for the Padé evaluation + 2 per squaring step, on an ``R × R`` matrix
    (instead of a ``2R × 2R`` augmented ``expm``).
    """
    r = A.shape[0]
    dtype = A.dtype
    eye = jnp.eye(r, dtype=dtype)

    # --- Scaling ---
    norm1 = jnp.linalg.norm(A.reshape(r, r), ord=1)
    s = jnp.maximum(0, jnp.ceil(jnp.log2(jnp.maximum(norm1, 1e-30) / _PHI1_PADE_THETA))).astype(int)
    B = A * (0.5 ** s)

    # --- [6/6] Padé via even/odd splitting (4 matmuls + 1 solve) ---
    n = jnp.asarray(_PHI1_PADE_N, dtype=dtype)
    d = jnp.asarray(_PHI1_PADE_D, dtype=dtype)

    B2 = B @ B
    B4 = B2 @ B2
    B6 = B4 @ B2

    N_even = n[0] * eye + n[2] * B2 + n[4] * B4 + n[6] * B6
    N_odd  = n[1] * eye + n[3] * B2 + n[5] * B4
    N_val  = N_even + B @ N_odd

    D_even = d[0] * eye + d[2] * B2 + d[4] * B4 + d[6] * B6
    D_odd  = d[1] * eye + d[3] * B2 + d[5] * B4
    D_val  = D_even + B @ D_odd

    phi1_B = jnp.linalg.solve(D_val, N_val)

    # --- Squaring: φ₁(2C) = ½ φ₁(C) (2I + C φ₁(C))  ---
    def _square(i, carry):
        C, phi1 = carry
        P = C @ phi1
        phi1 = 0.5 * (phi1 @ (2.0 * eye + P))
        C = 2.0 * C
        return C, phi1

    _, phi1_A = jax.lax.fori_loop(0, s, _square, (B, phi1_B))
    return phi1_A


@jax.jit
def _dense_phi1_batched(matrix: Array, dt: Array) -> Array:
    r"""Return batched ``\varphi_1(-dt\,M)`` via direct [6/6] Padé approximant.

    Uses scaling-and-squaring on the ``R × R`` matrix directly instead of
    the ``2R × 2R`` augmented exponential, halving the effective matrix size.
    """
    scaled = -dt[:, None, None] * matrix[None, :, :]
    return jax.vmap(_phi1_pade_ss)(scaled)


# ---------------------------------------------------------------------------
# Direct Padé + scaling-and-squaring for phi_2
# ---------------------------------------------------------------------------

# [6/6] Padé coefficients for φ₂(z) = (e^z − z − 1)/z² = Σ z^k/(k+2)!
def _compute_phi2_pade_coeffs() -> tuple[np.ndarray, np.ndarray]:
    """Return (numerator, denominator) coefficient vectors for the [6/6] Padé of φ₂."""
    from math import factorial as _fac
    p = 6
    c = np.array([1.0 / _fac(k + 2) for k in range(2 * p + 1)], dtype=np.float64)
    # Denominator d_1,…,d_p  (d_0 = 1)
    _A = np.zeros((p, p))
    _rhs = np.zeros(p)
    for _i in range(p):
        _k = p + 1 + _i
        _rhs[_i] = -c[_k]
        for _j in range(1, p + 1):
            _A[_i, _j - 1] = c[_k - _j]
    _d = np.linalg.solve(_A, _rhs)
    d = np.concatenate([[1.0], _d])
    # Numerator n_0,…,n_p
    n = np.zeros(p + 1)
    for _k in range(p + 1):
        for _j in range(min(_k, p) + 1):
            n[_k] += c[_k - _j] * d[_j]
    return n, d

_PHI2_PADE_N, _PHI2_PADE_D = _compute_phi2_pade_coeffs()
_PHI2_PADE_THETA = 1.0


@jax.jit
def _phi2_pade_ss(A: Array) -> Array:
    r"""Compute :math:`\varphi_2(A) = (e^A - A - I) A^{-2}` via [6/6] Padé + scaling-and-squaring.

    Jointly tracks ``\varphi_1`` during squaring because the recurrence
    ``\varphi_2(2C) = \tfrac12 \varphi_2(C) + \tfrac14 [\varphi_1(C)]^2``
    requires both functions.
    """
    r = A.shape[0]
    dtype = A.dtype
    eye = jnp.eye(r, dtype=dtype)

    # --- Scaling ---
    norm1 = jnp.linalg.norm(A.reshape(r, r), ord=1)
    s = jnp.maximum(0, jnp.ceil(jnp.log2(jnp.maximum(norm1, 1e-30) / _PHI2_PADE_THETA))).astype(int)
    B = A * (0.5 ** s)

    B2 = B @ B
    B4 = B2 @ B2
    B6 = B4 @ B2

    # --- phi1(B) via [6/6] Padé ---
    n1 = jnp.asarray(_PHI1_PADE_N, dtype=dtype)
    d1 = jnp.asarray(_PHI1_PADE_D, dtype=dtype)
    N1_even = n1[0] * eye + n1[2] * B2 + n1[4] * B4 + n1[6] * B6
    N1_odd  = n1[1] * eye + n1[3] * B2 + n1[5] * B4
    N1_val  = N1_even + B @ N1_odd
    D1_even = d1[0] * eye + d1[2] * B2 + d1[4] * B4 + d1[6] * B6
    D1_odd  = d1[1] * eye + d1[3] * B2 + d1[5] * B4
    D1_val  = D1_even + B @ D1_odd
    phi1_B = jnp.linalg.solve(D1_val, N1_val)

    # --- phi2(B) via [6/6] Padé ---
    n2 = jnp.asarray(_PHI2_PADE_N, dtype=dtype)
    d2 = jnp.asarray(_PHI2_PADE_D, dtype=dtype)
    N2_even = n2[0] * eye + n2[2] * B2 + n2[4] * B4 + n2[6] * B6
    N2_odd  = n2[1] * eye + n2[3] * B2 + n2[5] * B4
    N2_val  = N2_even + B @ N2_odd
    D2_even = d2[0] * eye + d2[2] * B2 + d2[4] * B4 + d2[6] * B6
    D2_odd  = d2[1] * eye + d2[3] * B2 + d2[5] * B4
    D2_val  = D2_even + B @ D2_odd
    phi2_B = jnp.linalg.solve(D2_val, N2_val)

    # --- Squaring: φ₁(2C) = ½ φ₁(C)(2I + C φ₁(C))
    #               φ₂(2C) = ½ φ₂(C) + ¼ [φ₁(C)]²  ---
    def _square(i, carry):
        C, phi1, phi2 = carry
        phi1_sq = phi1 @ phi1
        P = C @ phi1
        phi1_new = 0.5 * (phi1 @ (2.0 * eye + P))
        phi2_new = 0.5 * phi2 + 0.25 * phi1_sq
        C_new = 2.0 * C
        return C_new, phi1_new, phi2_new

    _, _, phi2_A = jax.lax.fori_loop(0, s, _square, (B, phi1_B, phi2_B))
    return phi2_A


@jax.jit
def _dense_phi2_batched(matrix: Array, dt: Array) -> Array:
    r"""Return batched ``\varphi_2(-dt\,M)`` via direct [6/6] Padé approximant."""
    scaled = -dt[:, None, None] * matrix[None, :, :]
    return jax.vmap(_phi2_pade_ss)(scaled)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Lambda(ABC):
    """Abstract state-space operator used by finite-state-space kernels."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension ``R`` of the underlying state space."""
        ...

    @abstractmethod
    def expm(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``exp(-dt * Lambda)``."""
        ...

    @abstractmethod
    def phi1(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``phi_1(-dt * Lambda)``."""
        ...

    @abstractmethod
    def phi2(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``phi_2(-dt * Lambda)``."""
        ...

    @abstractmethod
    def solve_shifted(
        self, zeta: complex | Array, dt: Array | float, rhs: Array,
        *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Solve ``(zeta I + dt Lambda) x = rhs``."""
        ...

    @abstractmethod
    def solve_shifted_transpose(
        self, zeta: complex | Array, dt: Array | float, rhs: Array,
        *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Solve ``(zeta I + dt Lambda)^T x = rhs``."""
        ...

    @abstractmethod
    def lambda_multiply_left(self, rhs: Array, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``Lambda @ rhs`` acting on the second-to-last axis of ``rhs``."""
        ...

    @abstractmethod
    def lambda_multiply_right(self, lhs: Array, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``lhs @ Lambda^T`` acting on the last axis of ``lhs``."""
        ...

    @abstractmethod
    def expm_multiply_left(
        self, dt: Array | float, rhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``exp(-dt * Lambda) @ rhs``."""
        ...

    @abstractmethod
    def expm_multiply_right(
        self, dt: Array | float, lhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``lhs @ exp(-dt * Lambda^T)``."""
        ...

    @abstractmethod
    def phi1_multiply_left(
        self, dt: Array | float, rhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``phi_1(-dt * Lambda) @ rhs``."""
        ...

    @abstractmethod
    def phi1_multiply_right(
        self, dt: Array | float, lhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``lhs @ phi_1(-dt * Lambda^T)``."""
        ...

    @abstractmethod
    def phi2_multiply_left(
        self, dt: Array | float, rhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``phi_2(-dt * Lambda) @ rhs``."""
        ...

    @abstractmethod
    def phi2_multiply_right(
        self, dt: Array | float, lhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``lhs @ phi_2(-dt * Lambda^T)``."""
        ...

    def precompute_expm(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> "Lambda":
        """Return a copy with ``exp(-dt * Lambda)`` cached per step. Default: returns self."""
        return self

    def precompute_phi1(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> "Lambda":
        """Return a copy with ``phi_1(-dt * Lambda)`` cached per step. Default: returns self."""
        return self

    def precompute_phi2(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> "Lambda":
        """Return a copy with ``phi_2(-dt * Lambda)`` cached per step. Default: returns self."""
        return self

    def __getitem__(self, ix) -> "Lambda":
        """Slice precomputed step-local blocks to step ``ix``. Default: returns self."""
        return self


# ---------------------------------------------------------------------------
# DenseLambda
# ---------------------------------------------------------------------------


# Condition-number threshold: fall back to Padé expm when V is too singular.
_EIGEN_COND_THRESHOLD = 1e10


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class DenseLambda(Lambda):
    """Dense matrix-backed implementation of :class:`Lambda`.

    Caches an eigendecomposition ``Lambda = V diag(eigvals) V^{-1}`` at
    construction time so that operator *actions* (``expm_multiply_*``,
    ``phi1_multiply_*``, ``solve_shifted*``) run in O(R²) per column
    instead of requiring an O(R³) ``expm`` materialisation.

    For near-defective matrices (condition number of *V* above
    ``_EIGEN_COND_THRESHOLD``) the class transparently falls back to the
    standard Padé-based ``expm``/``phi1`` and direct ``linalg.solve``.
    """

    matrix: Array
    _eigvals: Array = field(init=False, repr=False)
    _V: Array = field(init=False, repr=False)
    _V_inv: Array = field(init=False, repr=False)
    _use_eigen: bool = field(init=False, repr=False)
    _expm_mat: Optional[Array] = field(init=False, repr=False, default=None)
    _phi1_mat: Optional[Array] = field(init=False, repr=False, default=None)
    _phi2_mat: Optional[Array] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        matrix = jnp.asarray(self.matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "DenseLambda.matrix must have shape (R, R); "
                f"got {tuple(matrix.shape)}."
            )
        object.__setattr__(self, "matrix", matrix)

        # Cache eigendecomposition for O(R²) actions.
        cdtype = _complex_dtype_for(matrix.dtype)
        eigvals, V = jnp.linalg.eig(matrix.astype(cdtype))
        V_inv = jnp.linalg.inv(V)
        use_eigen = bool(jnp.linalg.cond(V) < _EIGEN_COND_THRESHOLD)
        object.__setattr__(self, "_eigvals", eigvals)
        object.__setattr__(self, "_V", V)
        object.__setattr__(self, "_V_inv", V_inv)
        object.__setattr__(self, "_use_eigen", use_eigen)

    # -- pytree: matrix is the single source of truth; eigen arrays travel
    #    alongside it so that JIT does not need to recompute them. ----------

    def tree_flatten(self):
        has_expm = self._expm_mat is not None
        has_phi1 = self._phi1_mat is not None
        has_phi2 = self._phi2_mat is not None
        leaves = [self.matrix, self._eigvals, self._V, self._V_inv]
        if has_expm:
            leaves.append(self._expm_mat)
        if has_phi1:
            leaves.append(self._phi1_mat)
        if has_phi2:
            leaves.append(self._phi2_mat)
        return tuple(leaves), (self._use_eigen, has_expm, has_phi1, has_phi2)

    @classmethod
    def tree_unflatten(cls, aux, children):
        use_eigen, has_expm, has_phi1, has_phi2 = aux
        matrix, eigvals, V, V_inv = children[:4]
        rest = list(children[4:])
        expm_mat = phi1_mat = phi2_mat = None
        if has_expm:
            expm_mat = rest.pop(0)
        if has_phi1:
            phi1_mat = rest.pop(0)
        if has_phi2:
            phi2_mat = rest.pop(0)
        obj = object.__new__(cls)
        object.__setattr__(obj, "matrix", matrix)
        object.__setattr__(obj, "_eigvals", eigvals)
        object.__setattr__(obj, "_V", V)
        object.__setattr__(obj, "_V_inv", V_inv)
        object.__setattr__(obj, "_use_eigen", use_eigen)
        object.__setattr__(obj, "_expm_mat", expm_mat)
        object.__setattr__(obj, "_phi1_mat", phi1_mat)
        object.__setattr__(obj, "_phi2_mat", phi2_mat)
        return obj

    @property
    def state_dim(self) -> int:
        return int(self.matrix.shape[0])

    # -- internal helpers ---------------------------------------------------

    def _mat(self, dtype: jnp.dtype, *, transpose: bool = False) -> Array:
        m = self.matrix.astype(dtype)
        return m.T if transpose else m

    def _default_dtype(self, dtype: Optional[jnp.dtype]) -> jnp.dtype:
        return jnp.dtype(dtype or self.matrix.dtype)

    def _eigen_components(self, dtype: jnp.dtype, *, transpose: bool = False) -> tuple[Array, Array, Array]:
        """Return (eigvals, V, V_inv) cast to a complex dtype.

        When *transpose* is True the returned pair (V, V_inv) is swapped and
        transposed so that ``V diag(f) V_inv`` gives the *transpose* of the
        original operator.
        """
        cdtype = _complex_dtype_for(dtype)
        eigvals = self._eigvals.astype(cdtype)
        V = self._V.astype(cdtype)
        V_inv = self._V_inv.astype(cdtype)
        if transpose:
            V, V_inv = V_inv.T, V.T
        return eigvals, V, V_inv

    @staticmethod
    def _spectral_fn(x: Array, kind: str) -> Array:
        """Evaluate the scalar spectral function for each eigenvalue."""
        if kind == "expm":
            return jnp.exp(-x)
        if kind == "phi1":
            return _phi1_scalar_neg(x)
        if kind == "phi2":
            return _phi2_scalar_neg(x)
        raise ValueError(f"unknown dense operator kind {kind!r}")

    def _dense_op(self, dt: Array, *, dtype: jnp.dtype, kind: str, transpose: bool = False) -> Array:
        """Materialise the full operator matrix via eigendecomposition (or Padé fallback)."""
        if not self._use_eigen:
            mat = self._mat(dtype, transpose=transpose)
            if kind == "expm":
                return _dense_expm_batched(mat, dt.astype(dtype))
            if kind == "phi1":
                return _dense_phi1_batched(mat, dt.astype(dtype))
            if kind == "phi2":
                return _dense_phi2_batched(mat, dt.astype(dtype))
            raise ValueError(f"unknown dense operator kind {kind!r}")
        eigvals, V, V_inv = self._eigen_components(dtype, transpose=transpose)
        cdtype = V.dtype
        x = dt[:, None].astype(cdtype) * eigvals[None, :]  # (m, R)
        f = self._spectral_fn(x, kind)                      # (m, R)
        result = jnp.einsum("ij,mj,jk->mik", V, f, V_inv)
        if jnp.issubdtype(dtype, jnp.floating):
            return result.real.astype(dtype)
        return result.astype(dtype)

    # -- unified action / solve (O(R²) via eigendecomposition) ---------------

    def _action(
        self, dt: Array | float, x: Array, *, dtype: jnp.dtype, kind: str, side: _SIDE,
    ) -> Array:
        dt, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)
        name = "lhs" if side == "right" else "rhs"

        if not self._use_eigen:
            # Fallback: materialise full operator, then multiply.
            if side == "right" and was_scalar:
                lhs = jnp.asarray(x, dtype=dtype)
                if lhs.ndim < 2:
                    raise ValueError("lhs must have shape (..., k, R).")
                op_t = self._dense_op(dt, dtype=dtype, kind=kind, transpose=True)[0]
                return _right_multiply(lhs, op_t)

            def scalar_action_dense(x_flat: Array) -> Array:
                op = self._dense_op(dt, dtype=x_flat.dtype, kind=kind)[0]
                return _left_multiply(op, x_flat)

            def batched_action_dense(dt_vec: Array, x_flat: Array) -> Array:
                op = self._dense_op(dt_vec, dtype=x_flat.dtype, kind=kind)
                return jnp.einsum("mab,mbc->mac", op, x_flat)

            return _apply_dt_action(
                dt, was_scalar, x, dtype=dtype, side=side, name=name,
                scalar_action=scalar_action_dense, batched_action=batched_action_dense,
            )

        # --- Eigen path (O(R²) per column) ---
        is_real = jnp.issubdtype(dtype, jnp.floating)

        # Precompute eigen components (un-transposed; the flatten/unflatten
        # path handles side="right" via axis swaps).
        eigvals, V, V_inv = self._eigen_components(dtype, transpose=False)
        cdtype = V.dtype

        # Fast path: scalar dt, right multiply — O(R²) without materialising.
        if side == "right" and was_scalar:
            lhs = jnp.asarray(x, dtype=dtype)
            if lhs.ndim < 2:
                raise ValueError("lhs must have shape (..., k, R).")
            f = self._spectral_fn(dt[0].astype(cdtype) * eigvals, kind)  # (R,)
            # lhs @ op^T  =  lhs @ V_inv^T diag(f) V^T
            result = _right_multiply(
                _right_multiply(lhs.astype(cdtype), V_inv.T) * f,
                V.T,
            )
            return result.real.astype(dtype) if is_real else result.astype(dtype)

        def scalar_action(x_flat: Array) -> Array:
            f = self._spectral_fn(dt[0].astype(cdtype) * eigvals, kind)  # (R,)
            projected = _left_multiply(V_inv, x_flat.astype(cdtype))
            result = _left_multiply(V, f[None, :, None] * projected)
            return result.real.astype(dtype) if is_real else result.astype(dtype)

        def batched_action(dt_vec: Array, x_flat: Array) -> Array:
            x_dt = dt_vec[:, None].astype(cdtype) * eigvals[None, :]  # (m, R)
            f = self._spectral_fn(x_dt, kind)                         # (m, R)
            projected = jnp.einsum("ab,mbc->mac", V_inv, x_flat.astype(cdtype))
            result = jnp.einsum("ab,mbc->mac", V, f[:, :, None] * projected)
            return result.real.astype(dtype) if is_real else result.astype(dtype)

        return _apply_dt_action(
            dt, was_scalar, x, dtype=dtype, side=side, name=name,
            scalar_action=scalar_action, batched_action=batched_action,
        )

    def _solve(
        self, zeta: complex | Array, dt: Array | float, rhs: Array,
        *, dtype: jnp.dtype, transpose: bool,
    ) -> Array:
        dt, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)

        if not self._use_eigen:
            # Fallback: direct dense linear solve.
            mat = self._mat(dtype, transpose=transpose)
            eye = jnp.eye(self.state_dim, dtype=dtype)

            def solve_dense(zeta_vec: Array, dt_vec: Array, rhs_flat: Array) -> Array:
                system = (
                    zeta_vec[:, None, None] * eye[None, :, :]
                    + dt_vec.astype(dtype)[:, None, None] * mat[None, :, :]
                )
                return jnp.linalg.solve(system, rhs_flat)

            return _apply_dt_solve(
                zeta, dt, was_scalar, rhs, dtype=dtype,
                scalar_solve=solve_dense, batched_solve=solve_dense,
            )

        # --- Eigen path (O(R²) per column) ---
        eigvals, V, V_inv = self._eigen_components(dtype, transpose=transpose)
        cdtype = V.dtype

        def solve(zeta_vec: Array, dt_vec: Array, rhs_flat: Array) -> Array:
            # (zeta I + dt Lambda)^{-1} = V diag(1/(zeta + dt*eigval)) V_inv
            denom = (zeta_vec[:, None].astype(cdtype)
                     + dt_vec[:, None].astype(cdtype) * eigvals[None, :])  # (m, R)
            inv_denom = 1.0 / denom
            projected = jnp.einsum("ab,mbc->mac", V_inv, rhs_flat.astype(cdtype))
            result = jnp.einsum("ab,mbc->mac", V, inv_denom[:, :, None] * projected)
            return result.astype(dtype)

        return _apply_dt_solve(
            zeta, dt, was_scalar, rhs, dtype=dtype,
            scalar_solve=solve, batched_solve=solve,
        )

    # -- precomputation helpers --------------------------------------------

    def _with_precomputed(self, **kwargs) -> "DenseLambda":
        """Return a copy with selected precomputed matrices replaced."""
        obj = object.__new__(DenseLambda)
        object.__setattr__(obj, "matrix", self.matrix)
        object.__setattr__(obj, "_eigvals", self._eigvals)
        object.__setattr__(obj, "_V", self._V)
        object.__setattr__(obj, "_V_inv", self._V_inv)
        object.__setattr__(obj, "_use_eigen", self._use_eigen)
        object.__setattr__(obj, "_expm_mat", kwargs.get("_expm_mat", self._expm_mat))
        object.__setattr__(obj, "_phi1_mat", kwargs.get("_phi1_mat", self._phi1_mat))
        object.__setattr__(obj, "_phi2_mat", kwargs.get("_phi2_mat", self._phi2_mat))
        return obj

    def precompute_expm(self, dt, *, dtype=None) -> "DenseLambda":
        d = self._default_dtype(dtype)
        dt_arr, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)
        mat = self._dense_op(dt_arr, dtype=d, kind="expm")
        return self._with_precomputed(_expm_mat=mat[0] if was_scalar else mat)

    def precompute_phi1(self, dt, *, dtype=None) -> "DenseLambda":
        d = self._default_dtype(dtype)
        dt_arr, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)
        mat = self._dense_op(dt_arr, dtype=d, kind="phi1")
        return self._with_precomputed(_phi1_mat=mat[0] if was_scalar else mat)

    def precompute_phi2(self, dt, *, dtype=None) -> "DenseLambda":
        d = self._default_dtype(dtype)
        dt_arr, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)
        mat = self._dense_op(dt_arr, dtype=d, kind="phi2")
        return self._with_precomputed(_phi2_mat=mat[0] if was_scalar else mat)

    def __getitem__(self, ix) -> "DenseLambda":
        def _s(m): return m if (m is None or m.ndim == 2) else m[ix]
        return self._with_precomputed(
            _expm_mat=_s(self._expm_mat),
            _phi1_mat=_s(self._phi1_mat),
            _phi2_mat=_s(self._phi2_mat),
        )

    # -- public API ---------------------------------------------------------

    def expm(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        dt, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)
        d = self._default_dtype(dtype)
        return _restore_dt_api(self._dense_op(dt, dtype=d, kind="expm"), was_scalar)

    def phi1(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        dt, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)
        d = self._default_dtype(dtype)
        return _restore_dt_api(self._dense_op(dt, dtype=d, kind="phi1"), was_scalar)

    def phi2(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        dt, was_scalar = _normalize_dt_api(dt, dtype=self.matrix.dtype)
        d = self._default_dtype(dtype)
        return _restore_dt_api(self._dense_op(dt, dtype=d, kind="phi2"), was_scalar)

    def solve_shifted(self, zeta, dt, rhs, *, dtype=None) -> Array:
        return self._solve(zeta, dt, rhs, dtype=_complex_dtype_for(self._default_dtype(dtype)), transpose=False)

    def solve_shifted_transpose(self, zeta, dt, rhs, *, dtype=None) -> Array:
        return self._solve(zeta, dt, rhs, dtype=_complex_dtype_for(self._default_dtype(dtype)), transpose=True)

    def lambda_multiply_left(self, rhs, *, dtype=None) -> Array:
        rhs = jnp.asarray(rhs, dtype=self._default_dtype(dtype))
        return _left_multiply(self._mat(rhs.dtype), rhs)

    def lambda_multiply_right(self, lhs, *, dtype=None) -> Array:
        lhs = jnp.asarray(lhs, dtype=self._default_dtype(dtype))
        return _right_multiply(lhs, self._mat(lhs.dtype, transpose=True))

    def expm_multiply_left(self, dt, rhs, *, dtype=None) -> Array:
        d = self._default_dtype(dtype)
        if self._expm_mat is not None:
            return jnp.einsum("ab,...bc->...ac", self._expm_mat.astype(d), jnp.asarray(rhs, dtype=d))
        return self._action(dt, rhs, dtype=d, kind="expm", side="left")

    def expm_multiply_right(self, dt, lhs, *, dtype=None) -> Array:
        d = self._default_dtype(dtype)
        if self._expm_mat is not None:
            return jnp.einsum("...ab,cb->...ac", jnp.asarray(lhs, dtype=d), self._expm_mat.astype(d))
        return self._action(dt, lhs, dtype=d, kind="expm", side="right")

    def phi1_multiply_left(self, dt, rhs, *, dtype=None) -> Array:
        d = self._default_dtype(dtype)
        if self._phi1_mat is not None:
            return jnp.einsum("ab,...bc->...ac", self._phi1_mat.astype(d), jnp.asarray(rhs, dtype=d))
        return self._action(dt, rhs, dtype=d, kind="phi1", side="left")

    def phi1_multiply_right(self, dt, lhs, *, dtype=None) -> Array:
        d = self._default_dtype(dtype)
        if self._phi1_mat is not None:
            return jnp.einsum("...ab,cb->...ac", jnp.asarray(lhs, dtype=d), self._phi1_mat.astype(d))
        return self._action(dt, lhs, dtype=d, kind="phi1", side="right")

    def phi2_multiply_left(self, dt, rhs, *, dtype=None) -> Array:
        d = self._default_dtype(dtype)
        if self._phi2_mat is not None:
            return jnp.einsum("ab,...bc->...ac", self._phi2_mat.astype(d), jnp.asarray(rhs, dtype=d))
        return self._action(dt, rhs, dtype=d, kind="phi2", side="left")

    def phi2_multiply_right(self, dt, lhs, *, dtype=None) -> Array:
        d = self._default_dtype(dtype)
        if self._phi2_mat is not None:
            return jnp.einsum("...ab,cb->...ac", jnp.asarray(lhs, dtype=d), self._phi2_mat.astype(d))
        return self._action(dt, lhs, dtype=d, kind="phi2", side="right")


# ---------------------------------------------------------------------------
# Jordan-chain coefficient helpers
# ---------------------------------------------------------------------------


def _factorials(n: int, dtype: jnp.dtype) -> Array:
    """Return ``[0!, 1!, ..., (n-1)!]``."""
    vals = np.empty(n, dtype=np.float64)
    vals[0] = 1.0
    for k in range(1, n):
        vals[k] = vals[k - 1] * k
    return jnp.array(vals, dtype=dtype)


@jax.jit
def _phi1_scalar_neg(x: Array) -> Array:
    """Return the stable scalar kernel ``phi_1(-x) = (1 - e^{-x}) / x``."""
    one = jnp.ones_like(x)
    safe_x = jnp.where(x == 0, one, x)
    out = -jnp.expm1(-safe_x) / safe_x
    return jnp.where(x == 0, one, out)


@jax.jit
def _phi2_scalar_neg(x: Array) -> Array:
    r"""Return the stable scalar kernel ``\varphi_2(-x) = (e^{-x} + x - 1) / x^2``."""
    half = 0.5 * jnp.ones_like(x)
    safe_x = jnp.where(x == 0, jnp.ones_like(x), x)
    out = (jnp.exp(-safe_x) + safe_x - 1.0) / (safe_x * safe_x)
    return jnp.where(x == 0, half, out)


def _real_expm_coeffs(rates: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    """Return ``e^{-rate dt} dt^k / k!``."""
    k = jnp.arange(nmax, dtype=dtype)
    fac = _factorials(nmax, dtype)
    return (
        jnp.exp(-dt[:, None] * rates[None, :])[:, :, None]
        * (dt[:, None, None] ** k[None, None, :])
        / fac[None, None, :]
    )


# Number of Taylor terms used when |rate * dt| is small in phi1 recurrence.
_PHI1_TAYLOR_TERMS = 20
_PHI1_SMALL_THRESHOLD = 0.05


def _phi1_chain_coeffs(rates: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    """Return chain coefficients of ``phi_1(-dt(J_rate))`` for real or complex rates.

    Uses a Taylor-series fallback when ``|rate * dt|`` is small to avoid
    catastrophic cancellation in the ``(prev - edge) / rate`` recurrence.
    """
    if nmax == 0:
        return jnp.zeros((dt.shape[0], rates.shape[0], 0), dtype=dtype)

    real_dtype = jnp.real(jnp.asarray(0, dtype=dtype)).dtype
    dt = dt.astype(real_dtype)
    rates = rates.astype(dtype)
    x = dt[:, None] * rates[None, :]
    exp_term = jnp.exp(-x)
    c0 = _phi1_scalar_neg(x)
    if nmax == 1:
        return c0[:, :, None]

    fac = _factorials(nmax + _PHI1_TAYLOR_TERMS + 2, dtype)
    zero_mask = rates[None, :] == 0
    small = jnp.abs(x) < _PHI1_SMALL_THRESHOLD

    # Pre-compute Taylor powers for fallback: (-x)^j for j in [0, T)
    js = jnp.arange(_PHI1_TAYLOR_TERMS, dtype=dtype)
    neg_x_powers = (-x)[:, :, None] ** js[None, None, :]  # (m, b, T)

    def _taylor_c(k: Array) -> Array:
        """Compute c_k via truncated Taylor series for small |x|."""
        # c_k = dt^k * sum_j (-x)^j / (k+j+1)!
        js = jnp.arange(_PHI1_TAYLOR_TERMS)
        fac_denom = fac[k + js + 1]  # (T,)
        return (dt[:, None] ** k) * jnp.sum(neg_x_powers / fac_denom[None, None, :], axis=-1)

    def body(prev: Array, k: Array) -> tuple[Array, Array]:
        edge = exp_term * (dt[:, None] ** (k - 1)) / fac[k]
        nxt_recur = jnp.where(
            zero_mask,
            (dt[:, None] ** k) / fac[k + 1],
            (prev - edge) / rates[None, :],
        )
        nxt_taylor = _taylor_c(k)
        nxt = jnp.where(small, nxt_taylor, nxt_recur)
        return nxt, nxt

    _, tail = jax.lax.scan(body, c0, jnp.arange(1, nmax))
    return jnp.concatenate([c0[:, :, None], jnp.moveaxis(tail, 0, -1)], axis=-1)


# Number of Taylor terms used when |rate * dt| is small in phi2 recurrence.
_PHI2_TAYLOR_TERMS = 20
_PHI2_SMALL_THRESHOLD = 0.05


def _phi2_chain_coeffs(rates: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    r"""Return chain coefficients of ``\varphi_2(-dt\,J_{\mathrm{rate}})`` for real or complex rates.

    Uses the recurrence ``d_k = (dt\,d_{k-1} - c_k^{(1)}) / x`` where
    ``c_k^{(1)}`` are the ``\varphi_1`` chain coefficients and ``x = dt \cdot \mathrm{rate}``.
    Falls back to a Taylor series when ``|x|`` is small.
    """
    if nmax == 0:
        return jnp.zeros((dt.shape[0], rates.shape[0], 0), dtype=dtype)

    real_dtype = jnp.real(jnp.asarray(0, dtype=dtype)).dtype
    dt = dt.astype(real_dtype)
    rates = rates.astype(dtype)
    x = dt[:, None] * rates[None, :]  # (m, b)

    # phi1 chain coefficients (needed for the recurrence)
    phi1_c = _phi1_chain_coeffs(rates, nmax, dt, dtype)  # (m, b, nmax)

    d0 = _phi2_scalar_neg(x)  # (m, b)
    if nmax == 1:
        return d0[:, :, None]

    fac = _factorials(nmax + _PHI2_TAYLOR_TERMS + 4, dtype)
    zero_mask = rates[None, :] == 0
    small = jnp.abs(x) < _PHI2_SMALL_THRESHOLD

    # Pre-compute Taylor powers for fallback
    js = jnp.arange(_PHI2_TAYLOR_TERMS)  # integer dtype for indexing into fac
    neg_x_powers = (-x)[:, :, None] ** js[None, None, :].astype(dtype)  # (m, b, T)

    def _taylor_d(k: Array) -> Array:
        r"""Compute d_k via truncated Taylor series for small |x|.

        d_k = dt^k \sum_j (-x)^j / (k!\,j!\,(k+j+1)(k+j+2))
        """
        denom = fac[k] * fac[js] * (k + js + 1) * (k + js + 2)
        return (dt[:, None] ** k) * jnp.sum(neg_x_powers / denom[None, None, :], axis=-1)

    # phi1 coefficients for k=1..nmax-1, reshaped for scan: (nmax-1, m, b)
    phi1_tail = jnp.moveaxis(phi1_c[:, :, 1:], -1, 0)

    def body(prev: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        k, ck = inputs
        # Recurrence: d_k = (dt * d_{k-1} - c_k^{(1)}) / x
        nxt_recur = jnp.where(
            zero_mask,
            (dt[:, None] ** k) / fac[k + 2],
            (dt[:, None] * prev - ck) / x,
        )
        nxt_taylor = _taylor_d(k)
        nxt = jnp.where(small, nxt_taylor, nxt_recur)
        return nxt, nxt

    _, tail = jax.lax.scan(body, d0, (jnp.arange(1, nmax), phi1_tail))
    return jnp.concatenate([d0[:, :, None], jnp.moveaxis(tail, 0, -1)], axis=-1)


def _shifted_coeffs(rates: Array, nmax: int, zeta: Array, dt: Array, dtype: jnp.dtype) -> Array:
    """Return ``dt^k / (zeta + dt rate)^{k+1}`` via cumulative product.

    Uses a stable cumulative-ratio formulation instead of explicit powers.
    """
    if nmax == 0:
        return jnp.zeros((dt.shape[0], rates.shape[0], 0), dtype=dtype)

    rates = rates.astype(dtype)
    a = zeta[:, None] + dt[:, None].astype(dtype) * rates[None, :]  # (m, b)
    inv_a = 1.0 / a  # (m, b)
    ratio = dt[:, None].astype(dtype) * inv_a  # dt / a, shape (m, b)

    c0 = inv_a  # (m, b)
    if nmax == 1:
        return c0[:, :, None]

    def scan_body(carry: Array, _: Array) -> tuple[Array, Array]:
        nxt = carry * ratio
        return nxt, nxt

    _, tail = jax.lax.scan(scan_body, c0, None, length=nmax - 1)
    return jnp.concatenate([c0[:, :, None], jnp.moveaxis(tail, 0, -1)], axis=-1)


# ---------------------------------------------------------------------------
# Complex ↔ real-2×2 helpers
# ---------------------------------------------------------------------------


@jax.jit
def _complex_to_real2x2(z: Array) -> Array:
    """Map complex scalars to their real ``2 x 2`` representation."""
    re, im = z.real, z.imag
    return jnp.stack(
        [jnp.stack([re, -im], axis=-1), jnp.stack([im, re], axis=-1)],
        axis=-2,
    )


def _osc_rates(decays: Array, freqs: Array, dtype: jnp.dtype, *, transpose: bool = False) -> Array:
    """Return complex rates representing real ``2 x 2`` oscillatory blocks.

    The un-transposed block convention is ``[[μ, -ω], [ω, μ]]``, whose
    eigenvalue with positive imaginary part is ``μ + iω``.
    """
    real_dtype = jnp.finfo(dtype).dtype
    sign = -1.0 if transpose else 1.0
    return decays.astype(real_dtype) + (1j * sign) * freqs.astype(real_dtype)


def _osc_expm_coeffs(decays: Array, freqs: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    if nmax == 0:
        return jnp.zeros((dt.shape[0], decays.shape[0], 0, 2, 2), dtype=dtype)
    cdtype = _complex_dtype_for(dtype)
    return _complex_to_real2x2(
        _real_expm_coeffs(_osc_rates(decays, freqs, cdtype), nmax, dt, cdtype)
    )


def _osc_phi1_coeffs(decays: Array, freqs: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    if nmax == 0:
        return jnp.zeros((dt.shape[0], decays.shape[0], 0, 2, 2), dtype=dtype)
    cdtype = _complex_dtype_for(dtype)
    return _complex_to_real2x2(
        _phi1_chain_coeffs(_osc_rates(decays, freqs, cdtype), nmax, dt, cdtype)
    )


def _osc_phi2_coeffs(decays: Array, freqs: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    if nmax == 0:
        return jnp.zeros((dt.shape[0], decays.shape[0], 0, 2, 2), dtype=dtype)
    cdtype = _complex_dtype_for(dtype)
    return _complex_to_real2x2(
        _phi2_chain_coeffs(_osc_rates(decays, freqs, cdtype), nmax, dt, cdtype)
    )


def _osc_shifted_coeffs(
    decays: Array, freqs: Array, nmax: int, zeta: Array, dt: Array, dtype: jnp.dtype,
    *, transpose: bool,
) -> Array:
    if nmax == 0:
        return jnp.zeros((dt.shape[0], decays.shape[0], 0, 2, 2), dtype=dtype)
    cdtype = _complex_dtype_for(dtype)
    return _complex_to_real2x2(
        _shifted_coeffs(_osc_rates(decays, freqs, cdtype, transpose=transpose),
                        nmax, zeta.astype(cdtype), dt, cdtype)
    )


# ---------------------------------------------------------------------------
# Jordan chain contraction
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _chain_offset_indices(nmax: int, *, lower: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return Toeplitz-style offset indices for Jordan chain contractions.

    Cached so that repeated calls inside JIT-traced code re-use the same
    constant arrays without re-tracing.
    """
    i = np.arange(nmax)[:, None]
    j = np.arange(nmax)[None, :]
    k = (i - j) if lower else (j - i)
    mask = (k >= 0) & (k < nmax)
    idx = np.where(mask, k, 0).astype(np.int32)
    return idx, mask


@partial(jax.jit, static_argnames=("lower",))
def _apply_chain(coeff: Array, rhs: Array, *, lower: bool) -> Array:
    """Apply ``sum_k c_k N^k`` (or ``(N^T)^k``) to scalar or matrix chains.

    Automatically dispatches based on coefficient ndim:
    - scalar chains: coeff shape ``(t, b, n)``, rhs ``(t, b, n, c)``
    - matrix chains: coeff shape ``(t, b, n, u, v)``, rhs ``(t, b, n, v, c)``
    """
    nmax = rhs.shape[2]
    idx, mask = _chain_offset_indices(nmax, lower=lower)
    idx_j, mask_j = jnp.asarray(idx), jnp.asarray(mask)
    if coeff.ndim == 3:
        # Scalar chain
        weights = jnp.take(coeff, idx_j, axis=2) * mask_j[None, None, :, :]
        return jnp.einsum("tbij,tbjc->tbic", weights, rhs)
    else:
        # Matrix (2x2 block) chain
        weights = jnp.take(coeff, idx_j, axis=2) * mask_j[None, None, :, :, None, None]
        return jnp.einsum("tbijuv,tbjvc->tbiuc", weights, rhs)


# ---------------------------------------------------------------------------
# Jordan-structured apply / solve  (batched)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("real_sizes", "osc_sizes"))
def _jordan_lambda_left_unbatched(
    real_rates: Array,
    real_sizes: tuple[int, ...],
    osc_decays: Array,
    osc_freqs: Array,
    osc_sizes: tuple[int, ...],
    rhs: Array,
) -> Array:
    """Apply ``Lambda`` on the left without ``dt`` pairing."""
    parts: list[Array] = []
    start = 0

    for i, size in enumerate(real_sizes):
        block = rhs[..., start : start + size, :]
        start += size
        out = real_rates[i] * block
        if size > 1:
            out = out.at[..., :-1, :].add(-block[..., 1:, :])
        parts.append(out)

    for i, size in enumerate(osc_sizes):
        width = 2 * size
        block = rhs[..., start : start + width, :].reshape(
            rhs.shape[:-2] + (size, 2, rhs.shape[-1])
        )
        start += width
        base = jnp.array(
            [[osc_decays[i], -osc_freqs[i]], [osc_freqs[i], osc_decays[i]]],
            dtype=rhs.dtype,
        )
        out = jnp.einsum("uv,...nvc->...nuc", base, block)
        if size > 1:
            out = out.at[..., :-1, :, :].add(-block[..., 1:, :, :])
        parts.append(out.reshape(rhs.shape[:-2] + (width, rhs.shape[-1])))

    return jnp.concatenate(parts, axis=-2) if parts else jnp.zeros_like(rhs)


@partial(
    jax.jit,
    static_argnames=("real_sizes", "osc_sizes", "lower", "real_coeffs", "osc_coeffs"),
)
def _jordan_apply_batched(
    real_rates: Array,
    real_sizes: tuple[int, ...],
    osc_decays: Array,
    osc_freqs: Array,
    osc_sizes: tuple[int, ...],
    dt: Array,
    rhs: Array,
    *,
    lower: bool,
    real_coeffs: Callable[[Array, int, Array, jnp.dtype], Array],
    osc_coeffs: Callable[[Array, Array, int, Array, jnp.dtype], Array],
) -> Array:
    """Apply structured Jordan operators blockwise to paired batched operands."""
    batch, _, cols = rhs.shape
    parts: list[Array] = []
    start = 0
    dt_real = dt.astype(rhs.real.dtype)

    for i, size in enumerate(real_sizes):
        block = rhs[:, start : start + size, :]
        start += size
        coeff = real_coeffs(real_rates[i : i + 1], size, dt_real, rhs.dtype)
        y = _apply_chain(coeff, block[:, None, :, :], lower=lower)[:, 0, :, :]
        parts.append(y)

    for i, size in enumerate(osc_sizes):
        width = 2 * size
        block = rhs[:, start : start + width, :].reshape(batch, size, 2, cols)
        start += width
        coeff = osc_coeffs(osc_decays[i : i + 1], osc_freqs[i : i + 1], size, dt_real, rhs.dtype)
        y = _apply_chain(coeff, block[:, None, :, :, :], lower=lower)[:, 0, :, :, :]
        parts.append(y.reshape(batch, width, cols))

    return jnp.concatenate(parts, axis=1) if parts else jnp.zeros_like(rhs)


@partial(jax.jit, static_argnames=("real_sizes", "osc_sizes", "transpose"))
def _jordan_solve_batched(
    real_rates: Array,
    real_sizes: tuple[int, ...],
    osc_decays: Array,
    osc_freqs: Array,
    osc_sizes: tuple[int, ...],
    zeta: Array,
    dt: Array,
    rhs: Array,
    *,
    transpose: bool,
) -> Array:
    """Solve structured shifted systems blockwise for paired batched operands."""
    batch, _, cols = rhs.shape
    parts: list[Array] = []
    start = 0
    dt_real = dt.astype(rhs.real.dtype)
    zeta = zeta.astype(rhs.dtype)

    for i, size in enumerate(real_sizes):
        block = rhs[:, start : start + size, :]
        start += size
        coeff = _shifted_coeffs(real_rates[i : i + 1], size, zeta, dt_real, rhs.dtype)
        y = _apply_chain(coeff, block[:, None, :, :], lower=transpose)[:, 0, :, :]
        parts.append(y)

    for i, size in enumerate(osc_sizes):
        width = 2 * size
        block = rhs[:, start : start + width, :]
        start += width
        # Direct dense solve for oscillatory blocks — the structured approach
        # via _osc_shifted_coeffs does not handle complex zeta correctly.
        osc_block = _osc_lambda_block_matrix(osc_decays[i], osc_freqs[i], size, rhs.real.dtype)
        def _solve_osc(h: Array, rhs_slice: Array) -> Array:
            mat = zeta[0] * jnp.eye(width, dtype=rhs.dtype) + h.astype(rhs.dtype) * osc_block.astype(rhs.dtype)
            if transpose:
                mat = mat.T
            return jnp.linalg.solve(mat, rhs_slice)
        y = jax.vmap(_solve_osc)(dt_real, block)
        parts.append(y)

    return jnp.concatenate(parts, axis=1) if parts else jnp.zeros_like(rhs)


# ---------------------------------------------------------------------------
# Block-diagonal assembly helpers
# ---------------------------------------------------------------------------


def _osc_lambda_block_matrix(decay: Array, freq: Array, size: int, dtype: jnp.dtype) -> Array:
    """Build the real 2×2 block-Jordan matrix for one oscillatory chain."""
    width = 2 * size
    base = jnp.array([[decay, -freq], [freq, decay]], dtype=dtype)
    neg_eye2 = -jnp.eye(2, dtype=dtype)

    def _body(k, block):
        row = 2 * k
        block = jax.lax.dynamic_update_slice(block, base, (row, row))
        off = jax.lax.dynamic_update_slice(block, neg_eye2, (row, row + 2))
        return jnp.where(k + 1 < size, off, block)

    return jax.lax.fori_loop(0, size, _body, jnp.zeros((width, width), dtype=dtype))


def _jordan_block_diag(blocks: list[Array], batch: int, dtype: jnp.dtype) -> Array:
    """Assemble a batched block-diagonal matrix via ``jax.scipy.linalg.block_diag``.

    Each element of *blocks* has shape ``(batch, w_i, w_i)``.  We vmap the
    scalar ``block_diag`` over the batch axis.
    """
    if not blocks:
        return jnp.zeros((batch, 0, 0), dtype=dtype)
    return jax.vmap(lambda *bs: jsp_linalg.block_diag(*bs))(*blocks)


def _real_chain_matrix_from_coeff(coeff: Array, size: int) -> Array:
    """Materialize a batched upper-triangular scalar Jordan chain matrix."""
    idx, mask = _chain_offset_indices(size, lower=False)
    idx_j, mask_j = jnp.asarray(idx), jnp.asarray(mask)
    return jnp.take(coeff[:, 0, :], idx_j, axis=1) * mask_j[None, :, :]


def _osc_chain_matrix_from_coeff(coeff: Array, size: int) -> Array:
    """Materialize a batched upper-triangular real 2x2 block-Jordan matrix."""
    idx, mask = _chain_offset_indices(size, lower=False)
    idx_j, mask_j = jnp.asarray(idx), jnp.asarray(mask)
    blocks = jnp.take(coeff[:, 0, :, :, :], idx_j, axis=1) * mask_j[None, :, :, None, None]
    return jnp.transpose(blocks, (0, 1, 3, 2, 4)).reshape(coeff.shape[0], 2 * size, 2 * size)


# ---------------------------------------------------------------------------
# JordanLambda
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class JordanLambda(Lambda):
    r"""Structured real-Jordan implementation of :class:`Lambda`.

    Real scalar poles are represented by upper-triangular Jordan chains, while
    oscillatory pole pairs are represented by real ``2 x 2`` block-Jordan
    chains.

    Rates / decays / frequencies are stored as JAX arrays so that they are
    pytree *leaves*.  This means JIT does **not** re-trace when only rate
    values change (critical for parameter optimisation and autodiff through
    rates).  Block *sizes* remain static (pytree auxiliary data).
    """

    real_rates: Array = field(default_factory=lambda: jnp.zeros(0))
    real_sizes: tuple[int, ...] = field(default_factory=tuple, metadata={"static": True})
    osc_decays: Array = field(default_factory=lambda: jnp.zeros(0))
    osc_freqs: Array = field(default_factory=lambda: jnp.zeros(0))
    osc_sizes: tuple[int, ...] = field(default_factory=tuple, metadata={"static": True})
    _expm_blocks: Optional[tuple] = field(init=False, repr=False, default=None)
    _phi1_blocks: Optional[tuple] = field(init=False, repr=False, default=None)
    _phi2_blocks: Optional[tuple] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        real_rates = jnp.asarray(self.real_rates, dtype=jnp.float64).ravel()
        real_sizes = _as_int_tuple(self.real_sizes)
        osc_decays = jnp.asarray(self.osc_decays, dtype=jnp.float64).ravel()
        osc_freqs = jnp.asarray(self.osc_freqs, dtype=jnp.float64).ravel()
        osc_sizes = _as_int_tuple(self.osc_sizes)

        if real_rates.shape[0] != len(real_sizes):
            raise ValueError("real_rates and real_sizes must have the same length.")
        if osc_decays.shape[0] != osc_freqs.shape[0] or osc_decays.shape[0] != len(osc_sizes):
            raise ValueError("osc_decays, osc_freqs and osc_sizes must have the same length.")
        if any(s <= 0 for s in real_sizes):
            raise ValueError("real_sizes must be strictly positive.")
        if any(s <= 0 for s in osc_sizes):
            raise ValueError("osc_sizes must be strictly positive.")
        if len(osc_sizes) > 0 and jnp.any(osc_freqs <= 0):
            raise ValueError("osc_freqs must be strictly positive.")

        object.__setattr__(self, "real_rates", real_rates)
        object.__setattr__(self, "real_sizes", real_sizes)
        object.__setattr__(self, "osc_decays", osc_decays)
        object.__setattr__(self, "osc_freqs", osc_freqs)
        object.__setattr__(self, "osc_sizes", osc_sizes)

    # -- pytree: rates are leaves, sizes are static -----------------------

    def tree_flatten(self):
        has_expm = self._expm_blocks is not None
        has_phi1 = self._phi1_blocks is not None
        has_phi2 = self._phi2_blocks is not None
        leaves = [self.real_rates, self.osc_decays, self.osc_freqs]
        if has_expm:
            leaves.extend(self._expm_blocks)
        if has_phi1:
            leaves.extend(self._phi1_blocks)
        if has_phi2:
            leaves.extend(self._phi2_blocks)
        aux = (self.real_sizes, self.osc_sizes, has_expm, has_phi1, has_phi2)
        return tuple(leaves), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        real_sizes, osc_sizes, has_expm, has_phi1, has_phi2 = aux_data
        n_blocks = len(real_sizes) + len(osc_sizes)
        real_rates, osc_decays, osc_freqs = children[0], children[1], children[2]
        rest = list(children[3:])
        expm_blocks = phi1_blocks = phi2_blocks = None
        if has_expm:
            expm_blocks = tuple(rest[:n_blocks])
            rest = rest[n_blocks:]
        if has_phi1:
            phi1_blocks = tuple(rest[:n_blocks])
            rest = rest[n_blocks:]
        if has_phi2:
            phi2_blocks = tuple(rest[:n_blocks])
        obj = object.__new__(cls)
        object.__setattr__(obj, "real_rates", real_rates)
        object.__setattr__(obj, "real_sizes", real_sizes)
        object.__setattr__(obj, "osc_decays", osc_decays)
        object.__setattr__(obj, "osc_freqs", osc_freqs)
        object.__setattr__(obj, "osc_sizes", osc_sizes)
        object.__setattr__(obj, "_expm_blocks", expm_blocks)
        object.__setattr__(obj, "_phi1_blocks", phi1_blocks)
        object.__setattr__(obj, "_phi2_blocks", phi2_blocks)
        return obj

    @property
    def state_dim(self) -> int:
        return sum(self.real_sizes) + 2 * sum(self.osc_sizes)

    # -- unified internal helpers ------------------------------------------

    def _action(
        self, dt: Array | float, x: Array, *, dtype: jnp.dtype, side: _SIDE,
        real_coeffs: Callable, osc_coeffs: Callable,
    ) -> Array:
        """Unified structured action for expm / phi1, left / right."""
        dt, was_scalar = _normalize_dt_api(dt, dtype=jnp.float64)
        name = "lhs" if side == "right" else "rhs"

        def batched(dt_vec: Array, x_flat: Array) -> Array:
            return _jordan_apply_batched(
                self.real_rates.astype(x_flat.dtype),
                self.real_sizes,
                self.osc_decays.astype(x_flat.real.dtype),
                self.osc_freqs.astype(x_flat.real.dtype),
                self.osc_sizes,
                dt_vec.astype(x_flat.real.dtype),
                x_flat,
                lower=False,
                real_coeffs=real_coeffs,
                osc_coeffs=osc_coeffs,
            )

        def scalar_action(x_flat: Array) -> Array:
            return batched(
                jnp.full((x_flat.shape[0],), dt[0], dtype=x_flat.real.dtype),
                x_flat,
            )

        return _apply_dt_action(
            dt, was_scalar, x, dtype=dtype, side=side, name=name,
            scalar_action=scalar_action, batched_action=batched,
        )

    def _solve(
        self, zeta: complex | Array, dt: Array | float, rhs: Array,
        *, dtype: jnp.dtype, transpose: bool,
    ) -> Array:
        """Unified structured shifted solve."""
        dt, was_scalar = _normalize_dt_api(dt, dtype=jnp.float64)

        def solve(zeta_vec: Array, dt_vec: Array, rhs_flat: Array) -> Array:
            return _jordan_solve_batched(
                self.real_rates.astype(rhs_flat.dtype),
                self.real_sizes,
                self.osc_decays.astype(rhs_flat.real.dtype),
                self.osc_freqs.astype(rhs_flat.real.dtype),
                self.osc_sizes,
                zeta_vec, dt_vec.astype(rhs_flat.real.dtype), rhs_flat,
                transpose=transpose,
            )

        return _apply_dt_solve(
            zeta, dt, was_scalar, rhs, dtype=dtype,
            scalar_solve=solve, batched_solve=solve,
        )

    def _materialize_operator(
        self, dt: Array | float, *, dtype: jnp.dtype,
        real_coeffs: Callable, osc_coeffs: Callable,
    ) -> tuple[Array, bool]:
        """Build the full (batched) operator matrix for expm or phi1."""
        dt, was_scalar = _normalize_dt_api(dt, dtype=jnp.float64)
        dt_real = dt.astype(jnp.real(jnp.asarray(0, dtype=dtype)).dtype)
        blocks: list[Array] = []

        for i, size in enumerate(self.real_sizes):
            coeff = real_coeffs(self.real_rates[i : i + 1].astype(dtype), size, dt_real, dtype)
            blocks.append(_real_chain_matrix_from_coeff(coeff, size))

        for i, size in enumerate(self.osc_sizes):
            coeff = osc_coeffs(
                self.osc_decays[i : i + 1].astype(dt_real.dtype),
                self.osc_freqs[i : i + 1].astype(dt_real.dtype),
                size, dt_real, dtype,
            )
            blocks.append(_osc_chain_matrix_from_coeff(coeff, size))

        return _jordan_block_diag(blocks, dt.shape[0], dtype), was_scalar

    # -- public API --------------------------------------------------------

    def expm(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        out, was_scalar = self._materialize_operator(
            dt, dtype=d, real_coeffs=_real_expm_coeffs, osc_coeffs=_osc_expm_coeffs,
        )
        return _restore_dt_api(out, was_scalar)

    def phi1(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        out, was_scalar = self._materialize_operator(
            dt, dtype=d, real_coeffs=_phi1_chain_coeffs, osc_coeffs=_osc_phi1_coeffs,
        )
        return _restore_dt_api(out, was_scalar)

    def phi2(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        out, was_scalar = self._materialize_operator(
            dt, dtype=d, real_coeffs=_phi2_chain_coeffs, osc_coeffs=_osc_phi2_coeffs,
        )
        return _restore_dt_api(out, was_scalar)

    def solve_shifted(self, zeta, dt, rhs, *, dtype=None) -> Array:
        return self._solve(zeta, dt, rhs, dtype=_complex_dtype_for(dtype or jnp.float64), transpose=False)

    def solve_shifted_transpose(self, zeta, dt, rhs, *, dtype=None) -> Array:
        return self._solve(zeta, dt, rhs, dtype=_complex_dtype_for(dtype or jnp.float64), transpose=True)

    def lambda_multiply_left(self, rhs, *, dtype=None) -> Array:
        rhs = jnp.asarray(rhs, dtype=dtype or jnp.float64)
        return _jordan_lambda_left_unbatched(
            self.real_rates.astype(rhs.dtype), self.real_sizes,
            self.osc_decays.astype(rhs.dtype), self.osc_freqs.astype(rhs.dtype),
            self.osc_sizes, rhs,
        )

    def lambda_multiply_right(self, lhs, *, dtype=None) -> Array:
        lhs = jnp.asarray(lhs, dtype=dtype or jnp.float64)
        return jnp.swapaxes(
            _jordan_lambda_left_unbatched(
                self.real_rates.astype(lhs.dtype), self.real_sizes,
                self.osc_decays.astype(lhs.dtype), self.osc_freqs.astype(lhs.dtype),
                self.osc_sizes, jnp.swapaxes(lhs, -1, -2),
            ),
            -1, -2,
        )

    # -- precomputation helpers -------------------------------------------

    def _with_precomputed(self, **kwargs) -> "JordanLambda":
        """Return a copy with selected precomputed block tuples replaced."""
        obj = object.__new__(JordanLambda)
        object.__setattr__(obj, "real_rates", self.real_rates)
        object.__setattr__(obj, "real_sizes", self.real_sizes)
        object.__setattr__(obj, "osc_decays", self.osc_decays)
        object.__setattr__(obj, "osc_freqs", self.osc_freqs)
        object.__setattr__(obj, "osc_sizes", self.osc_sizes)
        object.__setattr__(obj, "_expm_blocks", kwargs.get("_expm_blocks", self._expm_blocks))
        object.__setattr__(obj, "_phi1_blocks", kwargs.get("_phi1_blocks", self._phi1_blocks))
        object.__setattr__(obj, "_phi2_blocks", kwargs.get("_phi2_blocks", self._phi2_blocks))
        return obj

    def _precompute_blocks(self, dt, *, dtype, real_coeffs, osc_coeffs) -> tuple:
        """Build per-block matrices (2D for scalar dt, 3D for array dt)."""
        dt_arr, was_scalar = _normalize_dt_api(dt, dtype=jnp.float64)
        dt_real = dt_arr.astype(jnp.real(jnp.asarray(0, dtype=dtype)).dtype)
        blocks = []
        for i, size in enumerate(self.real_sizes):
            coeff = real_coeffs(self.real_rates[i : i + 1].astype(dtype), size, dt_real, dtype)
            mat = _real_chain_matrix_from_coeff(coeff, size)  # (n_steps, size, size)
            blocks.append(mat[0] if was_scalar else mat)
        for i, size in enumerate(self.osc_sizes):
            coeff = osc_coeffs(
                self.osc_decays[i : i + 1].astype(dt_real.dtype),
                self.osc_freqs[i : i + 1].astype(dt_real.dtype),
                size, dt_real, dtype,
            )
            mat = _osc_chain_matrix_from_coeff(coeff, size)  # (n_steps, 2*size, 2*size)
            blocks.append(mat[0] if was_scalar else mat)
        return tuple(blocks)

    def precompute_expm(self, dt, *, dtype=None) -> "JordanLambda":
        d = jnp.dtype(dtype or jnp.float64)
        return self._with_precomputed(
            _expm_blocks=self._precompute_blocks(dt, dtype=d,
                real_coeffs=_real_expm_coeffs, osc_coeffs=_osc_expm_coeffs))

    def precompute_phi1(self, dt, *, dtype=None) -> "JordanLambda":
        d = jnp.dtype(dtype or jnp.float64)
        return self._with_precomputed(
            _phi1_blocks=self._precompute_blocks(dt, dtype=d,
                real_coeffs=_phi1_chain_coeffs, osc_coeffs=_osc_phi1_coeffs))

    def precompute_phi2(self, dt, *, dtype=None) -> "JordanLambda":
        d = jnp.dtype(dtype or jnp.float64)
        return self._with_precomputed(
            _phi2_blocks=self._precompute_blocks(dt, dtype=d,
                real_coeffs=_phi2_chain_coeffs, osc_coeffs=_osc_phi2_coeffs))

    def __getitem__(self, ix) -> "JordanLambda":
        def _s(blocks):
            if blocks is None:
                return None
            return tuple(b if b.ndim == 2 else b[ix] for b in blocks)
        return self._with_precomputed(
            _expm_blocks=_s(self._expm_blocks),
            _phi1_blocks=_s(self._phi1_blocks),
            _phi2_blocks=_s(self._phi2_blocks),
        )

    def _block_entries(self) -> list[tuple[int, int, int]]:
        """Return (block_idx, start, width) for every block in R order."""
        entries = []
        start = 0
        for i, size in enumerate(self.real_sizes):
            entries.append((i, start, size))
            start += size
        n_real = len(self.real_sizes)
        for j, size in enumerate(self.osc_sizes):
            width = 2 * size
            entries.append((n_real + j, start, width))
            start += width
        return entries

    def _apply_blocks_left(self, blocks, rhs, *, dtype) -> Array:
        """Apply block-diagonal operator from the left: ``block_i @ rhs_i``.

        Same-width blocks are batched into a single einsum to reduce XLA ops.
        """
        entries = self._block_entries()
        if not entries:
            return jnp.zeros_like(rhs)
        by_width: dict[int, list[tuple[int, int]]] = {}
        for orig_idx, s, w in entries:
            by_width.setdefault(w, []).append((orig_idx, s))
        result_map: dict[int, Array] = {}
        for w, grp in by_width.items():
            orig_idxs = [g[0] for g in grp]
            grp_starts = [g[1] for g in grp]
            block_stack = jnp.stack([blocks[oi].astype(dtype) for oi in orig_idxs])   # (n, w, w)
            rhs_stack = jnp.stack([rhs[..., s:s+w, :] for s in grp_starts], axis=-3)  # (..., n, w, cols)
            res = jnp.einsum("nab,...nbc->...nac", block_stack, rhs_stack)             # (..., n, w, cols)
            for i, oi in enumerate(orig_idxs):
                result_map[oi] = res[..., i, :, :]
        return jnp.concatenate([result_map[i] for i in range(len(entries))], axis=-2)

    def _apply_blocks_right(self, blocks, lhs, *, dtype) -> Array:
        """Apply block-diagonal operator from the right: ``lhs_i @ block_i^T``.

        Same-width blocks are batched into a single einsum to reduce XLA ops.
        """
        entries = self._block_entries()
        if not entries:
            return jnp.zeros_like(lhs)
        by_width: dict[int, list[tuple[int, int]]] = {}
        for orig_idx, s, w in entries:
            by_width.setdefault(w, []).append((orig_idx, s))
        result_map: dict[int, Array] = {}
        for w, grp in by_width.items():
            orig_idxs = [g[0] for g in grp]
            grp_starts = [g[1] for g in grp]
            block_stack = jnp.stack([blocks[oi].astype(dtype) for oi in orig_idxs])   # (n, w, w)
            lhs_stack = jnp.stack([lhs[..., s:s+w] for s in grp_starts], axis=-2)     # (..., k, n, w)
            res = jnp.einsum("nab,...knb->...kna", block_stack, lhs_stack)             # (..., k, n, w)
            for i, oi in enumerate(orig_idxs):
                result_map[oi] = res[..., i, :]
        return jnp.concatenate([result_map[i] for i in range(len(entries))], axis=-1)

    def expm_multiply_left(self, dt, rhs, *, dtype=None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        if self._expm_blocks is not None:
            return self._apply_blocks_left(self._expm_blocks, jnp.asarray(rhs, dtype=d), dtype=d)
        return self._action(dt, rhs, dtype=d, side="left",
                            real_coeffs=_real_expm_coeffs, osc_coeffs=_osc_expm_coeffs)

    def expm_multiply_right(self, dt, lhs, *, dtype=None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        if self._expm_blocks is not None:
            return self._apply_blocks_right(self._expm_blocks, jnp.asarray(lhs, dtype=d), dtype=d)
        return self._action(dt, lhs, dtype=d, side="right",
                            real_coeffs=_real_expm_coeffs, osc_coeffs=_osc_expm_coeffs)

    def phi1_multiply_left(self, dt, rhs, *, dtype=None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        if self._phi1_blocks is not None:
            return self._apply_blocks_left(self._phi1_blocks, jnp.asarray(rhs, dtype=d), dtype=d)
        return self._action(dt, rhs, dtype=d, side="left",
                            real_coeffs=_phi1_chain_coeffs, osc_coeffs=_osc_phi1_coeffs)

    def phi1_multiply_right(self, dt, lhs, *, dtype=None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        if self._phi1_blocks is not None:
            return self._apply_blocks_right(self._phi1_blocks, jnp.asarray(lhs, dtype=d), dtype=d)
        return self._action(dt, lhs, dtype=d, side="right",
                            real_coeffs=_phi1_chain_coeffs, osc_coeffs=_osc_phi1_coeffs)

    def phi2_multiply_left(self, dt, rhs, *, dtype=None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        if self._phi2_blocks is not None:
            return self._apply_blocks_left(self._phi2_blocks, jnp.asarray(rhs, dtype=d), dtype=d)
        return self._action(dt, rhs, dtype=d, side="left",
                            real_coeffs=_phi2_chain_coeffs, osc_coeffs=_osc_phi2_coeffs)

    def phi2_multiply_right(self, dt, lhs, *, dtype=None) -> Array:
        d = jnp.dtype(dtype or jnp.float64)
        if self._phi2_blocks is not None:
            return self._apply_blocks_right(self._phi2_blocks, jnp.asarray(lhs, dtype=d), dtype=d)
        return self._action(dt, lhs, dtype=d, side="right",
                            real_coeffs=_phi2_chain_coeffs, osc_coeffs=_osc_phi2_coeffs)

    def b_from_prony(
        self,
        *,
        alpha: Optional[Array] = None,
        beta: Optional[Array] = None,
        delta: Optional[Array] = None,
    ) -> Array:
        """Construct the Jordan-basis vectors ``b`` from Prony coefficients."""
        q = None
        pieces: list[Array] = []

        if self.real_sizes:
            if alpha is None:
                raise ValueError("alpha must be provided when real blocks are present.")
            alpha_arr = jnp.asarray(alpha)
            if alpha_arr.ndim != 2 or alpha_arr.shape[1] != sum(self.real_sizes):
                raise ValueError("alpha must have shape (n, sum(real_sizes)).")
            q = int(alpha_arr.shape[0])
            start = 0
            out_parts: list[Array] = []
            for size in self.real_sizes:
                block = alpha_arr[:, start : start + size]
                start += size
                out_parts.append(block - jnp.pad(block[:, 1:], ((0, 0), (0, 1))))
            pieces.append(jnp.concatenate(out_parts, axis=1))

        if self.osc_sizes:
            if beta is None or delta is None:
                raise ValueError("beta and delta must be provided when oscillatory blocks are present.")
            beta_arr = jnp.asarray(beta)
            delta_arr = jnp.asarray(delta, dtype=beta_arr.dtype)
            total_osc = sum(self.osc_sizes)
            if beta_arr.ndim != 2 or beta_arr.shape != delta_arr.shape or beta_arr.shape[1] != total_osc:
                raise ValueError("beta and delta must have shape (n, sum(osc_sizes)).")
            q = int(beta_arr.shape[0]) if q is None else q
            if int(beta_arr.shape[0]) != q:
                raise ValueError("alpha, beta and delta must share the same leading n axis.")
            start = 0
            osc_parts: list[Array] = []
            for size in self.osc_sizes:
                beta_block = beta_arr[:, start : start + size]
                delta_block = delta_arr[:, start : start + size]
                start += size
                dbeta = beta_block - jnp.pad(beta_block[:, 1:], ((0, 0), (0, 1)))
                ddelta = delta_block - jnp.pad(delta_block[:, 1:], ((0, 0), (0, 1)))
                pair = jnp.stack([0.5 * (dbeta - ddelta), 0.5 * (dbeta + ddelta)], axis=-1)
                osc_parts.append(pair.reshape(q, 2 * size))
            pieces.append(jnp.concatenate(osc_parts, axis=1))

        if q is None:
            raise ValueError("At least one block must be present to construct b.")
        return jnp.concatenate(pieces, axis=1) if pieces else jnp.zeros((q, 0))

    def matrix(self, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Materialize the dense matrix represented by this operator."""
        out_dtype = jnp.dtype(dtype or jnp.float64)
        blocks: list[Array] = []

        for i, size in enumerate(self.real_sizes):
            rate = self.real_rates[i]
            block = jnp.asarray(rate, dtype=out_dtype) * jnp.eye(size, dtype=out_dtype)
            if size > 1:
                block = block - jnp.eye(size, k=1, dtype=out_dtype)
            blocks.append(block)

        for i, size in enumerate(self.osc_sizes):
            blocks.append(_osc_lambda_block_matrix(
                self.osc_decays[i], self.osc_freqs[i], size, out_dtype,
            ))

        if not blocks:
            return jnp.zeros((0, 0), dtype=out_dtype)
        return jsp_linalg.block_diag(*blocks)


# ---------------------------------------------------------------------------
# Small pure-Python helpers
# ---------------------------------------------------------------------------


def _as_int_tuple(x) -> tuple[int, ...]:
    if x is None or (isinstance(x, (tuple, list)) and len(x) == 0):
        return ()
    arr = np.asarray(x, dtype=np.int32)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D array, got shape {tuple(arr.shape)}")
    return tuple(int(v) for v in arr.tolist())




def _complex_dtype_for(dtype: jnp.dtype) -> jnp.dtype:
    return jnp.result_type(dtype, jnp.complex64)


__all__ = ["Lambda", "DenseLambda", "JordanLambda"]
