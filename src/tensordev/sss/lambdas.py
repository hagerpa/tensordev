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
- Internal batched kernels use ``dt.shape == (m,)``.
- For paired batched actions, the operand leading axis must match ``dt.shape[0]``.
"""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Callable, Literal, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from tensordev.core.utils.xp import xp

Array = jax.Array

# ---------------------------------------------------------------------------
# Scalar / batched dt normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_dt_api(dt: Array | float, *, dtype: jnp.dtype) -> tuple[Array, bool]:
    """Return ``dt`` as shape ``(m,)`` in the real dtype of ``dtype``, and whether it was scalar."""
    dt_arr = jnp.asarray(dt, dtype=jnp.finfo(dtype).dtype)
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
    solve: Callable[[Array, Array, Array], Array],
) -> Array:
    """Apply a shifted solve with shared scalar/batched shape handling."""
    if was_scalar_dt:
        rhs_flat, prefix = _flatten_scalar_dt_operand(rhs, dtype=dtype, side="left", name="rhs")
        zeta_flat = _normalize_zeta_api(zeta, batch=rhs_flat.shape[0], dtype=dtype)
        dt_flat = jnp.full((rhs_flat.shape[0],), dt[0], dtype=dt.dtype)
        out = solve(zeta_flat, dt_flat, rhs_flat)
        return _restore_scalar_dt_operand(out, prefix, side="left")

    rhs_flat, rhs_was_unbatched, extra, cols = _flatten_paired_operand(
        dt, rhs, dtype=dtype, side="left", name="rhs",
    )
    zeta_vec = _normalize_zeta_api(zeta, batch=dt.shape[0], dtype=dtype)
    out = solve(zeta_vec, dt, rhs_flat)
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
    return xp.einsum("ab,...bc->...ac", m, x)


def _on_side(apply_left: Callable[[Array], Array], x: Array, side: _SIDE) -> Array:
    r"""Apply :math:`x\mapsto Ax` from ``side``: :math:`Ax` (``"left"``) or :math:`xA^T` (``"right"``)."""
    if side == "left":
        return apply_left(x)
    return jnp.swapaxes(apply_left(jnp.swapaxes(x, -1, -2)), -1, -2)


# ---------------------------------------------------------------------------
# exp, phi_1 and phi_2 of dense matrices
# ---------------------------------------------------------------------------

# Lengths of the squaring loop: the shortest that covers every matrix of a batch runs, with no
# branch inside, so a GPU decides once per evaluation.  Static lengths keep the loop differentiable
# in reverse mode; inputs needing more squarings than the longest give NaN.
_SQUARING_LOOPS = (0, 4, 16, 64)


# [6/6] Padé coefficients for φ_k(z) = Σ z^j/(j+k)!, computed at import time.
def _compute_phi_pade_coeffs(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, d) [6/6] Padé coefficients for φ_order whose Taylor series is Σ z^k/(k+order)!"""
    from math import factorial as _fac
    p = 6
    c = np.array([1.0 / _fac(k + order) for k in range(2 * p + 1)], dtype=np.float64)
    _A = np.zeros((p, p))
    _rhs = np.zeros(p)
    for _i in range(p):
        _k = p + 1 + _i
        _rhs[_i] = -c[_k]
        for _j in range(1, p + 1):
            _A[_i, _j - 1] = c[_k - _j]
    _d = np.linalg.solve(_A, _rhs)
    d = np.concatenate([[1.0], _d])
    n = np.zeros(p + 1)
    for _k in range(p + 1):
        for _j in range(min(_k, p) + 1):
            n[_k] += c[_k - _j] * d[_j]
    return n, d

_PHI_PADE = (_compute_phi_pade_coeffs(1), _compute_phi_pade_coeffs(2))
_PADE_THETA = 1.0  # shared 1-norm scaling threshold for [6/6] Padé (full float64 accuracy)


def _pade_eval(B: Array, B2: Array, B4: Array, B6: Array,
               n: Array, d: Array, eye: Array) -> Array:
    """Return :math:`D(B)^{-1} N(B)` for the [6/6] Padé coefficients ``n`` and ``d``."""
    def poly(c):
        return c[0] * eye + c[2] * B2 + c[4] * B4 + c[6] * B6 + xp.matmul(B, c[1] * eye + c[3] * B2 + c[5] * B4)

    return jnp.linalg.solve(poly(d), poly(n))


@partial(jax.jit, static_argnames=("order",))
def _phi(A: Array, *, order: int) -> Array:
    r"""Return :math:`\varphi_k(A)` for a batch of matrices, :math:`k` = ``order`` :math:`\in\{0,1,2\}`.

    With :math:`\varphi_0=\exp`, :math:`s=\max(0,\lceil\log_2\|A\|_1\rceil)` and
    :math:`C=2^{-s}A`, the [6/6] Padé approximants of :math:`\varphi_1(C)` and
    :math:`\varphi_2(C)` and :math:`e^C=I+C\varphi_1(C)` are doubled :math:`s` times by

    .. math::
        e^{2C}=\bigl(e^C\bigr)^2,\qquad
        \varphi_1(2C)=\tfrac12\varphi_1(C)\bigl(e^C+I\bigr),\qquad
        \varphi_2(2C)=\tfrac12\varphi_2(C)+\tfrac14\varphi_1(C)^2.

    Matrices with nonfinite :math:`\|A\|_1` or more squarings than the longest loop give NaN.
    """
    dtype = A.dtype
    eye = jnp.eye(A.shape[-1], dtype=dtype)
    norm1 = jnp.linalg.norm(A, ord=1, axis=(-2, -1))
    scaling = jnp.maximum(0, jnp.ceil(jnp.log2(jnp.maximum(norm1, 1e-30) / _PADE_THETA)))
    valid = jnp.isfinite(norm1) & (scaling <= _SQUARING_LOOPS[-1])
    s = jnp.where(valid, scaling, 0).astype(jnp.int32)
    B = jnp.where(valid[..., None, None], A, 0) * jnp.exp2(-s.astype(norm1.dtype))[..., None, None]
    B2 = xp.matmul(B, B)
    B4 = xp.matmul(B2, B2)
    B6 = xp.matmul(B4, B2)
    approximants = tuple(
        _pade_eval(B, B2, B4, B6, jnp.asarray(n, dtype=dtype), jnp.asarray(d, dtype=dtype), eye)
        for n, d in _PHI_PADE[: max(order, 1)]
    )

    def double(values):
        exp, *phi = values
        doubled = [xp.matmul(exp, exp)]
        if order >= 1:
            doubled.append(0.5 * xp.matmul(phi[0], exp + eye))
        if order == 2:
            doubled.append(0.5 * phi[1] + 0.25 * xp.matmul(phi[0], phi[0]))
        return tuple(doubled)

    def step(values, i):
        mask = (i < s)[..., None, None]
        # Finished matrices enter as zeros, so their overflow cannot reach a gradient.
        doubled = double(tuple(jnp.where(mask, v, 0) for v in values))
        return tuple(jnp.where(mask, new, old) for new, old in zip(doubled, values)), None

    def square(length, values):
        return jax.lax.scan(step, values, jnp.arange(length))[0]

    values = ((eye + xp.matmul(B, approximants[0]),) + approximants)[: order + 1]
    loop = jnp.sum(jnp.max(s, initial=0) > jnp.asarray(_SQUARING_LOOPS[:-1]))
    values = jax.lax.switch(loop, [partial(square, length) for length in _SQUARING_LOOPS], values)
    return jnp.where(valid[..., None, None], values[order], jnp.nan)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

_STORED = ("_expm", "_phi1", "_phi2")
_PHI_ORDER = {"expm": 0, "phi1": 1, "phi2": 2}


@lru_cache(maxsize=None)
def _pytree_fields(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Names of the array fields and of the static fields of a ``Lambda`` dataclass."""
    fields = dataclasses.fields(cls)
    return (
        tuple(f.name for f in fields if not f.metadata.get("static")),
        tuple(f.name for f in fields if f.metadata.get("static")),
    )


class Lambda(ABC):
    r"""State operator :math:`\Lambda` of a finite-state-space kernel.

    For ``dt`` of shape ``()`` or ``(m,)`` the methods evaluate :math:`f(-dt\,\Lambda)` for

    .. math::
        f=\exp\quad\text{or}\quad
        f=\varphi_k,\ \ \varphi_k(z)=\sum_{j=0}^{\infty}\frac{z^j}{(j+k)!},\ k=1,2,

    apply it to operands, and solve :math:`(\zeta I+dt\,\Lambda)x=b`. ``precompute_*`` returns a
    copy that stores :math:`f(-dt\,\Lambda)` per step; its ``*_multiply_*`` methods apply the
    stored values and ignore ``dt``. The stored values are constants: to differentiate with
    respect to :math:`\Lambda`, call ``precompute_*`` inside the differentiated function.

    A subclass is a frozen dataclass with the fields ``_expm``, ``_phi1`` and ``_phi2`` and
    defines ``state_dim``, ``lambda_multiply_left``, ``_dtype``, ``_operator``,
    ``_action_paired`` and ``_solve_paired``.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension ``R`` of the state space."""

    @property
    @abstractmethod
    def _dtype(self) -> jnp.dtype:
        """Default working dtype."""

    @abstractmethod
    def lambda_multiply_left(self, rhs: Array, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``Lambda @ rhs`` acting on the second-to-last axis of ``rhs``."""

    @abstractmethod
    def _operator(self, dt: Array, *, dtype: jnp.dtype, kind: str) -> Array:
        r""":math:`f(-dt_i\Lambda)` for every ``dt_i``, shape ``(m, R, R)``."""

    @abstractmethod
    def _action_paired(self, dt: Array, x: Array, *, kind: str) -> Array:
        r""":math:`f(-dt_i\Lambda)\,x_i` for ``x`` of shape ``(m, R, k)``."""

    @abstractmethod
    def _solve_paired(self, zeta: Array, dt: Array, rhs: Array, *, transpose: bool) -> Array:
        r"""Solve :math:`(\zeta_i I+dt_i\Lambda)x_i=b_i` or its transpose; ``rhs`` is ``(m, R, k)``."""

    def _action_shared(self, dt: Array, x: Array, *, kind: str) -> Array:
        r""":math:`f(-dt\,\Lambda)\,x_i` for one ``dt`` and ``x`` of shape ``(m, R, k)``."""
        return self._action_paired(jnp.full((x.shape[0],), dt, dtype=x.real.dtype), x, kind=kind)

    def _stored(self, dt: Array, *, dtype: jnp.dtype, kind: str) -> Any:
        r"""Values that ``precompute_*`` stores: :math:`f(-dt_i\Lambda)`, leading axis ``m``."""
        return self._operator(dt, dtype=dtype, kind=kind)

    def _apply_stored(self, values: Any, x: Array) -> Array:
        """Apply the stored values of one step to ``x`` from the left."""
        return _left_multiply(values.astype(x.dtype), x)

    def _default_dtype(self, dtype: Optional[jnp.dtype]) -> jnp.dtype:
        return jnp.dtype(self._dtype if dtype is None else dtype)

    def _replace(self, **fields) -> "Lambda":
        obj = copy.copy(self)
        for name, value in fields.items():
            object.__setattr__(obj, name, value)
        return obj

    def tree_flatten(self):
        leaves, static = _pytree_fields(type(self))
        return tuple(getattr(self, n) for n in leaves), tuple(getattr(self, n) for n in static)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        leaves, static = _pytree_fields(cls)
        for name, value in zip(leaves + static, tuple(children) + tuple(aux)):
            object.__setattr__(obj, name, value)
        return obj

    def _function(self, kind: str, dt: Array | float, dtype: Optional[jnp.dtype]) -> Array:
        d = self._default_dtype(dtype)
        dt, was_scalar = _normalize_dt_api(dt, dtype=d)
        return _restore_dt_api(self._operator(dt, dtype=d, kind=kind), was_scalar)

    def _multiply(
        self, kind: str, dt: Array | float, x: Array, dtype: Optional[jnp.dtype], side: _SIDE,
    ) -> Array:
        d = self._default_dtype(dtype)
        stored = getattr(self, "_" + kind)
        if stored is not None:
            return _on_side(partial(self._apply_stored, stored), jnp.asarray(x, dtype=d), side)
        dt, was_scalar = _normalize_dt_api(dt, dtype=d)
        return _apply_dt_action(
            dt, was_scalar, x, dtype=d, side=side, name="lhs" if side == "right" else "rhs",
            scalar_action=partial(self._action_shared, dt[0], kind=kind),
            batched_action=partial(self._action_paired, kind=kind),
        )

    def _solve(
        self, zeta: complex | Array, dt: Array | float, rhs: Array,
        dtype: Optional[jnp.dtype], *, transpose: bool,
    ) -> Array:
        d = _complex_dtype_for(self._default_dtype(dtype))
        dt, was_scalar = _normalize_dt_api(dt, dtype=d)
        return _apply_dt_solve(
            zeta, dt, was_scalar, rhs, dtype=d, solve=partial(self._solve_paired, transpose=transpose)
        )

    def _precompute(self, kind: str, dt: Array | float, dtype: Optional[jnp.dtype]) -> "Lambda":
        d = self._default_dtype(dtype)
        dt, was_scalar = _normalize_dt_api(dt, dtype=d)
        return self._replace(**{"_" + kind: jax.tree.map(
            lambda v: _restore_dt_api(v, was_scalar), self._stored(dt, dtype=d, kind=kind)
        )})

    def expm(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``exp(-dt * Lambda)``."""
        return self._function("expm", dt, dtype)

    def phi1(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``phi_1(-dt * Lambda)``."""
        return self._function("phi1", dt, dtype)

    def phi2(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``phi_2(-dt * Lambda)``."""
        return self._function("phi2", dt, dtype)

    def solve_shifted(
        self, zeta: complex | Array, dt: Array | float, rhs: Array,
        *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Solve ``(zeta I + dt Lambda) x = rhs``."""
        return self._solve(zeta, dt, rhs, dtype, transpose=False)

    def solve_shifted_transpose(
        self, zeta: complex | Array, dt: Array | float, rhs: Array,
        *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Solve ``(zeta I + dt Lambda)^T x = rhs``."""
        return self._solve(zeta, dt, rhs, dtype, transpose=True)

    def lambda_multiply_right(self, lhs: Array, *, dtype: Optional[jnp.dtype] = None) -> Array:
        """Return ``lhs @ Lambda^T`` acting on the last axis of ``lhs``."""
        return _on_side(partial(self.lambda_multiply_left, dtype=dtype), lhs, "right")

    def expm_multiply_left(
        self, dt: Array | float, rhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``exp(-dt * Lambda) @ rhs``."""
        return self._multiply("expm", dt, rhs, dtype, "left")

    def expm_multiply_right(
        self, dt: Array | float, lhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``lhs @ exp(-dt * Lambda^T)``."""
        return self._multiply("expm", dt, lhs, dtype, "right")

    def phi1_multiply_left(
        self, dt: Array | float, rhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``phi_1(-dt * Lambda) @ rhs``."""
        return self._multiply("phi1", dt, rhs, dtype, "left")

    def phi1_multiply_right(
        self, dt: Array | float, lhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``lhs @ phi_1(-dt * Lambda^T)``."""
        return self._multiply("phi1", dt, lhs, dtype, "right")

    def phi2_multiply_left(
        self, dt: Array | float, rhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``phi_2(-dt * Lambda) @ rhs``."""
        return self._multiply("phi2", dt, rhs, dtype, "left")

    def phi2_multiply_right(
        self, dt: Array | float, lhs: Array, *, dtype: Optional[jnp.dtype] = None,
    ) -> Array:
        """Return ``lhs @ phi_2(-dt * Lambda^T)``."""
        return self._multiply("phi2", dt, lhs, dtype, "right")

    def precompute_expm(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> "Lambda":
        """Return a copy that stores ``exp(-dt * Lambda)`` per step."""
        return self._precompute("expm", dt, dtype)

    def precompute_phi1(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> "Lambda":
        """Return a copy that stores ``phi_1(-dt * Lambda)`` per step."""
        return self._precompute("phi1", dt, dtype)

    def precompute_phi2(self, dt: Array | float, *, dtype: Optional[jnp.dtype] = None) -> "Lambda":
        """Return a copy that stores ``phi_2(-dt * Lambda)`` per step."""
        return self._precompute("phi2", dt, dtype)

    def __getitem__(self, ix) -> "Lambda":
        """Return a copy whose stored values are those of step ``ix``."""
        return self._replace(**{
            name: jax.tree.map(lambda v: v if v.ndim == 2 else v[ix], getattr(self, name))
            for name in _STORED
        })


# ---------------------------------------------------------------------------
# DenseLambda
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class DenseLambda(Lambda):
    r"""State operator :math:`\Lambda` given by ``matrix`` of shape ``(R, R)``.

    :math:`e^{-t\Lambda}` is :func:`jax.scipy.linalg.expm`, and :math:`\varphi_k(-t\Lambda)`
    is a [6/6] Padé approximant with scaling and doubling.
    """

    matrix: Array
    _expm: Optional[Array] = field(init=False, repr=False, default=None)
    _phi1: Optional[Array] = field(init=False, repr=False, default=None)
    _phi2: Optional[Array] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        matrix = jnp.asarray(self.matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "DenseLambda.matrix must have shape (R, R); "
                f"got {tuple(matrix.shape)}."
            )
        matrix = matrix.astype(jnp.result_type(matrix.dtype, jnp.float32))
        object.__setattr__(self, "matrix", matrix)

    @property
    def state_dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def _dtype(self) -> jnp.dtype:
        return self.matrix.dtype

    def _mat(self, dtype: jnp.dtype, *, transpose: bool = False) -> Array:
        m = self.matrix.astype(dtype)
        return m.T if transpose else m

    def _operator(self, dt: Array, *, dtype: jnp.dtype, kind: str) -> Array:
        return _phi(-dt.astype(dtype)[:, None, None] * self._mat(dtype), order=_PHI_ORDER[kind])

    def _action_shared(self, dt: Array, x: Array, *, kind: str) -> Array:
        return _left_multiply(self._operator(dt[None], dtype=x.dtype, kind=kind)[0], x)

    def _action_paired(self, dt: Array, x: Array, *, kind: str) -> Array:
        return xp.matmul(self._operator(dt, dtype=x.dtype, kind=kind), x)

    def _solve_paired(self, zeta: Array, dt: Array, rhs: Array, *, transpose: bool) -> Array:
        return jnp.linalg.solve(
            zeta[:, None, None] * jnp.eye(self.state_dim, dtype=rhs.dtype)
            + dt.astype(rhs.dtype)[:, None, None] * self._mat(rhs.dtype, transpose=transpose),
            rhs,
        )

    def lambda_multiply_left(self, rhs, *, dtype=None) -> Array:
        rhs = jnp.asarray(rhs, dtype=self._default_dtype(dtype))
        return _left_multiply(self._mat(rhs.dtype), rhs)


# ---------------------------------------------------------------------------
# Jordan-chain coefficient helpers
# ---------------------------------------------------------------------------


def _real_expm_coeffs(rate: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    r"""Return :math:`e^{-\lambda h} h^k / k!`, :math:`k < n`, for one block; shape ``(m, n)``."""
    dt = dt.astype(dtype)[:, None]
    return jnp.exp(-dt * rate.astype(dtype)) * jnp.cumprod(
        jnp.concatenate((jnp.ones_like(dt), dt / jnp.arange(1, nmax, dtype=dtype)), axis=-1), axis=-1
    )


def _real_jordan_A(rate: Array, nmax: int, dt: Array, rdtype: jnp.dtype) -> Array:
    """Build A = -dt*(rate*I - N) for one block. Returns (m, nmax, nmax)."""
    lam = rate.astype(rdtype) * jnp.eye(nmax, dtype=rdtype) - jnp.eye(nmax, k=1, dtype=rdtype)
    return -dt[:, None, None].astype(rdtype) * lam[None]


def _osc_jordan_A(decay: Array, freq: Array, nmax: int, dt: Array, rdtype: jnp.dtype) -> Array:
    """Build A = -dt*J_osc for one block. Returns (m, 2*nmax, 2*nmax)."""
    lam = _osc_lambda_block_matrix(decay.astype(rdtype), freq.astype(rdtype), nmax, rdtype)
    return -dt[:, None, None].astype(rdtype) * lam[None]


def _phi_chain_coeffs(rate: Array, nmax: int, dt: Array, dtype: jnp.dtype, *, order: int) -> Array:
    r"""First row of :math:`\varphi_k(-h(\lambda I-N))`, :math:`k` = ``order``; shape ``(m, n)``."""
    rdtype = jnp.finfo(dtype).dtype
    A = _real_jordan_A(rate.astype(rdtype), nmax, dt.astype(rdtype), rdtype)
    return _phi(A, order=order).astype(dtype)[:, 0, :]


def _shifted_coeffs(rate: Array, nmax: int, zeta: Array, dt: Array, dtype: jnp.dtype) -> Array:
    r"""Return :math:`h^k/(\zeta+h\lambda)^{k+1}`, :math:`k<n`, for one block; shape ``(m, n)``."""
    dt = dt.astype(dtype)
    a = zeta + dt * rate.astype(dtype)
    return jnp.cumprod(jnp.concatenate(
        ((1.0 / a)[:, None], jnp.broadcast_to((dt / a)[:, None], (a.shape[0], nmax - 1))), axis=-1
    ), axis=-1)


# ---------------------------------------------------------------------------
# Complex ↔ real-2×2 helpers
# ---------------------------------------------------------------------------


@jax.jit
def _complex_to_real2x2(z: Array) -> Array:
    """Map complex scalars to their real ``2 x 2`` matrix form."""
    re, im = z.real, z.imag
    return jnp.stack(
        [jnp.stack([re, -im], axis=-1), jnp.stack([im, re], axis=-1)],
        axis=-2,
    )


def _osc_expm_coeffs(decay: Array, freq: Array, nmax: int, dt: Array, dtype: jnp.dtype) -> Array:
    r"""Chain coefficients of :math:`e^{-hJ}` for one oscillatory block; shape ``(m, n, 2, 2)``."""
    cdtype = _complex_dtype_for(dtype)
    rate = decay.astype(jnp.finfo(dtype).dtype) + 1j * freq.astype(jnp.finfo(dtype).dtype)
    return _complex_to_real2x2(_real_expm_coeffs(rate, nmax, dt, cdtype))


def _osc_phi_coeffs(
    decay: Array, freq: Array, nmax: int, dt: Array, dtype: jnp.dtype, *, order: int,
) -> Array:
    r"""Chain coefficients of :math:`\varphi_k(-hJ)`, :math:`k` = ``order``; ``(m, n, 2, 2)``."""
    rdtype = jnp.finfo(dtype).dtype
    A = _osc_jordan_A(decay.astype(rdtype), freq.astype(rdtype), nmax, dt.astype(rdtype), rdtype)
    phi = _phi(A, order=order)
    return phi[:, :2, :].reshape(dt.shape[0], 2, nmax, 2).transpose(0, 2, 1, 3).astype(dtype)


# Chain coefficients of f(-h J) for the real and the oscillatory blocks, by ``kind``.
_JORDAN_COEFFS = {
    "expm": (_real_expm_coeffs, _osc_expm_coeffs),
    "phi1": (partial(_phi_chain_coeffs, order=1), partial(_osc_phi_coeffs, order=1)),
    "phi2": (partial(_phi_chain_coeffs, order=2), partial(_osc_phi_coeffs, order=2)),
}


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
    - scalar chains: coeff shape ``(t, n)``, rhs ``(t, n, c)``
    - matrix chains: coeff shape ``(t, n, u, v)``, rhs ``(t, n, v, c)``
    """
    nmax = rhs.shape[1]
    idx, mask = _chain_offset_indices(nmax, lower=lower)
    idx_j, mask_j = jnp.asarray(idx), jnp.asarray(mask)
    if coeff.ndim == 2:
        # Scalar chain
        weights = jnp.take(coeff, idx_j, axis=1) * mask_j[None, :, :]
        return xp.einsum("tij,tjc->tic", weights, rhs)
    else:
        # Matrix (2x2 block) chain
        weights = jnp.take(coeff, idx_j, axis=1) * mask_j[None, :, :, None, None]
        return xp.einsum("tijuv,tjvc->tiuc", weights, rhs)


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
        out = xp.einsum("uv,...nvc->...nuc", base, block)
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
        coeff = real_coeffs(real_rates[i], size, dt_real, rhs.dtype)
        y = _apply_chain(coeff, block, lower=lower)
        parts.append(y)

    for i, size in enumerate(osc_sizes):
        width = 2 * size
        block = rhs[:, start : start + width, :].reshape(batch, size, 2, cols)
        start += width
        coeff = osc_coeffs(osc_decays[i], osc_freqs[i], size, dt_real, rhs.dtype)
        y = _apply_chain(coeff, block, lower=lower)
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
        coeff = _shifted_coeffs(real_rates[i], size, zeta, dt_real, rhs.dtype)
        y = _apply_chain(coeff, block, lower=transpose)
        parts.append(y)

    for i, size in enumerate(osc_sizes):
        width = 2 * size
        block = rhs[:, start : start + width, :]
        start += width
        # For complex zeta, zeta I + h J is not the real 2 x 2 form of a complex
        # chain, so oscillatory blocks are solved densely.
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
    return jnp.take(coeff, idx_j, axis=1) * mask_j[None, :, :]


def _osc_chain_matrix_from_coeff(coeff: Array, size: int) -> Array:
    """Materialize a batched upper-triangular real 2x2 block-Jordan matrix."""
    idx, mask = _chain_offset_indices(size, lower=False)
    idx_j, mask_j = jnp.asarray(idx), jnp.asarray(mask)
    blocks = jnp.take(coeff, idx_j, axis=1) * mask_j[None, :, :, None, None]
    return jnp.transpose(blocks, (0, 1, 3, 2, 4)).reshape(coeff.shape[0], 2 * size, 2 * size)


# ---------------------------------------------------------------------------
# JordanLambda
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class JordanLambda(Lambda):
    r"""Block-diagonal :math:`\Lambda` with real and oscillatory Jordan blocks

    .. math::
        \lambda I_n-N_n,\qquad
        I_n\otimes\begin{pmatrix}\mu&-\omega\\\omega&\mu\end{pmatrix}-N_n\otimes I_2,

    with :math:`(\lambda, n)` from ``real_rates`` and ``real_sizes``, :math:`(\mu, \omega, n)`
    from ``osc_decays``, ``osc_freqs`` and ``osc_sizes``, and :math:`N_n` the :math:`n\times n`
    matrix with ones on its first superdiagonal. The sizes are static; :math:`\omega>0` is checked
    for concrete frequencies.
    """

    real_rates: Array = field(default_factory=lambda: jnp.zeros(0))
    real_sizes: tuple[int, ...] = field(default_factory=tuple, metadata={"static": True})
    osc_decays: Array = field(default_factory=lambda: jnp.zeros(0))
    osc_freqs: Array = field(default_factory=lambda: jnp.zeros(0))
    osc_sizes: tuple[int, ...] = field(default_factory=tuple, metadata={"static": True})
    _expm: Optional[tuple] = field(init=False, repr=False, default=None)
    _phi1: Optional[tuple] = field(init=False, repr=False, default=None)
    _phi2: Optional[tuple] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        parameters = tuple(jnp.asarray(value).ravel() for value in (
            self.real_rates, self.osc_decays, self.osc_freqs
        ))
        dtype = jnp.result_type(jnp.float32, *(x.dtype for x in parameters if x.size))
        real_rates, osc_decays, osc_freqs = (x.astype(dtype) for x in parameters)
        real_sizes = _as_int_tuple(self.real_sizes)
        osc_sizes = _as_int_tuple(self.osc_sizes)

        if real_rates.shape[0] != len(real_sizes):
            raise ValueError("real_rates and real_sizes must have the same length.")
        if osc_decays.shape[0] != osc_freqs.shape[0] or osc_decays.shape[0] != len(osc_sizes):
            raise ValueError("osc_decays, osc_freqs and osc_sizes must have the same length.")
        if any(s <= 0 for s in real_sizes):
            raise ValueError("real_sizes must be strictly positive.")
        if any(s <= 0 for s in osc_sizes):
            raise ValueError("osc_sizes must be strictly positive.")
        if (len(osc_sizes) > 0 and not isinstance(osc_freqs, jax.core.Tracer)
                and bool(jnp.any(osc_freqs <= 0))):
            raise ValueError("osc_freqs must be strictly positive.")

        object.__setattr__(self, "real_rates", real_rates)
        object.__setattr__(self, "real_sizes", real_sizes)
        object.__setattr__(self, "osc_decays", osc_decays)
        object.__setattr__(self, "osc_freqs", osc_freqs)
        object.__setattr__(self, "osc_sizes", osc_sizes)

    @property
    def state_dim(self) -> int:
        return sum(self.real_sizes) + 2 * sum(self.osc_sizes)

    @property
    def _dtype(self) -> jnp.dtype:
        return self.real_rates.dtype

    def _blocks(self, dt: Array, *, dtype: jnp.dtype, kind: str) -> list[Array]:
        r""":math:`f(-dt_i B)` for every diagonal block :math:`B`, each of shape ``(m, w, w)``."""
        real_coeffs, osc_coeffs = _JORDAN_COEFFS[kind]
        return [
            _real_chain_matrix_from_coeff(real_coeffs(rate.astype(dtype), size, dt, dtype), size)
            for rate, size in zip(self.real_rates, self.real_sizes)
        ] + [
            _osc_chain_matrix_from_coeff(
                osc_coeffs(decay.astype(dt.dtype), freq.astype(dt.dtype), size, dt, dtype), size
            )
            for decay, freq, size in zip(self.osc_decays, self.osc_freqs, self.osc_sizes)
        ]

    def _operator(self, dt: Array, *, dtype: jnp.dtype, kind: str) -> Array:
        return _jordan_block_diag(self._blocks(dt, dtype=dtype, kind=kind), dt.shape[0], dtype)

    def _stored(self, dt: Array, *, dtype: jnp.dtype, kind: str) -> tuple:
        return tuple(self._blocks(dt, dtype=dtype, kind=kind))

    def _action_paired(self, dt: Array, x: Array, *, kind: str) -> Array:
        real_coeffs, osc_coeffs = _JORDAN_COEFFS[kind]
        real = x.real.dtype
        return _jordan_apply_batched(
            self.real_rates.astype(x.dtype), self.real_sizes,
            self.osc_decays.astype(real), self.osc_freqs.astype(real), self.osc_sizes,
            dt.astype(real), x, lower=False, real_coeffs=real_coeffs, osc_coeffs=osc_coeffs,
        )

    def _solve_paired(self, zeta: Array, dt: Array, rhs: Array, *, transpose: bool) -> Array:
        real = rhs.real.dtype
        return _jordan_solve_batched(
            self.real_rates.astype(rhs.dtype), self.real_sizes,
            self.osc_decays.astype(real), self.osc_freqs.astype(real), self.osc_sizes,
            zeta, dt.astype(real), rhs, transpose=transpose,
        )

    def lambda_multiply_left(self, rhs, *, dtype=None) -> Array:
        rhs = jnp.asarray(rhs, dtype=self._default_dtype(dtype))
        return _jordan_lambda_left_unbatched(
            self.real_rates.astype(rhs.dtype), self.real_sizes,
            self.osc_decays.astype(rhs.dtype), self.osc_freqs.astype(rhs.dtype),
            self.osc_sizes, rhs,
        )

    def _apply_stored(self, values: tuple, x: Array) -> Array:
        """Apply the stored blocks to ``x`` from the left, with same-width blocks in one einsum."""
        entries = self._block_entries()
        if not entries:
            return jnp.zeros_like(x)
        by_width: dict[int, list[tuple[int, int]]] = {}
        for orig_idx, s, w in entries:
            by_width.setdefault(w, []).append((orig_idx, s))
        result_map: dict[int, Array] = {}
        for w, grp in by_width.items():
            orig_idxs = [g[0] for g in grp]
            block_stack = jnp.stack([values[oi].astype(x.dtype) for oi in orig_idxs])  # (n, w, w)
            x_stack = jnp.stack([x[..., s:s+w, :] for _, s in grp], axis=-3)          # (..., n, w, cols)
            res = xp.einsum("nab,...nbc->...nac", block_stack, x_stack)               # (..., n, w, cols)
            for i, oi in enumerate(orig_idxs):
                result_map[oi] = res[..., i, :, :]
        return jnp.concatenate([result_map[i] for i in range(len(entries))], axis=-2)

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
        out_dtype = self._default_dtype(dtype)
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
