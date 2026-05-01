from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem, DenseElemFirstOn
from tensordev.volterra.coeffs import VolterraCoefficients, validate_volterra_coefficients


Array = jax.Array

_CORE = Jax()


def eval_e(
    y: Array,
    coef: VolterraCoefficients,
) -> DenseElem:
    r"""Evaluate the scalar local Volterra increment ``E``.

    This is the ``q == 1`` fast path.  The returned dense element has levels
    ``0, ..., trunc`` and degree-zero level equal to zero.
    """
    e_first = _eval_e_first_on(y, coef, core=_CORE)
    if not e_first:
        raise ValueError("eval_e requires positive truncation.")
    zero = jnp.zeros(e_first[0].shape[:-1] + (1,), dtype=e_first[0].dtype)
    return (zero,) + e_first


def eval_vte(
    v: DenseElem,
    y: Array,
    coef: VolterraCoefficients,
) -> DenseElem:
    r"""Evaluate the scalar local contribution ``v tensor E``.

    The returned dense element has levels ``0, ..., trunc`` and zero
    degree-zero level.  It is the local summand used by the outer quadratic
    Volterra-Chen recursion.
    """
    if len(v) == 0:
        raise ValueError("v must contain at least the degree-zero level.")
    if len(v) > coef.trunc + 1:
        v = tuple(v[: coef.trunc + 1])

    e_first = _eval_e_first_on(y, coef, core=_CORE)
    out_first = _CORE.tensor_product(
        tuple(v),
        e_first,
        trunc=coef.trunc,
        b_first_on=True,
    )
    if not out_first:
        raise ValueError("eval_vte requires positive truncation.")
    zero = jnp.zeros(out_first[0].shape[:-1] + (1,), dtype=out_first[0].dtype)
    return (zero,) + tuple(out_first)


@partial(jax.jit, static_argnames=("core",))
def _eval_e_first_on(
    y: Array,
    coef: VolterraCoefficients,
    *,
    core: Jax,
) -> DenseElemFirstOn:
    """Return scalar ``E`` in first-on format, i.e. levels ``1, ..., trunc``."""
    validate_volterra_coefficients(coef)
    if coef.trunc <= 0:
        raise ValueError(f"trunc must be positive, got {coef.trunc}.")
    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires q == 1, got q={coef.q}.")

    y = _normalize_y_q1(y, coef)
    dtype = jnp.result_type(y, coef.alpha)
    y = y.astype(dtype)
    alpha = coef.alpha.astype(dtype)

    if coef.alpha.shape[-2:] != (1, coef.trunc):
        raise ValueError(
            "For q == 1, alpha.shape[-2:] must be (1, trunc); "
            f"got {coef.alpha.shape[-2:]} and trunc={coef.trunc}."
        )

    batch_shape = jnp.broadcast_shapes(coef.leading_shape, y.shape[:-1])
    power = jnp.ones(batch_shape + (1,), dtype=dtype)
    out: list[Array] = []
    for n in range(1, coef.trunc + 1):
        power = core.tensor_product_homogeneous(power, y)
        scale = alpha[..., 0, n - 1]
        out.append(scale[..., None] * power)
    return tuple(out)


def _normalize_y_q1(y: Array, coef: VolterraCoefficients) -> Array:
    """Return scalar projected increment with trailing shape ``(m,)``."""
    y = jnp.asarray(y, dtype=coef.alpha.dtype)
    if y.shape[-1] == coef.m:
        if y.ndim >= 2 and y.shape[-2:] == (1, coef.m):
            return y[..., 0, :]
        return y
    raise ValueError(f"y must have trailing shape ({coef.m},) or (1, {coef.m}), got {tuple(y.shape)}.")


__all__ = ["eval_e", "eval_vte"]
