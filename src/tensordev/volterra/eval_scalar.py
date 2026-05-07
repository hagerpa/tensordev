from __future__ import annotations

import jax
import jax.numpy as jnp

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem
from tensordev.volterra.coeffs import VolterraCoefficients, validate_volterra_coefficients


Array = jax.Array

_CORE = Jax()


def eval_vte(
    v: DenseElem,
    y: Array,
    coef: VolterraCoefficients,
) -> DenseElem:
    r"""Evaluate ``v ⊗_N E`` via the Horner scheme (Algorithm EvalVtE, q=1).

    Computes v' = v ⊗_N E directly without forming E first, following
    the Horner-type algorithm from the paper (scalar case q=1).
    The outer loop over output levels n is data-independent and can be
    parallelised; the inner recursion over k is sequential.

    The returned dense element has levels ``0, ..., trunc`` with a zero
    degree-zero level.  It is the local summand used by the outer quadratic
    Volterra-Chen recursion.
    """
    if len(v) == 0:
        raise ValueError("v must contain at least the degree-zero level.")
    if len(v) > coef.trunc + 1:
        v = tuple(v[: coef.trunc + 1])

    validate_volterra_coefficients(coef)
    if coef.trunc <= 0:
        raise ValueError(f"trunc must be positive, got {coef.trunc}.")
    if coef.q != 1:
        raise ValueError(f"Horner scheme requires q == 1, got q={coef.q}.")
    if coef.alpha.shape[-2:] != (1, coef.trunc):
        raise ValueError(
            "For q == 1, alpha.shape[-2:] must be (1, trunc); "
            f"got {coef.alpha.shape[-2:]} and trunc={coef.trunc}."
        )

    y = _normalize_y_q1(y, coef)
    dtype = jnp.result_type(y, coef.alpha)
    y = y.astype(dtype)
    alpha = coef.alpha.astype(dtype)

    batch_shape = jnp.broadcast_shapes(coef.leading_shape, y.shape[:-1])

    # Pad v with zero levels for any missing entries (levels >= len(v) contribute 0).
    v_levels = [lv.astype(dtype) for lv in v]
    for k in range(len(v_levels), coef.trunc):
        v_levels.append(jnp.zeros(batch_shape + (coef.m ** k,), dtype=dtype))

    # Horner scheme: build v'^{(n)} = W ⊗ y, where W is accumulated as:
    #   W  = v^{(0)} * β_n
    #   W  = (W ⊗ y) + v^{(k)} * β_{n-k}   for k = 1, ..., n-1
    # with β_n = alpha[..., 0, n-1].
    zero = jnp.zeros(batch_shape + (1,), dtype=dtype)
    out = [zero]  # level 0 of v' is always 0 (E has no level-0 term)
    for n in range(1, coef.trunc + 1):
        W = v_levels[0] * alpha[..., 0, n - 1][..., None]
        for k in range(1, n):
            W = _CORE.tensor_product_homogeneous(W, y) + v_levels[k] * alpha[..., 0, n - k - 1][..., None]
        out.append(_CORE.tensor_product_homogeneous(W, y))

    return tuple(out)


def _normalize_y_q1(y: Array, coef: VolterraCoefficients) -> Array:
    """Return scalar projected increment with trailing shape ``(m,)``."""
    y = jnp.asarray(y, dtype=coef.alpha.dtype)
    if y.shape[-1] == coef.m:
        if y.ndim >= 2 and y.shape[-2:] == (1, coef.m):
            return y[..., 0, :]
        return y
    raise ValueError(f"y must have trailing shape ({coef.m},) or (1, {coef.m}), got {tuple(y.shape)}.")


__all__ = ["eval_vte"]
