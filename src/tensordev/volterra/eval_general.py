from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem, DenseElemFirstOn
from tensordev.volterra.coeffs import VolterraCoefficients, validate_volterra_coefficients
from tensordev.util.combinatorics import multiindex_batched_navigation


Array = jax.Array

_CORE = Jax()


def eval_e(
    y: Array,
    coef: VolterraCoefficients,
) -> DenseElem:
    r"""Evaluate the packed multi-index local Volterra increment ``E``.

    The returned dense element has levels ``0, ..., trunc`` and degree-zero
    level equal to zero.  This evaluator implements the general ``n >= 1``
    packed multi-index recursion under the coefficient symmetry hypothesis.
    In the outer algorithm it is used for ``n > 1``; ``n == 1`` has a cheaper
    scalar fast path in :mod:`tensordev.volterra.eval_scalar`.
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
    r"""Evaluate the packed multi-index local contribution ``v tensor E``."""
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
    r"""Return ``E`` in first-on format, i.e. levels ``1, ..., trunc``.

    Computes

        sum_p ( sum_ell alpha[p, ell] / ell! *
                y_1^{shuffle ell_1} shuffle ... shuffle y_q^{shuffle ell_q})
              tensor y_p.
    """
    validate_volterra_coefficients(coef)
    if coef.trunc <= 0:
        raise ValueError(f"trunc must be positive, got {coef.trunc}.")

    y = _normalize_y_multiindex(y, coef)
    dtype = jnp.result_type(y, coef.alpha)
    y = y.astype(dtype)
    alpha = coef.alpha.astype(dtype)

    N = coef.trunc
    q = coef.q
    global_idx_by_deg, succ_local_by_n_r = multiindex_batched_navigation(q, N - 1)
    inv_factorial = coef.layout.inv_factorial.astype(dtype)
    back_trans = coef.layout.backward_transition.astype(dtype)  # (layout.size, n)

    # F_stack[n]: DenseElem, level k shape → batch + (num_n, n, m**k)
    F_stack: list[DenseElem | None] = [None] * N

    for n in range(N - 1, -1, -1):
        max_degree = N - 1 - n
        idx_n = global_idx_by_deg[n]  # numpy array of shape (num_n,), static
        num_n = len(idx_n)

        # scale factors: shape (num_n,)
        scale_n = inv_factorial[idx_n]

        # alpha[..., :, idx_n] → batch + (n, num_n)
        # moveaxis → batch + (num_n, n); [..., None] → batch + (num_n, n, 1)
        alpha_gathered = jnp.moveaxis(alpha[..., :, idx_n], -1, -2)
        f_base_n = alpha_gathered[..., None] * scale_n[:, None, None]
        f_stack_n: DenseElem = (f_base_n,)

        if n <= N - 2:
            succ_local_r = succ_local_by_n_r[n]  # tuple of n numpy int arrays

            for r in range(q):
                sl = succ_local_r[r]  # shape (num_n,) — local indices into degree n+1

                # transition scalars: shape (num_n,)
                trans_r = back_trans[idx_n, r]

                # F_stack[n+1][k]: batch + (num_{n+1}, n, m**k)
                # After gather:    batch + (num_n,     n, m**k)
                F_gathered = tuple(
                    lvl[..., sl, :, :] for lvl in F_stack[n + 1]
                )
                F_scaled = tuple(
                    lvl * trans_r[:, None, None] for lvl in F_gathered
                )
                fy = core.tensor_shuffle_vector(
                    F_scaled,
                    y[..., r, None, None, :],  # batch + (1, 1, m) — broadcasts over num_n
                    trunc=max_degree,
                )
                f_stack_n = core.tensor_summation(
                    f_stack_n,
                    (jnp.zeros_like(f_base_n),) + fy,
                    trunc=max_degree,
                )

        F_stack[n] = f_stack_n

    # F_stack[0][k]: batch + (1, n, m**k) → batch + (n, m**k)
    root: DenseElem = tuple(lvl[..., 0, :, :] for lvl in F_stack[0])

    by_final_letter = core.tensor_product(
        root,
        (y,),
        trunc=N,
        b_first_on=True,
    )
    return tuple(jnp.sum(level, axis=-2) for level in by_final_letter)


def _normalize_y_multiindex(y: Array, coef: VolterraCoefficients) -> Array:
    """Return projected increment with trailing shape ``(n, m)``."""
    y = jnp.asarray(y, dtype=coef.alpha.dtype)
    if coef.q == 1 and y.shape[-1:] == (coef.m,) and (y.ndim == 1 or y.shape[-2] != 1):
        return y[..., None, :]
    if y.ndim < 2 or y.shape[-2:] != (coef.q, coef.m):
        raise ValueError(f"y must have trailing shape ({coef.q}, {coef.m}), got {tuple(y.shape)}.")
    return y


__all__ = ["eval_e", "eval_vte"]
