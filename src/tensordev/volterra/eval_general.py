from __future__ import annotations

from functools import lru_cache, partial

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
    r"""Evaluate the packed multi-index local Volterra increment ``E``.

    The returned dense element has levels ``0, ..., trunc`` and degree-zero
    level equal to zero.  This evaluator implements the general ``q >= 1``
    packed multi-index recursion under the coefficient symmetry hypothesis.
    In the outer algorithm it is used for ``q > 1``; ``q == 1`` has a cheaper
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
    by_degree, plus = _packed_multiindex_navigation(q, N - 1)
    inv_factorial = coef.layout.inv_factorial.astype(dtype)

    F: list[DenseElem | None] = [None] * coef.layout.size

    for n in range(N - 1, -1, -1):
        max_degree = N - 1 - n
        for ell_idx in by_degree[n]:
            scale = inv_factorial[ell_idx]
            base: DenseElem = (alpha[..., :, ell_idx][..., None] * scale,)
            f_ell = base

            if n <= N - 2:
                for r in range(q):
                    succ = plus[ell_idx][r]
                    transition = coef.layout.backward_transition[ell_idx, r].astype(dtype)
                    fy = core.tensor_shuffle_vector(
                        F[succ],
                        y[..., r, :][..., None, :],
                        trunc=max_degree,
                    )
                    fy = tuple(transition * level for level in fy)
                    f_ell = core.tensor_summation(
                        f_ell,
                        (jnp.zeros_like(base[0]),) + fy,
                        trunc=max_degree,
                    )

            F[ell_idx] = f_ell

    root = F[0]
    if root is None:
        raise RuntimeError("internal error: root multi-index was not evaluated")

    by_final_letter = core.tensor_product(
        root,
        (y,),
        trunc=N,
        b_first_on=True,
    )
    return tuple(jnp.sum(level, axis=-2) for level in by_final_letter)


def _normalize_y_multiindex(y: Array, coef: VolterraCoefficients) -> Array:
    """Return projected increment with trailing shape ``(q, m)``."""
    y = jnp.asarray(y, dtype=coef.alpha.dtype)
    if y.ndim < 2 or y.shape[-2:] != (coef.q, coef.m):
        raise ValueError(f"y must have trailing shape ({coef.q}, {coef.m}), got {tuple(y.shape)}.")
    return y


@lru_cache(maxsize=None)
def _packed_multiindex_navigation(
    q: int,
    trunc: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Host-side degree blocks and successor lookup matching the layout order."""
    tuples: list[tuple[int, ...]] = []
    by_degree: list[list[int]] = []

    for n in range(trunc + 1):
        block = list(_compositions_desc(total=n, q=q))
        by_degree.append(list(range(len(tuples), len(tuples) + len(block))))
        tuples.extend(block)

    index = {multi: i for i, multi in enumerate(tuples)}
    plus: list[tuple[int, ...]] = []
    for multi in tuples:
        vals = list(multi)
        total = sum(vals)
        row = []
        for r in range(q):
            if total < trunc:
                vals[r] += 1
                row.append(index[tuple(vals)])
                vals[r] -= 1
            else:
                row.append(-1)
        plus.append(tuple(row))

    return tuple(tuple(block) for block in by_degree), tuple(plus)


def _compositions_desc(total: int, q: int):
    """Compositions in the same graded order as ``build_multiindex_layout``."""
    if q == 1:
        yield (total,)
        return
    prefix = [0] * q

    def rec(pos: int, remaining: int):
        if pos == q - 1:
            prefix[pos] = remaining
            yield tuple(prefix)
            return
        for first in range(remaining, -1, -1):
            prefix[pos] = first
            yield from rec(pos + 1, remaining - first)

    yield from rec(0, total)


__all__ = ["eval_e", "eval_vte"]
