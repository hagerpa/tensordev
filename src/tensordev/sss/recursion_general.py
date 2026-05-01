from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem, DenseElemFirstOn
from tensordev.sss.coeffs import FSSKCoefficients

Array = jax.Array


@partial(jax.jit, static_argnames=("core",))
def init_state(
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> DenseElemFirstOn:
    """
    Initialize the zero FSSK state for the general shuffle recursion.

    The state is stored in first-on format: level ``r`` carries tensor degree
    ``r + 1`` and has shape

        ``batch + (q, 1, R, m**(r + 1))``.

    Parameters
    ----------
    coef:
        Step-local FSSK coefficients.
    core:
        Standard tensor-algebra core.
    """
    del core

    batch_shape = coef.E.shape[:-2]
    return tuple(
        jnp.zeros(
            batch_shape + (coef.q, 1, coef.R, coef.m ** (r + 1)),
            dtype=coef.E.dtype,
        )
        for r in range(coef.trunc)
    )


@partial(jax.jit, static_argnames=("core",))
def eval_fg(
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> tuple[DenseElem, DenseElem]:
    r"""
    Evaluate the Part II ``EvalFG`` shuffle recursion.

    The coefficient arrays store packed families ``psi_ell`` and
    ``Phi_{p, ell}``.  The shuffle recursion initializes the dynamic programs
    with ``psi_ell / ell!`` and ``Phi_{p, ell} / ell!``; this is implemented
    through ``coef.layout.inv_factorial``.

    Parameters
    ----------
    y:
        Projected increment with trailing shape ``(q, m)``. For the q=1 sanity
        path, trailing shape ``(m,)`` is also accepted and normalized to
        ``(1, m)`` internally.
    coef:
        Step-local FSSK coefficients. ``coef.layout`` must be the packed
        multi-index layout for ``q`` and degree ``coef.trunc - 1``.
    core:
        Standard tensor-algebra core, used for level-wise summation and shuffle.

    Returns
    -------
    f:
        Dense element with degrees ``0, ..., N-1``. Level ``r`` has shape
        ``batch + (1, R, m**r)``.
    G:
        Dense element with degrees ``0, ..., N-2`` when ``N > 1``. Level ``r``
        has shape ``batch + (q, R, R, m**r)``. For ``N == 1`` a single zero
        degree-0 level is returned for shape stability.
    """
    y = _normalize_y(y, coef)

    N = coef.trunc
    q = coef.q
    R = coef.R
    dtype = coef.E.dtype

    if coef.psi.shape[-2:] != (coef.layout.size, R):
        raise ValueError(
            "General coefficients must satisfy "
            f"psi.shape[-2:] == ({coef.layout.size}, {R}), "
            f"got {coef.psi.shape[-2:]}"
        )

    if coef.phi.shape[-4:] != (q, coef.Mphi, R, R):
        raise ValueError(
            "General coefficients must satisfy "
            f"phi.shape[-4:] == ({q}, {coef.Mphi}, {R}, {R}), "
            f"got {coef.phi.shape[-4:]}"
        )

    by_degree, plus = _packed_multiindex_navigation(q, N - 1)
    inv_factorial = coef.layout.inv_factorial.astype(dtype)

    F: list[DenseElem | None] = [None] * coef.layout.size
    G: list[DenseElem | None] = [None] * coef.layout.size

    g_zero = _zero_g0(y, coef)

    for n in range(N - 1, -1, -1):
        max_f_degree = N - 1 - n
        max_g_degree = N - 2 - n

        for ell_idx in by_degree[n]:
            scale = inv_factorial[ell_idx]

            f_base: DenseElem = (
                coef.psi[..., ell_idx, :][..., None, :, None] * scale,
            )
            f_ell = f_base

            if n <= N - 2:
                g_base: DenseElem = (
                    coef.phi[..., :, ell_idx, :, :][..., None] * scale,
                )
            else:
                g_base = (g_zero,)
            g_ell = g_base

            if n <= N - 2:
                for r in range(q):
                    succ = plus[ell_idx][r]
                    transition = coef.layout.backward_transition[ell_idx, r].astype(dtype)

                    fy = core.tensor_shuffle_vector(
                        F[succ],
                        y[..., r, None, None, :],
                        trunc=max_f_degree,
                    )
                    fy = tuple(transition * fy_k for fy_k in fy)

                    f_ell = core.tensor_summation(
                        f_ell,
                        (jnp.zeros_like(f_base[0]),) + fy,
                        trunc=max_f_degree,
                    )

                    gy = core.tensor_shuffle_vector(
                        G[succ],
                        y[..., r, None, None, None, :],
                        trunc=max_g_degree,
                    )
                    gy = tuple(transition * gy_k for gy_k in gy)

                    g_ell = core.tensor_summation(
                        g_ell,
                        (jnp.zeros_like(g_base[0]),) + gy,
                        trunc=max_g_degree,
                    )

            F[ell_idx] = f_ell
            G[ell_idx] = g_ell
    f0 = F[0]
    g0 = G[0]
    if f0 is None or g0 is None:
        raise RuntimeError("internal error: root multi-index was not evaluated")

    if N == 1:
        g0 = (g_zero,)

    return f0, g0


@partial(jax.jit, static_argnames=("core",))
def update_state(
        Z: DenseElemFirstOn,
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Jax,
) -> DenseElemFirstOn:
    r"""
    Perform one general FSSK state update.

    The update is

        ``Z_new^p = Z^p . E + B \otimes y_p``

    with

        ``B = f + sum_l Z^l . G^l``.

    This is the q-component analogue of the scalar Horner update, with ``f``
    and ``G`` supplied by the Part II shuffle recursion.
    """
    y = _normalize_y(y, coef)

    if len(Z) != coef.trunc:
        raise ValueError(
            f"Z must have {coef.trunc} homogeneous levels (first-on), got {len(Z)}."
        )
    for r, z in enumerate(Z):
        expected = (coef.q, 1, coef.R, coef.m ** (r + 1))
        if z.shape[-4:] != expected:
            raise ValueError(
                f"Z[{r}] must have trailing shape {expected}, got {z.shape[-4:]}"
            )

    f, G = eval_fg(y, coef, core=core)

    zero_level = jnp.zeros(Z[0].shape[:-1] + (1,), dtype=Z[0].dtype)
    Z_dense = (zero_level,) + tuple(Z)

    ZE = core.tensor_matrix_product_right(
        Z_dense,
        coef.E[..., None, :, :],
        trunc=coef.trunc,
    )

    ZG = core.tensor_matrix_product(
        Z_dense,
        G,
        trunc=coef.trunc - 1,
    )

    B = core.tensor_summation(
        f,
        tuple(jnp.sum(level, axis=-4) for level in ZG),
        trunc=coef.trunc - 1,
    )

    # Insert the q/component axis into B and multiply by the degree-1 letters
    # y_p.  This uses the ordinary tensor product, not the shuffle product.
    B_by_component = tuple(jnp.expand_dims(level, axis=-3) for level in B)
    y_letters = (y[..., :, None, None, :],)

    By = (jnp.zeros_like(Z_dense[0]),) + core.tensor_product(
        B_by_component,
        y_letters,
        trunc=coef.trunc,
        b_first_on=True,
    )

    result_dense = core.tensor_summation(ZE, By, trunc=coef.trunc)
    return result_dense[1:]


def _normalize_y(y: Array, coef: FSSKCoefficients) -> Array:
    """Return projected increments with trailing shape ``(q, m)``."""
    y = jnp.asarray(y, dtype=coef.E.dtype)
    if coef.q == 1 and y.shape[-1] == coef.m:
        if y.ndim >= 2 and y.shape[-2:] == (1, coef.m):
            return y
        return y[..., None, :]
    if y.ndim < 2 or y.shape[-2:] != (coef.q, coef.m):
        raise ValueError(
            f"y must have trailing shape ({coef.q}, {coef.m}), got {tuple(y.shape)}."
        )
    return y


def _zero_g0(y: Array, coef: FSSKCoefficients) -> Array:
    """Degree-0 zero for the q-family of G matrices."""
    batch_shape = jnp.broadcast_shapes(coef.E.shape[:-2], y.shape[:-2])
    return jnp.zeros(batch_shape + (coef.q, coef.R, coef.R, 1), dtype=coef.E.dtype)


@lru_cache(maxsize=None)
def _packed_multiindex_navigation(
        q: int,
        trunc: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """
    Host-side packed multi-index navigation matching ``build_multiindex_layout``.

    The degree loops and successor lookups must stay static under ``jax.jit``.
    The coefficient arrays themselves are still indexed in the canonical packed
    order defined by ``coef.layout``.
    """
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
    """Compositions in the same graded order as ``tensordev.util.combinatorics``."""
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


__all__ = ["init_state", "eval_fg", "update_state"]
