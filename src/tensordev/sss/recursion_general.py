from __future__ import annotations

from functools import lru_cache, partial

import numpy as np
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


@lru_cache(maxsize=None)
def _precompute_navigation(
        q: int,
        trunc: int,
) -> tuple[
    tuple[list[int], ...],
    tuple[tuple[np.ndarray, ...] | None, ...],
]:
    """
    Host-side precomputation for the batched (stacked multi-index) recursion.

    Returned objects use only Python/NumPy and are cached so that JIT
    re-traces do not repeat work.

    Parameters
    ----------
    q, trunc:
        Same as ``_packed_multiindex_navigation``.

    Returns
    -------
    global_idx_by_deg:
        ``global_idx_by_deg[n]`` is a plain Python ``list[int]`` of the packed
        global indices for degree ``n``.
    succ_local_by_n_r:
        ``succ_local_by_n_r[n]`` is ``None`` when ``n == trunc``; otherwise it
        is a tuple of ``q`` NumPy integer arrays, where
        ``succ_local_by_n_r[n][r]`` has shape ``(num_n,)`` and stores the
        *local* index of ``plus[ell][r]`` within the degree-``(n+1)`` block,
        for each ``ell`` in ``global_idx_by_deg[n]``.
    """
    by_degree, plus = _packed_multiindex_navigation(q, trunc)

    # inverse map: global_idx -> local position within its degree block
    local_of_global: dict[int, int] = {}
    for block in by_degree:
        for local, gidx in enumerate(block):
            local_of_global[gidx] = local

    global_idx_by_deg = tuple(
        np.array(list(block), dtype=np.intp) for block in by_degree
    )

    succ_local_list: list[tuple[np.ndarray, ...] | None] = []
    for n, block in enumerate(by_degree):
        if n == trunc:
            succ_local_list.append(None)
        else:
            succ_r = tuple(
                np.array(
                    [local_of_global[plus[ell_idx][r]] for ell_idx in block],
                    dtype=np.intp,
                )
                for r in range(q)
            )
            succ_local_list.append(succ_r)

    return global_idx_by_deg, tuple(succ_local_list)


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

    Notes
    -----
    **Batched multi-index recursion** (compilation-cost reduction)

    The inner loop over multi-indices (whose count grows as C(N+q-2,q)) is
    replaced by a single batched operation per ``(degree, r)`` pair.  Instead
    of one ``DenseElem`` per multi-index, we keep one ``DenseElem`` per degree
    where the multi-index axis is a leading batch dimension:

    * ``F_stack[n]``: level ``k`` has shape ``batch + (num_n, 1, R, m**k)``
    * ``G_stack[n]``: level ``k`` has shape ``batch + (num_n, q, R, R, m**k)``

    All gather / scale / shuffle / summation operations act on these batched
    tensors, reducing the number of XLA ops from O(q·C(N+q-2,q)) to O(N·q).
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

    global_idx_by_deg, succ_local_by_n_r = _precompute_navigation(q, N - 1)

    inv_factorial = coef.layout.inv_factorial.astype(dtype)
    back_trans = coef.layout.backward_transition.astype(dtype)  # (layout.size, q)

    g_zero = _zero_g0(y, coef)  # batch + (q, R, R, 1)

    # F_stack[n]: DenseElem, level k shape → batch + (num_n, 1, R, m**k)
    # G_stack[n]: DenseElem, level k shape → batch + (num_n, q, R, R, m**k)
    F_stack: list[DenseElem | None] = [None] * N
    G_stack: list[DenseElem | None] = [None] * N

    for n in range(N - 1, -1, -1):
        max_f_degree = N - 1 - n
        max_g_degree = N - 2 - n

        idx_n = global_idx_by_deg[n]   # Python list[int], static
        num_n = len(idx_n)

        # scale factors: shape (num_n,)
        scale_n = inv_factorial[idx_n]

        # ── F base: batch + (num_n, 1, R, 1) ─────────────────────────────
        # coef.psi[..., idx_n, :] → batch + (num_n, R)
        # [..None, :, None] → batch + (num_n, 1, R, 1)
        f_base_n = (
            coef.psi[..., idx_n, :][..., None, :, None]
            * scale_n[:, None, None, None]
        )
        f_stack_n: DenseElem = (f_base_n,)

        # ── G base ────────────────────────────────────────────────────────
        if n <= N - 2:
            # coef.phi[..., :, idx_n, :, :] → batch + (q, num_n, R, R)
            # swapaxes(-4,-3)              → batch + (num_n, q, R, R)
            # [..., None] * scale          → batch + (num_n, q, R, R, 1)
            phi_gathered = jnp.swapaxes(
                coef.phi[..., :, idx_n, :, :], -4, -3
            )
            g_base_n = (
                phi_gathered[..., None]
                * scale_n[:, None, None, None, None]
            )
            g_stack_n: DenseElem = (g_base_n,)
        else:
            # n == N-1: no phi base, broadcast g_zero over num_n
            # g_zero: batch + (q, R, R, 1)
            # expand to: batch + (1, q, R, R, 1) then broadcast to num_n
            g_zero_exp = jnp.expand_dims(g_zero, axis=g_zero.ndim - 4)
            g_base_n = jnp.broadcast_to(
                g_zero_exp,
                g_zero_exp.shape[:-5] + (num_n,) + g_zero_exp.shape[-4:],
            )
            g_stack_n = (g_base_n,)

        # ── Accumulate over components r ──────────────────────────────────
        if n <= N - 2:
            succ_local_r = succ_local_by_n_r[n]  # tuple of q numpy int arrays

            for r in range(q):
                sl = succ_local_r[r]  # shape (num_n,) — local indices into degree n+1

                # transition scalars for this (degree, r): shape (num_n,)
                trans_r = back_trans[idx_n, r]

                # ── F: gather → scale → shuffle → accumulate ──────────────
                # F_stack[n+1][k]: batch + (num_{n+1}, 1, R, m**k)
                # After gather:    batch + (num_n,    1, R, m**k)
                F_gathered = tuple(
                    lvl[..., sl, :, :, :] for lvl in F_stack[n + 1]
                )
                F_scaled = tuple(
                    lvl * trans_r[:, None, None, None] for lvl in F_gathered
                )
                F_shuff = core.tensor_shuffle_vector(
                    F_scaled,
                    y[..., r, None, None, None, :],   # batch + (1, 1, 1, m)  ← extra None for num_n axis
                    trunc=max_f_degree,
                )
                f_stack_n = core.tensor_summation(
                    f_stack_n,
                    (jnp.zeros_like(f_base_n),) + F_shuff,
                    trunc=max_f_degree,
                )

                # ── G: gather → scale → shuffle → accumulate ──────────────
                # G_stack[n+1][k]: batch + (num_{n+1}, q, R, R, m**k)
                # After gather:    batch + (num_n,    q, R, R, m**k)
                G_gathered = tuple(
                    lvl[..., sl, :, :, :, :] for lvl in G_stack[n + 1]
                )
                G_scaled = tuple(
                    lvl * trans_r[:, None, None, None, None] for lvl in G_gathered
                )
                G_shuff = core.tensor_shuffle_vector(
                    G_scaled,
                    y[..., r, None, None, None, None, :],  # batch + (1, 1, 1, 1, m)  ← extra None for num_n axis
                    trunc=max_g_degree,
                )
                g_stack_n = core.tensor_summation(
                    g_stack_n,
                    (jnp.zeros_like(g_base_n),) + G_shuff,
                    trunc=max_g_degree,
                )

        F_stack[n] = f_stack_n
        G_stack[n] = g_stack_n

    # ── Extract the degree-0 root (local index 0 within degree-0 block) ───
    # F_stack[0][k]: batch + (1, 1, R, m**k)  →  batch + (1, R, m**k)
    f0: DenseElem = tuple(lvl[..., 0, :, :, :] for lvl in F_stack[0])

    if N == 1:
        g0: DenseElem = (g_zero,)
    else:
        # G_stack[0][k]: batch + (1, q, R, R, m**k)  →  batch + (q, R, R, m**k)
        g0 = tuple(lvl[..., 0, :, :, :, :] for lvl in G_stack[0])

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
