from __future__ import annotations

from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp

from tensordev.volterra.algebra import (
    ResolvedVolterraAlgebra,
    require_volterra_shuffle,
)
from tensordev.volterra.coeffs import VolterraCoefficients, validate_volterra_coefficients
from tensordev.volterra.eval_scalar import (
    _eval_e_positive_q1,
    _normalize_history_element,
    _resolve_evaluator_algebra,
    eval_vte as eval_vte_scalar,
)
from tensordev.util.combinatorics import multiindex_batched_navigation


Array = jax.Array


def eval_e(
    y: Array,
    coef: VolterraCoefficients,
    *,
    core: Any = None,
    trunc: Any = None,
    algebra: ResolvedVolterraAlgebra | None = None,
):
    r"""Evaluate the packed multi-index local Volterra increment ``E``.

    The returned native graded element has a zero degree-zero block.  This
    evaluator implements the general ``q >= 1`` packed multi-index recursion
    under the coefficient symmetry hypothesis.  In the outer algorithm it is
    used for ``q > 1``; ``q == 1`` has a cheaper scalar fast path in
    :mod:`tensordev.volterra.eval_scalar`.
    """
    resolved = _resolve_evaluator_algebra(
        coef, core=core, trunc=trunc, algebra=algebra
    )
    e_positive = _eval_e_first_on(y, coef, algebra=resolved)
    return resolved.embed_positive(e_positive)


def eval_vte(
    v,
    y: Array,
    coef: VolterraCoefficients,
    *,
    core: Any = None,
    trunc: Any = None,
    algebra: ResolvedVolterraAlgebra | None = None,
):
    r"""Evaluate the packed multi-index local contribution ``v tensor E``."""
    resolved = _resolve_evaluator_algebra(
        coef, core=core, trunc=trunc, algebra=algebra
    )
    if coef.q == 1:
        return eval_vte_scalar(v, y, coef, algebra=resolved)

    history = _normalize_history_element(v, resolved)
    # Keep E positive-only through the history product.  This avoids an
    # artificial scalar contribution and preserves the core's native
    # first-on layout.
    e_positive = _eval_e_first_on(y, coef, algebra=resolved)
    out_positive = resolved.core.tensor_product(
        history,
        e_positive,
        trunc=resolved.truncation,
        b_first_on=True,
    )
    return resolved.embed_positive(out_positive)


@lru_cache(maxsize=None)
def _coefficient_worksets(algebra: ResolvedVolterraAlgebra):
    """Exact sparse worksets for packed coefficient degrees ``0..D-1``.

    Workset ``d`` contains precisely the grades that can reach a retained
    positive local-factor grade after ``d`` shuffle generators and the final
    concatenated generator.
    """
    targets = tuple(algebra.positive_grades)
    worksets = []
    for _ in range(algebra.max_order):
        sources = {
            source_grade
            for target_grade in targets
            for source_grade, _ in algebra.generator_splits(target_grade)
        }
        grades = tuple(grade for grade in algebra.grades if grade in sources)
        workset = algebra.workset(grades)
        if not workset.contains(algebra.zero_grade):
            raise ValueError("packed coefficient workset lost the scalar grade.")
        worksets.append(workset)
        targets = grades
    return tuple(worksets)


def _eval_e_first_on(
    y: Array,
    coef: VolterraCoefficients,
    *,
    core: Any = None,
    trunc: Any = None,
    algebra: ResolvedVolterraAlgebra | None = None,
):
    r"""Return the native positive-only local Volterra factor ``E``."""
    resolved = _resolve_evaluator_algebra(
        coef, core=core, trunc=trunc, algebra=algebra
    )
    if coef.q == 1:
        return _eval_e_positive_q1(y, coef, algebra=resolved)
    require_volterra_shuffle(resolved, feature="The q > 1 Volterra evaluator")
    return _eval_e_positive_general(y, coef, algebra=resolved)


@partial(jax.jit, static_argnames=("algebra",))
def _eval_e_positive_general(
    y: Array,
    coef: VolterraCoefficients,
    *,
    algebra: ResolvedVolterraAlgebra,
):
    r"""Packed ``q > 1`` recursion on exact native grade worksets."""
    validate_volterra_coefficients(coef)
    if coef.q <= 1:
        raise ValueError(f"general evaluator requires q > 1, got q={coef.q}.")

    y = _normalize_y_multiindex(y, coef)
    dtype = jnp.result_type(y, coef.alpha)
    y = y.astype(dtype)
    alpha = coef.alpha.astype(dtype)

    depth = algebra.max_order
    q = coef.q
    global_idx_by_degree, successors = multiindex_batched_navigation(
        q, coef.trunc - 1
    )
    inverse_factorial = coef.layout.inv_factorial.astype(dtype)
    backward_transition = coef.layout.backward_transition.astype(dtype)
    batch_shape = jnp.broadcast_shapes(coef.leading_shape, y.shape[:-2])
    worksets = _coefficient_worksets(algebra)

    # F[d][h] has leading shape batch + (num_multiindices_d, q) and
    # coordinate width native to tensor grade h.
    stack: list[tuple[Array, ...] | None] = [None] * depth

    for degree in range(depth - 1, -1, -1):
        workset = worksets[degree]
        indices = global_idx_by_degree[degree]
        count = len(indices)
        scale = inverse_factorial[indices]
        gathered_alpha = jnp.moveaxis(alpha[..., :, indices], -1, -2)
        base = gathered_alpha[..., None] * scale[:, None, None]
        base = jnp.broadcast_to(base, batch_shape + (count, q, 1))

        values: list[Array | None] = [None] * workset.size
        values[workset.index(algebra.zero_grade)] = base

        if degree < depth - 1:
            next_workset = worksets[degree + 1]
            next_values = stack[degree + 1]
            if next_values is None:
                raise AssertionError("packed coefficient recursion is incomplete.")
            positive_target = algebra.workset(
                tuple(
                    grade
                    for grade in workset.grades
                    if grade != algebra.zero_grade
                )
            )
            successor_by_component = successors[degree]

            for component in range(q):
                successor_indices = successor_by_component[component]
                transition = backward_transition[indices, component]
                gathered = tuple(
                    block[..., successor_indices, :, :]
                    * transition[:, None, None]
                    for block in next_values
                )
                action = algebra.shuffle_generator_action(
                    next_workset,
                    gathered,
                    y[..., component, None, None, :],
                    positive_target,
                )
                for grade, block in zip(positive_target.grades, action):
                    index = workset.index(grade)
                    previous = values[index]
                    values[index] = block if previous is None else previous + block

        if any(block is None for block in values):
            raise AssertionError("packed coefficient workset has an unset block.")
        stack[degree] = tuple(values)  # type: ignore[arg-type]

    root_values = stack[0]
    if root_values is None:
        raise AssertionError("packed coefficient root was not constructed.")
    root_workset = worksets[0]
    root = tuple(block[..., 0, :, :] for block in root_values)

    # The coefficient-component axis is an ordinary leading batch axis here.
    # Pair it with the corresponding y_p, then sum that axis only after the
    # fused final-generator action has produced complete native blocks.
    positive_workset = algebra.workset(algebra.positive_grades)
    by_component = algebra.right_generator_action(
        root_workset, root, y, positive_workset
    )
    blocks = tuple(jnp.sum(block, axis=-2) for block in by_component)
    return algebra.assemble(blocks, positive=True)


def _normalize_y_multiindex(y: Array, coef: VolterraCoefficients) -> Array:
    """Return projected increment with trailing shape ``(n, m)``."""
    y = jnp.asarray(y, dtype=coef.alpha.dtype)
    if (
        coef.q == 1
        and y.shape[-1:] == (coef.m,)
        and (y.ndim == 1 or y.shape[-2] != 1)
    ):
        return y[..., None, :]
    if y.ndim < 2 or y.shape[-2:] != (coef.q, coef.m):
        raise ValueError(
            f"y must have trailing shape ({coef.q}, {coef.m}), "
            f"got {tuple(y.shape)}."
        )
    return y


__all__ = ["eval_e", "eval_vte"]
