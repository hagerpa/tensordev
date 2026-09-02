from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from tensordev._backend import get_default_core
from tensordev.volterra.algebra import (
    ResolvedVolterraAlgebra,
    resolve_volterra_algebra,
)
from tensordev.volterra.coeffs import (
    VolterraCoefficients,
    validate_volterra_coefficients,
)


Array = jax.Array

def _resolve_evaluator_algebra(
    coef: VolterraCoefficients,
    *,
    core: Any = None,
    trunc: Any = None,
    algebra: ResolvedVolterraAlgebra | None = None,
) -> ResolvedVolterraAlgebra:
    """Resolve one evaluator algebra from explicit or configured defaults."""
    if algebra is not None:
        if core is not None or trunc is not None:
            raise TypeError("Pass either algebra= or core=/trunc=, not both.")
        if algebra.alphabet_dim != coef.m:
            raise ValueError(
                f"resolved alphabet dimension {algebra.alphabet_dim} does not "
                f"match coefficient dimension m={coef.m}."
            )
        resolved = algebra
    else:
        selected_core = get_default_core() if core is None else core
        active = trunc
        if active is None and getattr(selected_core, "default_truncation", None) is None:
            # Preserve coefficient-depth inference for an unbounded core.
            active = coef.trunc
        resolved = resolve_volterra_algebra(selected_core, active, coef.m)

    if resolved.max_order > coef.trunc:
        raise ValueError(
            f"coefficient depth {coef.trunc} is smaller than the resolved "
            f"Volterra order {resolved.max_order}."
        )
    return resolved


def _normalize_y_q1(y: Array, coef: VolterraCoefficients) -> Array:
    """Return scalar projected increment with trailing shape ``(m,)``."""
    y = jnp.asarray(y, dtype=coef.alpha.dtype)
    if y.shape[-1] == coef.m:
        if y.ndim >= 2 and y.shape[-2:] == (1, coef.m):
            return y[..., 0, :]
        return y
    raise ValueError(
        f"y must have trailing shape ({coef.m},) or (1, {coef.m}), "
        f"got {tuple(y.shape)}."
    )


def _normalize_history_element(v, algebra: ResolvedVolterraAlgebra):
    """Normalize a variable-length total-degree tuple to the active depth."""
    if getattr(algebra.core, "grading", None) != "total_degree":
        # Native graded containers intentionally retain an exact-layout
        # invariant; their core validates the element during block access.
        return v

    levels = tuple(v)
    if not levels:
        raise ValueError("v must contain at least the degree-zero level.")
    required = algebra.max_order + 1
    selected = list(levels[:required])
    batch_shape = tuple(selected[0].shape[:-1])
    dtype = selected[0].dtype
    for degree in range(len(selected), required):
        selected.append(
            jnp.zeros(
                batch_shape + (algebra.block_width(degree),), dtype=dtype
            )
        )
    return tuple(selected)


@partial(jax.jit, static_argnames=("algebra",))
def _eval_e_positive_q1(
    y: Array,
    coef: VolterraCoefficients,
    *,
    algebra: ResolvedVolterraAlgebra,
):
    """Build the positive local factor for ``q == 1`` without shuffle plans."""
    validate_volterra_coefficients(coef)
    if coef.q != 1:
        raise ValueError(f"scalar evaluator requires q == 1, got q={coef.q}.")

    y = _normalize_y_q1(y, coef)
    dtype = jnp.result_type(y, coef.alpha)
    y = y.astype(dtype)
    alpha = coef.alpha.astype(dtype)
    batch_shape = jnp.broadcast_shapes(coef.leading_shape, y.shape[:-1])

    source = algebra.diagonal(0)
    powers = (
        jnp.ones(batch_shape + (source.widths[0],), dtype=dtype),
    )
    positive_by_grade: dict[Any, Array] = {}

    for order in range(1, algebra.max_order + 1):
        target = algebra.diagonal(order)
        powers = algebra.right_generator_action(source, powers, y, target)
        beta = alpha[..., 0, order - 1]
        for grade, block in zip(target.grades, powers):
            positive_by_grade[grade] = block * beta[..., None]
        source = target

    return algebra.assemble(
        tuple(positive_by_grade[grade] for grade in algebra.positive_grades),
        positive=True,
    )


@partial(jax.jit, static_argnames=("algebra",))
def _eval_vte_q1(
    v,
    y: Array,
    coef: VolterraCoefficients,
    *,
    algebra: ResolvedVolterraAlgebra,
):
    """Pruned diagonal Horner evaluation shared by total and bidegree."""
    validate_volterra_coefficients(coef)
    if coef.q != 1:
        raise ValueError(f"scalar evaluator requires q == 1, got q={coef.q}.")

    y = _normalize_y_q1(y, coef)
    v_zero = algebra.block(v, algebra.zero_grade)
    dtype = jnp.result_type(y, coef.alpha, v_zero)
    y = y.astype(dtype)
    alpha = coef.alpha.astype(dtype)

    def history_block(grade: Any) -> Array:
        return algebra.block(v, grade).astype(dtype)

    output_by_grade: dict[Any, Array] = {}
    scalar = algebra.diagonal(0)

    for order in range(1, algebra.max_order + 1):
        beta = alpha[..., 0, order - 1]
        values = (v_zero.astype(dtype) * beta[..., None],)
        source = scalar

        for current_order in range(1, order):
            target = algebra.diagonal(current_order)
            advanced = algebra.right_generator_action(source, values, y, target)
            coefficient = alpha[..., 0, order - current_order - 1]
            values = tuple(
                action + history_block(grade) * coefficient[..., None]
                for grade, action in zip(target.grades, advanced)
            )
            source = target

        target = algebra.diagonal(order)
        diagonal = algebra.right_generator_action(source, values, y, target)
        output_by_grade.update(zip(target.grades, diagonal))

    positive = algebra.assemble(
        tuple(output_by_grade[grade] for grade in algebra.positive_grades),
        positive=True,
    )
    return algebra.embed_positive(positive)


def eval_vte(
    v,
    y: Array,
    coef: VolterraCoefficients,
    *,
    core: Any = None,
    trunc: Any = None,
    algebra: ResolvedVolterraAlgebra | None = None,
):
    r"""Evaluate ``v \otimes E`` by the scalar pruned Horner scheme.

    The same diagonal recurrence is used for total-degree and bidegree cores.
    When no core is supplied, the configured default core is used; an
    unbounded total-degree default infers its active depth from ``coef.trunc``.
    """
    resolved = _resolve_evaluator_algebra(
        coef, core=core, trunc=trunc, algebra=algebra
    )
    history = _normalize_history_element(v, resolved)
    return _eval_vte_q1(history, y, coef, algebra=resolved)


__all__ = ["eval_vte"]
