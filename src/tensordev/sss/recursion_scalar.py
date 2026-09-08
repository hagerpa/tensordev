from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.grading import (
    graded_polynomial_horner_first_level,
    graded_right_multiply_first_level,
)
from tensordev.core.utils.pytrees import tree_first_leaf, tree_map
from tensordev.sss.coeffs import FSSKCoefficients


Array = jax.Array


def _is_bigraded_core(core: Any) -> bool:
    return getattr(core, "grading", None) == "bidegree"


def _scalar_element(core: Any, value: Array):
    """Return a scalar-only tensor carrying ``value`` in its scalar block."""
    if not _is_bigraded_core(core):
        return (value,)
    layout = core.resolve_layout((0, 0), include_scalar=True)
    return BigradedTensor((value,), layout.spec)


def _prepend_scalar(core: Any, positive, scalar: Array, trunc: Any):
    """Prepend ``scalar`` to a first-on tensor without changing its layout."""
    if not _is_bigraded_core(core):
        return (scalar,) + tuple(positive)
    layout = core.resolve_layout(trunc, include_scalar=True)
    prototype = tree_first_leaf(positive)
    scalar = jnp.broadcast_to(scalar, prototype.shape[:-1] + (1,))
    return BigradedTensor((scalar,) + positive.blocks, layout.spec)


def _prepend_zero_scalar(core: Any, positive, trunc: Any):
    prototype = tree_first_leaf(positive)
    scalar = jnp.zeros(prototype.shape[:-1] + (1,), dtype=prototype.dtype)
    return _prepend_scalar(core, positive, scalar, trunc)


def _drop_scalar(core: Any, element):
    if not _is_bigraded_core(core):
        return tuple(element)[1:]
    return BigradedTensor(element.blocks[1:], element.spec.with_scalar(False))


def _horner_truncation(core: Any, trunc: Any, coefficient_trunc: int):
    if _is_bigraded_core(core):
        return trunc
    return coefficient_trunc - 1


def _g_horner_truncation(core: Any, trunc: Any, coefficient_trunc: int):
    if _is_bigraded_core(core):
        return trunc
    return coefficient_trunc - 2


def _assemble_element(core: Any, layout: Any, blocks: tuple[Array, ...]):
    if _is_bigraded_core(core):
        return BigradedTensor(blocks, layout.spec)
    return blocks


def _right_multiply_generator(
        core: Any,
        source: Any,
        generator: Array,
        *,
        trunc: Any,
):
    layout = core.resolve_layout(trunc, include_scalar=False)
    if _is_bigraded_core(core):
        source_contains = source.spec.contains
        source_block = source.__getitem__
    else:
        source = tuple(source)
        source_contains = lambda grade: 0 <= grade < len(source)
        source_block = source.__getitem__
    return graded_right_multiply_first_level(
        layout,
        source_contains=source_contains,
        source_block=source_block,
        generator_blocks=core._generator_blocks(generator, layout=layout),
        right_generator_output_block=core._right_multiply_generator_output_block,
        assemble=lambda blocks: _assemble_element(core, layout, blocks),
    )


def _polynomial_horner(
        core: Any,
        generator: Array,
        coefficients: Array,
        *,
        trunc: Any,
        coefficient_axes: int,
):
    """Evaluate one coefficient polynomial by native first-level actions."""
    layout = core.resolve_layout(trunc, include_scalar=True)
    max_order = max(layout.total_degree(grade) for grade in layout.grades)
    coefficient_count = coefficients.shape[-coefficient_axes - 1]
    coefficient_batch = coefficients.shape[:-coefficient_axes - 1]
    value_shape = coefficients.shape[-coefficient_axes:]
    generator_batch = generator.shape[:-coefficient_axes - 2]
    batch = jnp.broadcast_shapes(generator_batch, coefficient_batch)
    scalar_shape = batch + (1,) + value_shape + (1,)

    def coefficient_block(index: int, grade: Any, like: Array | None):
        if grade == layout.zero_grade:
            if index < coefficient_count:
                coefficient_index = (
                    (..., index) + (slice(None),) * coefficient_axes
                )
                value = coefficients[coefficient_index]
                value = jnp.expand_dims(value, axis=-coefficient_axes - 1)
                return jnp.broadcast_to(value[..., None], scalar_shape)
            return jnp.zeros(scalar_shape, dtype=coefficients.dtype)
        return jnp.zeros_like(like)

    return graded_polynomial_horner_first_level(
        layout,
        max_order=max_order,
        coefficient_block=coefficient_block,
        generator_blocks=core._generator_blocks(generator, layout=layout),
        right_generator_output_block=core._right_multiply_generator_output_block,
        assemble=lambda blocks: _assemble_element(core, layout, blocks),
    )


@partial(jax.jit, static_argnames=("core", "trunc"))
def init_state(
        coef: FSSKCoefficients,
        *,
        core: Any,
        trunc: Any = None,
) -> Any:
    """
    Initialize the zero scalar FSSK state for a single-step coefficient object.

    This module assumes that ``coef`` is already step-local. Its leading axes,
    if any, are treated as ordinary batch axes.

    The scalar recursion keeps the explicit singleton family axis. The result
    uses the selected core's scalar-omitting, first-on layout.

    Parameters
    ----------
    coef : FSSKCoefficients
        Scalar step-local FSSK coefficients. This function requires
        ``coef.q == 1``.
    core : JAX core
        Core defining the state layout.
    trunc : int or pair, optional
        Active truncation. Defaults to ``coef.trunc`` for total degree and to
        the core default otherwise.

    Returns
    -------
    tuple or BigradedTensor
        Zero state in the core's first-on layout.
    """
    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")

    if trunc is None and not _is_bigraded_core(core):
        trunc = coef.trunc
    active = core.normalize_truncation(trunc)
    max_order = sum(active) if _is_bigraded_core(core) else active
    if max_order > coef.trunc:
        raise ValueError(
            f"coefficients cover total order {coef.trunc}, but the active "
            f"truncation requires order {max_order}."
        )
    batch_shape = coef.E.shape[:-2]
    if _is_bigraded_core(core):
        layout = core.resolve_layout(active, include_scalar=False)
        return BigradedTensor(
            tuple(
                jnp.zeros(
                    batch_shape + (1, 1, coef.R, layout.block_width(grade)),
                    dtype=coef.E.dtype,
                )
                for grade in layout.grades
            ),
            layout.spec,
        )
    return tuple(
        jnp.zeros(batch_shape + (1, 1, coef.R, coef.m ** (r + 1)), dtype=coef.E.dtype)
        for r in range(active)
    )


@partial(jax.jit, static_argnames=("core", "trunc"))
def eval_fg(
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Any,
        trunc: Any = None,
) -> tuple[Any, Any]:
    """
    Evaluate the scalar Horner recursions for ``f`` and ``G``.

    This function assumes that ``coef`` is already step-local, so:

    - ``coef.E`` has shape ``batch + (R, R)``,
    - ``coef.psi`` has shape ``batch + (trunc, R)``,
    - ``coef.phi`` has shape ``batch + (1, trunc - 1, R, R)``.

    The input ``y`` may have any batch shape broadcastable against the leading
    batch shape of the coefficients.

    The returned core-native elements are:

    - ``f`` in ``(mathfrak R_{trunc-1})^{1 x R}``,
    - ``G`` in ``((mathfrak R_{trunc-2})^{R x R})^1``.

    For total-degree cores, ``f`` ends at degree ``trunc - 1`` and ``G`` at
    degree ``trunc - 2``. For a bidegree core, both use the active rectangle;
    algebraically zero terminal diagonals remain present because the native
    container is rectangular.

    Parameters
    ----------
    y : Array
        Projected path increment with shape ``batch + (m,)``.
    coef : FSSKCoefficients
        Scalar step-local FSSK coefficients. This function requires
        ``coef.q == 1``.
    core : Jax
        JAX core object.

    Returns
    -------
    f, G : tuple or BigradedTensor
        Coefficient polynomials in the core's standard-coordinate layout.
        Their trailing axes before the tensor-coordinate axis are ``(1, R)``
        and ``(1, R, R)`` respectively.
    """
    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")
    if y.shape[-1] != coef.m:
        raise ValueError(
            f"y must have trailing shape ({coef.m},), got {tuple(y.shape)}."
        )
    if coef.psi.shape[-2:] != (coef.trunc, coef.R):
        raise ValueError(
            "Scalar step-local coefficients must satisfy "
            f"psi.shape[-2:] == ({coef.trunc}, {coef.R}), got {coef.psi.shape[-2:]}."
        )
    if coef.trunc > 1 and coef.phi.shape[-4:] != (1, coef.trunc - 1, coef.R, coef.R):
        raise ValueError(
            "Scalar step-local coefficients must satisfy "
            f"phi.shape[-4:] == (1, {coef.trunc - 1}, {coef.R}, {coef.R}), "
            f"got {coef.phi.shape[-4:]}."
        )

    trunc = core.normalize_truncation(
        coef.trunc if trunc is None else trunc
    )
    N = coef.trunc
    horner_trunc = _horner_truncation(core, trunc, N)
    g_horner_trunc = _g_horner_truncation(core, trunc, N)
    psi = coef.psi
    phi = coef.phi[..., 0, :, :, :] if N > 1 else None

    f = _polynomial_horner(
        core,
        y[..., None, None, :],
        psi,
        trunc=horner_trunc,
        coefficient_axes=1,
    )

    if N == 1:
        G = _scalar_element(
            core,
            jnp.zeros(
                jnp.broadcast_shapes(coef.E.shape[:-2], y.shape[:-1])
                + (1, coef.R, coef.R, 1),
                dtype=coef.E.dtype,
            ),
        )
    else:
        G = _polynomial_horner(
            core,
            y[..., None, None, None, :],
            phi,
            trunc=g_horner_trunc,
            coefficient_axes=2,
        )

    return f, G


@partial(jax.jit, static_argnames=("core", "trunc"))
def update_state(
        Z: Any,
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Any,
        trunc: Any = None,
) -> Any:
    """
    Perform one scalar FSSK state update.

    This function assumes that ``coef`` is already step-local, so ``coef.E``,
    ``coef.psi`` and ``coef.phi`` carry only ordinary leading batch axes.
    Those batch axes may broadcast against the batch axes of ``y`` and ``Z``.

    The state is stored in **first-on format**: the degree-0 block is always
    identically zero and is not stored. Total-degree states are tuples;
    bidegree states are scalar-omitting ``BigradedTensor`` objects. Every
    block has trailing state axes ``(1, 1, R)`` before its coordinate axis.

    Parameters
    ----------
    Z : tuple or BigradedTensor
        Core-native first-on state.
    y : Array
        Projected path increment with shape ``batch + (m,)``.
    coef : FSSKCoefficients
        Scalar step-local FSSK coefficients. This function requires
        ``coef.q == 1``.
    core : Jax
        JAX core object.

    Returns
    -------
    tuple or BigradedTensor
        Updated state in the same layout as ``Z``.
    """
    if coef.q != 1:
        raise ValueError(f"recursion_scalar requires coef.q == 1, got {coef.q}.")
    trunc = core.normalize_truncation(
        coef.trunc if trunc is None else trunc
    )
    expected_levels = (
        len(core.resolve_layout(trunc, include_scalar=False).grades)
        if _is_bigraded_core(core)
        else coef.trunc
    )
    if len(Z) != expected_levels:
        raise ValueError(
            f"Z must have {expected_levels} homogeneous blocks (first-on), got {len(Z)}."
        )
    if y.shape[-1] != coef.m:
        raise ValueError(
            f"y must have trailing shape ({coef.m},), got {tuple(y.shape)}."
        )

    f, G = eval_fg(y, coef, core=core, trunc=trunc)
    horner_trunc = _horner_truncation(core, trunc, coef.trunc)

    Z_dense = _prepend_zero_scalar(core, Z, trunc)

    ZE = core.tensor_matrix_product_right(
        Z_dense,
        coef.E[..., None, :, :],
        trunc=trunc,
    )

    ZG = core.tensor_matrix_product(
        Z_dense,
        G,
        trunc=horner_trunc,
    )

    B = core.tensor_summation(
        f,
        tree_map(lambda level: jnp.sum(level, axis=-4), ZG),
        trunc=horner_trunc,
    )

    By = tree_map(
        lambda level: jnp.expand_dims(level, axis=-3),
        _right_multiply_generator(
            core,
            B,
            y[..., None, None, :],
            trunc=trunc,
        ),
    )
    By_dense = _prepend_zero_scalar(core, By, trunc)

    result_dense = core.tensor_summation(ZE, By_dense, trunc=trunc)
    return _drop_scalar(core, result_dense)


__all__ = [
    "init_state",
    "eval_fg",
    "update_state",
]
