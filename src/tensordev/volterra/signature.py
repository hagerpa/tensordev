"""Volterra signature — high-level entry point and VolterraSignature wrapper."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tensordev.core.universal import DenseElem
from tensordev.volterra.kernel import ConvolutionKernel, FractionalKernel, GammaKernel
from tensordev.volterra.iteration_quad import quadratic_iteration as _vsig_quadratic
from tensordev.volterra.iteration_fft import fft_iteration as _vsig_fft, PrecomputedLagTables
from tensordev.volterra.iteration_pc import pc_iteration as _vsig_pc

Array = jax.Array


def vsig(
        X: Array,
        *,
        kernel: ConvolutionKernel,
        trunc: int,
        dt: Array | float = 1.0,
        axis: int = -2,
        block_size: Optional[int] = None,
        accumulate: bool = True,
        starting_point: Optional[DenseElem] = None,
        output_starting_point: bool = False,
        increment_input: bool = False,
        order: int = 0,
        dyadic_order: int = 0,
        scheme: Literal["auto", "fft", "quadratic", "adams"] = "auto",
        lag_tables: Optional[PrecomputedLagTables] = None,
) -> DenseElem:
    """Compute the truncated Volterra signature of ``X``.

    High-level entry point that handles scheme selection, blocking, and
    accumulation before delegating to the low-level iteration kernels.

    Parameters
    ----------
    X:
        Path nodes or increments.  The trailing axis is the path dimension
        ``kernel.path_dim``; ``axis`` is the step/node axis.
    kernel:
        Volterra kernel supplying projections and coefficient builders.
    trunc:
        Tensor truncation level (positive integer).
    dt:
        Step size(s).  A scalar gives a uniform grid; a 1-D array of length
        ``S`` gives a non-uniform grid via cumulative sums (default ``1.0``).
    axis:
        Step/node axis of ``X`` (default ``-2``).
    block_size:
        Number of steps per emitted block.  ``None`` (default) → single
        terminal result.  With ``accumulate=True`` the signature at the end
        of each block is returned (block axis inserted at ``axis``).  With
        ``accumulate=False`` each block is treated as an independent path
        via rebatch (block axis inserted at ``axis``).
    accumulate:
        If ``True`` (default) history carries across blocks (standard
        Volterra semantics).  If ``False`` each block is processed
        independently by reshaping the batch dimension.  Ignored when
        ``block_size`` is ``None``.
    starting_point:
        Optional unit-level seed prepended to the output when
        ``output_starting_point=True``.  Defaults to the tensor unit.
    output_starting_point:
        If ``True``, prepend the seed (unit or ``starting_point``) to the
        output along ``axis``.
    increment_input:
        If ``True``, treat ``X`` as increments and skip :func:`jnp.diff`.
    order:
        Quadrature order for the higher-order basis-expansion scheme.
        ``0`` (default) left-point approximation.
    dyadic_order:
        Non-negative integer.  Each increment is split into
        ``2**dyadic_order`` equal sub-increments.  ``0`` (default) leaves
        the path unchanged.
    scheme:
        Which iteration scheme to use.

        ``"auto"``
            Use ``"fft"`` when ``dt`` is a scalar (uniform grid) **and**
            the expected FFT op-count is lower than the quadratic op-count
            (comparing ``S·log₂J_eff·N^q`` vs ``S²`` or ``S²·N``, with
            ``q = kernel.q``);
            otherwise falls back to ``"quadratic"``.
        ``"fft"``
            FFT-based convolution — uniform grids only.
        ``"quadratic"``
            General quadratic recursion — supports non-uniform grids.
        ``"adams"``
            Adams predictor-corrector (product-integration PC/Euler).
            Requires a :class:`FractionalKernel` with ``q=1`` and a
            uniform grid.  ``order=0`` → Euler; ``order≥1`` → PC.

    Returns
    -------
    DenseElem
        Terminal Volterra signature when ``block_size`` is ``None``.
        With blocking, each level carries an extra block axis at ``axis``.
        With ``output_starting_point=True``, the seed is prepended along
        that axis.

    Raises
    ------
    ValueError
        For invalid truncation, scheme, block size, or non-divisible ``S``.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if scheme not in ("auto", "fft", "quadratic", "adams"):
        raise ValueError(f"scheme must be 'auto', 'fft', 'quadratic', or 'adams', got {scheme!r}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}.")
    if block_size is not None and block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size}.")

    # scheme selection
    _use_adams = scheme == "adams"
    if scheme == "auto":
        if np.ndim(dt) == 0:
            _x_shape = np.shape(X)
            _s_eff = (_x_shape[axis % len(_x_shape)] - (0 if increment_input else 1)) * (1 << dyadic_order)
            _j_eff = 1 << math.ceil(math.log2(max(2 * _s_eff - 1, 2))) if _s_eff > 1 else 1
            _fft_cost = _s_eff * math.log2(_j_eff) * (trunc ** kernel.q)
            _quad_cost = _s_eff ** 2 if kernel.q <= 1 else _s_eff ** 2 * trunc
            _use_fft = _s_eff > 1 and _fft_cost < _quad_cost
        else:
            _use_fft = False
    elif scheme == "fft":
        _use_fft = True
    else:
        _use_fft = False

    # preprocessing
    X = jnp.asarray(X)
    if X.ndim < 2:
        raise ValueError("X must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % X.ndim
    if axis_norm == X.ndim - 1:
        raise ValueError("axis must identify the step axis, not the trailing path dimension.")
    if X.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"X trailing dimension must be {kernel.path_dim}, got {X.shape[-1]}."
        )

    dX = X if increment_input else jnp.diff(X, axis=axis_norm)
    S_orig = dX.shape[axis_norm]
    if S_orig == 0:
        raise ValueError("vsig requires at least one increment.")

    # dyadic refinement is internal only; block_size always refers to original-grid steps
    factor = 1
    if dyadic_order > 0:
        factor = 1 << int(dyadic_order)
        dX = jnp.repeat(dX / factor, factor, axis=axis_norm)
        dt_arr = jnp.asarray(dt, dtype=dX.dtype)
        dt = dt_arr / factor if dt_arr.ndim == 0 else jnp.repeat(dt_arr / factor, factor)
    if block_size is not None and S_orig % block_size != 0:
        raise ValueError(
            f"S={S_orig} must be divisible by block_size={block_size}."
        )

    # --- routing ---
    if _use_adams:
        _iter_fn = _vsig_pc
        _extra_kwargs: dict = {}
    elif _use_fft:
        _iter_fn = _vsig_fft
        _extra_kwargs = {"lag_tables": lag_tables} if lag_tables is not None else {}
    else:
        _iter_fn = _vsig_quadratic
        _extra_kwargs = {}

    if block_size is None:
        result = _iter_fn(
            dX, kernel=kernel, trunc=trunc, dt=dt,
            axis=axis_norm, order=order, **_extra_kwargs,
        )
        if output_starting_point:
            seed = starting_point if starting_point is not None else tuple(
                jnp.ones_like(result[0]) if n == 0 else jnp.zeros_like(result[n])
                for n in range(len(result))
            )
            result = tuple(
                jnp.concatenate([
                    jnp.expand_dims(seed[n], axis=axis_norm),
                    jnp.expand_dims(result[n], axis=axis_norm),
                ], axis=axis_norm)
                for n in range(len(result))
            )
        return result  # type: ignore[return-value]

    num_blocks = S_orig // block_size
    block_size_ref = block_size * factor  # refined steps per original block

    if accumulate:
        # Output convenience only: run the full scan, subsample at block boundaries.
        # accumulate=True blocking carries no efficiency gain — O(S²) either way.
        full_traj = _iter_fn(
            dX, kernel=kernel, trunc=trunc, dt=dt,
            axis=axis_norm, return_trajectory=True, order=order, **_extra_kwargs,
        )
        # full_traj[n]: [V_1, ..., V_{S_ref}] along axis_norm (0-indexed).
        # Block boundary b maps to refined index b*block_size_ref - 1.
        block_indices = jnp.arange(1, num_blocks + 1) * block_size_ref - 1
        result = tuple(
            jnp.take(full_traj[n], block_indices, axis=axis_norm)
            for n in range(trunc + 1)
        )

    else:
        # Independent blocks: split step axis, vmap over blocks.
        # O(S · block_size) — genuine compute saving vs O(S²).
        pre_batch = dX.shape[:axis_norm]
        post_batch = dX.shape[axis_norm + 1:-1]
        dX_blocked = dX.reshape(pre_batch + (num_blocks, block_size_ref) + post_batch + (dX.shape[-1],))

        if np.ndim(dt) == 0:
            dt_per_block, dt_in_axis = dt, None
        else:
            dt_per_block, dt_in_axis = jnp.reshape(dt, (num_blocks, block_size_ref)), 0

        def _call_block(dX_b, dt_b):
            return _iter_fn(dX_b, kernel=kernel, trunc=trunc, dt=dt_b,
                            axis=axis_norm, order=order, **_extra_kwargs)

        result_stacked = jax.vmap(_call_block, in_axes=(axis_norm, dt_in_axis))(
            dX_blocked, dt_per_block
        )
        # vmap stacks output on axis 0; move to axis_norm.
        result = tuple(
            jnp.moveaxis(result_stacked[n], 0, axis_norm)
            for n in range(trunc + 1)
        )

    if output_starting_point:
        seed = starting_point if starting_point is not None else tuple(
            jnp.ones_like(jnp.take(result[0], jnp.array([0]), axis=axis_norm))
            if n == 0
            else jnp.zeros_like(jnp.take(result[n], jnp.array([0]), axis=axis_norm))
            for n in range(len(result))
        )
        result = tuple(
            jnp.concatenate([seed[n], result[n]], axis=axis_norm)
            for n in range(len(result))
        )
    return result  # type: ignore[return-value]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class VolterraSignature:
    """
    Thin wrapper around :class:`ConvolutionKernel` that bundles a truncation level.

    Parameters
    ----------
    kernel:
        The underlying Volterra kernel.
    trunc:
        Tensor truncation level. Required. Static (changes cause retracing).
    """

    kernel: ConvolutionKernel
    trunc: int = field(metadata={"static": True})

    def __post_init__(self) -> None:
        if self.trunc <= 0:
            raise ValueError(f"trunc must be positive, got {self.trunc}.")

    # ------------------------------------------------------------------
    # Convenience constructors — thin wrappers around kernel constructors.
    # ------------------------------------------------------------------

    @classmethod
    def fractional(
            cls,
            *,
            trunc: int,
            **kwargs,
    ) -> "VolterraSignature":
        """Construct from a fractional kernel. Forwards all kwargs to :class:`FractionalKernel`."""
        return cls(kernel=FractionalKernel(**kwargs), trunc=trunc)

    @classmethod
    def gamma(
            cls,
            *,
            trunc: int,
            **kwargs,
    ) -> "VolterraSignature":
        """Construct from a Gamma kernel. Forwards all kwargs to :class:`GammaKernel`."""
        return cls(kernel=GammaKernel(**kwargs), trunc=trunc)

    # ------------------------------------------------------------------
    # Forwarded properties
    # ------------------------------------------------------------------

    @property
    def q(self) -> int:
        """Number of scalar kernel components."""
        return self.kernel.q

    @property
    def m(self) -> int:
        """Latent output dimension of each ``A_p``."""
        return self.kernel.m

    @property
    def path_dim(self) -> int:
        """Input path dimension ``d``."""
        return self.kernel.path_dim

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def vsig(
            self,
            X: Array,
            *,
            dt: Array | float = 1.0,
            axis: int = -2,
            block_size: Optional[int] = None,
            accumulate: bool = True,
            starting_point: Optional[DenseElem] = None,
            output_starting_point: bool = False,
            increment_input: bool = False,
            dyadic_order: int = 0,
            order: int = 0,
            scheme: Literal["auto", "fft", "quadratic", "adams"] = "auto",
    ) -> DenseElem:
        """Compute the truncated Volterra signature of ``X``.

        Thin wrapper around the module-level :func:`vsig`.  ``self.kernel``
        and ``self.trunc`` are forwarded automatically; all other arguments
        are passed through unchanged.

        Parameters
        ----------
        X:
            Path nodes or increments ``(..., S, d)``; step axis ``axis``.
        dt:
            Step size(s).  Scalar → uniform grid; 1-D array → non-uniform.
            Default ``1.0``.
        axis:
            Step axis of ``X`` (default ``-2``).
        block_size:
            Steps per emitted block.  ``None`` → single terminal result.
        accumulate:
            Carry history across blocks (``True``) or treat blocks
            independently via rebatch (``False``).
        starting_point:
            Optional seed prepended when ``output_starting_point=True``.
        output_starting_point:
            If ``True``, prepend the seed to the output along ``axis``.
        increment_input:
            Treat ``X`` as increments; skip :func:`jnp.diff`.
        dyadic_order:
            Dyadic refinement order (default ``0``).
        order:
            Quadrature order for the basis-expansion scheme (default ``0``).
        scheme:
            ``"auto"`` (default), ``"fft"``, or ``"quadratic"``.

        Returns
        -------
        DenseElem
            Volterra signature; see module-level :func:`vsig` for details.
        """
        return vsig(
            X,
            kernel=self.kernel,
            trunc=self.trunc,
            dt=dt,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            starting_point=starting_point,
            output_starting_point=output_starting_point,
            increment_input=increment_input,
            dyadic_order=dyadic_order,
            order=order,
            scheme=scheme,
        )


__all__ = ["vsig", "VolterraSignature"]
