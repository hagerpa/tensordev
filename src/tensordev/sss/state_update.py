from __future__ import annotations

import itertools as _itertools
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem, DenseElemFirstOn
from tensordev.sss.coeffs import FSSKCoefficients
from tensordev.sss.kernel import FSSK
from tensordev.sss.recursion_scalar import update_state as update_state_scalar
from tensordev.sss.recursion_general import update_state as update_state_general

Array = jax.Array

_CORE = Jax()


@partial(
    jax.jit,
    static_argnames=(
            "trunc",
            "axis",
            "block_size",
            "accumulate",
            "output_starting_state",
            "increment_input",
    ),
)
def fssk_state(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: int,
        axis: int = -2,
        block_size: int | None = None,
        accumulate: bool = True,
        initial_state: DenseElemFirstOn | None = None,
        output_starting_state: bool = False,
        increment_input: bool = False,
) -> DenseElemFirstOn:
    """
    Compute hidden FSSK recursion states from path nodes or increments.

    Projects each increment through ``kernel.A``, builds coefficients via
    ``kernel.coef``, and delegates to :func:`fssk_state_from_coef`.

    Parameters
    ----------
    X:
        Path nodes or increments. Trailing axis is the coordinate dim
        ``kernel.path_dim``; ``axis`` is the step axis.  Set
        ``increment_input=True`` to skip :func:`jnp.diff`.
    kernel:
        Finite-state-space Volterra kernel.
    dt:
        Step size(s). Accepted shapes: scalar, ``(1,)``, ``(S,)``, or
        matching batch/step axes of ``X`` without the trailing coordinate axis.
    trunc:
        Tensor truncation level (positive integer).
    axis:
        Step axis of ``X``.
    block_size:
        Steps per emitted block (``None`` = full sequence).
    accumulate:
        Carry hidden state across blocks.
    initial_state:
        Optional seed in first-on format: ``trunc`` levels where level ``r``
        has trailing shape ``(n, 1, R, m**(r+1))``.
    output_starting_state:
        Prepend seed state to the output.
    increment_input:
        Treat ``X`` as increments rather than path nodes.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")

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

    dtype = X.dtype
    dX = (X if increment_input else jnp.diff(X, axis=axis_norm)).astype(dtype)
    S = dX.shape[axis_norm]
    if S == 0:
        raise ValueError("fssk_state requires at least one increment.")

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    y = projected[..., 0, :] if kernel.q == 1 else projected

    # Optimisation: when dt is uniform across all steps (scalar input or a
    # length-1 1-D array), compute exactly *one* coefficient set and let
    # broadcast_time(S) inside fssk_state_from_coef expand it lazily.
    # For genuinely time-varying dt we fall back to the full per-step grid.
    dt_arr = jnp.asarray(dt)
    if dt_arr.ndim == 0 or (dt_arr.ndim == 1 and dt_arr.shape[0] == 1):
        dt_for_coef = dt_arr.reshape(())   # scalar → one matrix-exponential
    else:
        dt_for_coef = _normalize_dt(dt, increment_shape=dX.shape, S=S, axis_norm=axis_norm)

    coef = kernel.coef(dt_for_coef, trunc=trunc, dtype=dtype)

    return fssk_state_from_coef(
        y,
        coef=coef,
        axis=axis_norm,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
    )


@partial(
    jax.jit,
    static_argnames=(
            "axis",
            "block_size",
            "accumulate",
            "output_starting_state",
    ),
)
def fssk_state_from_coef(
        y: Array,
        *,
        coef: FSSKCoefficients,
        axis: int = 0,
        block_size: int | None = None,
        accumulate: bool = True,
        initial_state: DenseElemFirstOn | None = None,
        output_starting_state: bool = False,
) -> DenseElemFirstOn:
    """
    Compute hidden FSSK recursion states from projected increments and coefficients.

    Parameters
    ----------
    y:
        Projected increments.  For ``n==1``, trailing shape ``(m,)`` or
        ``(1, m)``; for ``n>1``, trailing shape ``(n, m)``.
        ``axis`` is the step axis.
    coef:
        FSSK coefficients.  Leading axes are broadcast / time axes.
    axis:
        Step axis of ``y``.
    block_size:
        Steps per emitted block (``None`` = full sequence).
    accumulate:
        Carry hidden state across blocks.
    initial_state:
        Optional seed in first-on format: ``trunc`` levels where level ``r``
        has trailing shape ``(n, 1, R, m**(r+1))``.
    output_starting_state:
        Prepend seed state to the output.
    """
    y = jnp.asarray(y)
    if y.ndim < 2:
        raise ValueError("y must have at least a step axis and a trailing latent dimension.")

    axis_norm = axis % y.ndim
    if axis_norm == y.ndim - 1:
        raise ValueError("axis must identify the step axis, not the trailing latent dimension.")

    dtype = y.dtype
    y_time = jnp.moveaxis(y, axis_norm, 0).astype(dtype)
    S = y_time.shape[0]
    if S == 0:
        raise ValueError("fssk_state_from_coef requires at least one increment.")

    y_time = _normalize_projected_y(y_time, coef)

    # Cast dtype — reuse methods on FSSKCoefficients.
    coef = replace(
        coef,
        E=coef.E.astype(dtype),
        psi=coef.psi.astype(dtype),
        phi=coef.phi.astype(dtype),
    )
    coef = coef.broadcast_time(S)

    B = S if block_size is None else int(block_size)
    if B <= 0:
        raise ValueError(f"block_size must be positive or None, got {block_size}.")
    n_blocks, rem = divmod(S, B)
    if rem:
        raise ValueError(
            f"block_size must divide S; got S={S}, block_size={B}."
        )

    if initial_state is not None:
        _init = tuple(initial_state)
        if not _init:
            raise ValueError("initial_state must not be empty.")
        init_batch: tuple[int, ...] = _init[0].shape[:-4]
    else:
        init_batch = ()

    batch_shape = jnp.broadcast_shapes(
        y_time.shape[1:-1] if coef.q == 1 else y_time.shape[1:-2],
        coef.leading_shape[1:],  # leading_shape[0] is the time axis
        init_batch,
    )

    seed = _make_seed(initial_state, batch_shape=batch_shape, coef=coef, dtype=dtype)

    # Reshape (S, ...) → (n_blocks, B, ...) for the outer scan.
    y_blocks = y_time.reshape((n_blocks, B) + y_time.shape[1:])
    E_blocks = coef.E.reshape((n_blocks, B) + coef.E.shape[1:])
    psi_blocks = coef.psi.reshape((n_blocks, B) + coef.psi.shape[1:])
    phi_blocks = coef.phi.reshape((n_blocks, B) + coef.phi.shape[1:])

    def block_step(carry: DenseElemFirstOn, block) -> tuple[DenseElemFirstOn, DenseElemFirstOn]:
        y_block, E_block, psi_block, phi_block = block
        Z0 = carry if accumulate else seed

        def step(Z: DenseElemFirstOn, xs) -> tuple[DenseElemFirstOn, None]:
            y_t, E_t, psi_t, phi_t = xs
            coef_t = replace(coef, E=E_t, psi=psi_t, phi=phi_t)
            return _update_state(Z, y_t, coef_t), None

        ZT, _ = lax.scan(step, Z0, (y_block, E_block, psi_block, phi_block))
        return (ZT if accumulate else carry), ZT

    _, states = lax.scan(
        block_step,
        seed,
        (y_blocks, E_blocks, psi_blocks, phi_blocks),
    )

    if output_starting_state:
        states = tuple(
            jnp.concatenate((z0[None], z), axis=0)
            for z0, z in zip(seed, states)
        )

    if not output_starting_state and n_blocks == 1:
        return tuple(z[0] for z in states)

    return tuple(jnp.moveaxis(z, 0, axis_norm) for z in states)


def _update_state(
        Z: DenseElemFirstOn,
        y: Array,
        coef: FSSKCoefficients,
) -> DenseElemFirstOn:
    """Per-step state transition dispatcher."""
    if coef.q == 1:
        return update_state_scalar(Z, y, coef, core=_CORE)
    return update_state_general(Z, y, coef, core=_CORE)


@partial(
    jax.jit,
    static_argnames=(
            "trunc",
            "axis",
            "block_size",
            "accumulate",
            "output_starting_state",
            "increment_input",
    ),
)
def fssk_vsig(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: int,
        axis: int = -2,
        block_size: int | None = None,
        accumulate: bool = True,
        initial_state: DenseElemFirstOn | None = None,
        output_starting_state: bool = False,
        tau_dt: Array | float = 0.0,
        increment_input: bool = False,
) -> DenseElem:
    """
    Compute the Volterra signature of a path via the FSSK recursion.

    Equivalent to calling :func:`fssk_state` followed by :func:`fssk_readout`.
    When ``block_size`` is ``None`` (default) and ``output_starting_state=False``,
    returns the signature at the single terminal time.  Set ``block_size=1`` and
    ``output_starting_state=True`` to obtain a full per-step signature trajectory.

    Parameters
    ----------
    X:
        Path nodes or increments. Trailing axis is the coordinate dim
        ``kernel.path_dim``; ``axis`` is the step axis.  Set
        ``increment_input=True`` to skip :func:`jnp.diff`.
    kernel:
        Finite-state-space Volterra kernel.
    dt:
        Step size(s). Accepted shapes: scalar, ``(1,)``, ``(S,)``, or
        matching batch/step axes of ``X`` without the trailing coordinate axis.
    trunc:
        Tensor truncation level (positive integer).
    axis:
        Step axis of ``X`` (default ``-2``).
    block_size:
        Steps per emitted block (``None`` = full sequence).
    accumulate:
        Carry hidden state across blocks (default ``True``).
    initial_state:
        Optional seed in first-on format: ``trunc`` levels where level ``r``
        has trailing shape ``(n, 1, R, m**(r+1))``.
    output_starting_state:
        Include the readout of the seed state (default ``False``).
    tau_dt:
        Non-negative readout lag ``tau - t``; broadcasts against batch axes.
    increment_input:
        Treat ``X`` as increments rather than path nodes.

    Returns
    -------
    DenseElem
        Volterra signature levels.  Without blocking, level ``r`` has trailing
        shape ``(m**r,)``.  With blocking, an extra block/time axis appears at
        ``axis``.
    """
    hidden = fssk_state(
        X,
        kernel=kernel,
        dt=dt,
        trunc=trunc,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        increment_input=increment_input,
    )
    return fssk_readout(hidden, kernel=kernel, tau_dt=tau_dt)


@jax.jit
def fssk_readout(
        state: DenseElemFirstOn,
        *,
        kernel: FSSK,
        tau_dt: Array | float = 0.0,
) -> DenseElem:
    """
    Read out the truncated Volterra signature from hidden FSSK states.

    If ``state`` is the recursion state at time ``t``, this evaluates the
    linear readout

        ``1 + sum_p Z^p . exp(-Lambda * (tau - t)) b_p``.

    Parameters
    ----------
    state:
        Hidden FSSK state in **first-on format**: ``trunc`` levels where
        level ``r`` (tuple index) carries degree ``r+1`` and has trailing
        shape ``(n, 1, R, m**(r+1))``.  Type: :data:`DenseElemFirstOn`.
    kernel:
        Finite-state-space Volterra kernel supplying ``Lambda`` and ``b``.
    tau_dt:
        Non-negative readout lag ``tau - t``.  Scalars and arbitrary array
        batch shapes are accepted; batch axes broadcast against the state
        leading axes.

    Returns
    -------
    DenseElem
        Truncated Volterra signature levels.  Level ``r`` has trailing shape
        ``(m**r,)`` and level zero includes the unit.
    """
    Z = tuple(state)
    if not Z:
        raise ValueError("state must not be empty.")

    dtype = Z[0].dtype
    tail_prefix = (kernel.q, 1, kernel.state_dim)
    for r, z in enumerate(Z):
        if z.shape[-4:-1] != tail_prefix or z.shape[-1] != kernel.m ** (r + 1):
            expected = tail_prefix + (kernel.m ** (r + 1),)
            raise ValueError(
                f"state[{r}] must have trailing shape {expected} "
                f"(first-on: degree {r + 1}), got {z.shape[-4:]}."
            )

    tau_dt = jnp.asarray(tau_dt, dtype=dtype)
    E = _readout_expm(kernel, tau_dt, dtype=dtype)
    weights = jnp.einsum("...rs,qs->...qr", E, kernel.b.astype(dtype))

    # Degree-0 signature level is always the unit element 1.
    batch_shape = Z[0].shape[:-4]
    out = [jnp.ones(batch_shape + (1,), dtype=dtype)]

    for z in Z:
        level = jnp.sum(
            z[..., :, 0, :, :] * weights[..., :, :, None],
            axis=(-3, -2),
        )
        out.append(level)
    return tuple(out)


def _readout_expm(
        kernel: FSSK,
        tau_dt: Array,
        *,
        dtype: jnp.dtype,
) -> Array:
    """Materialise ``exp(-Lambda * tau_dt)`` while preserving tau batch shape."""
    if tau_dt.ndim == 0:
        return kernel.Lambda.expm(tau_dt, dtype=dtype)

    tau_shape = tau_dt.shape
    E_flat = kernel.Lambda.expm(tau_dt.reshape(-1), dtype=dtype)
    return E_flat.reshape(tau_shape + E_flat.shape[-2:])


def _normalize_projected_y(y_time: Array, coef: FSSKCoefficients) -> Array:
    """Validate and normalise projected increments (time at axis 0)."""
    if coef.q == 1:
        if y_time.shape[-1] == coef.m:
            return y_time
        if y_time.ndim >= 3 and y_time.shape[-2:] == (1, coef.m):
            return y_time[..., 0, :]
        raise ValueError(
            f"For n=1, y must have trailing shape (m,) or (1, m); "
            f"expected m={coef.m}, got shape {y_time.shape}."
        )
    if y_time.ndim < 3 or y_time.shape[-2:] != (coef.q, coef.m):
        raise ValueError(
            f"For n>1, y must have trailing shape ({coef.q}, {coef.m}), "
            f"got {y_time.shape}."
        )
    return y_time


def _normalize_dt(
        dt: Array | float,
        *,
        increment_shape: tuple[int, ...],
        S: int,
        axis_norm: int,
) -> Array:
    """Normalise dt to a time-first shape matching the increment batch axes."""
    dt = jnp.asarray(dt)

    step_batch_shape = tuple(increment_shape[:-1])
    time_batch_shape = (
        (step_batch_shape[axis_norm],)
        + step_batch_shape[:axis_norm]
        + step_batch_shape[axis_norm + 1:]
    )

    if dt.ndim == 0:
        return jnp.full(time_batch_shape, dt, dtype=dt.dtype)

    if dt.ndim == 1:
        if dt.shape[0] not in (1, S):
            raise ValueError(f"1D dt must have length 1 or S={S}, got {dt.shape[0]}.")
        dt_time = jnp.broadcast_to(dt, (S,))
        return jnp.broadcast_to(
            dt_time.reshape((S,) + (1,) * (len(time_batch_shape) - 1)),
            time_batch_shape,
        )

    if dt.ndim == len(increment_shape) - 1:
        dt_time = jnp.moveaxis(dt, axis_norm, 0)
        if dt_time.shape[0] not in (1, S):
            raise ValueError(
                f"dt time length must be 1 or S={S}, got {dt_time.shape[0]}."
            )
        return jnp.broadcast_to(dt_time, time_batch_shape)

    raise ValueError(
        "dt must be scalar, shape (1,), shape (S,), or match the batch/step "
        "axes of X without the trailing path-coordinate dimension."
    )


def _make_seed(
        initial_state: DenseElemFirstOn | None,
        *,
        batch_shape: tuple[int, ...],
        coef: FSSKCoefficients,
        dtype: jnp.dtype,
) -> DenseElemFirstOn:
    # First-on format: trunc levels, index r = degree r+1.
    tails = tuple(
        (coef.q, 1, coef.R, coef.m ** (r + 1))
        for r in range(coef.trunc)
    )

    if initial_state is None:
        return tuple(jnp.zeros(batch_shape + tail, dtype=dtype) for tail in tails)

    Z = tuple(initial_state)
    if len(Z) != coef.trunc:
        raise ValueError(
            f"initial_state must have {coef.trunc} levels (first-on), got {len(Z)}."
        )

    out = []
    for r, (z, tail) in enumerate(zip(Z, tails)):
        z = jnp.asarray(z, dtype=dtype)
        if z.shape[-4:] != tail:
            raise ValueError(
                f"initial_state[{r}] must have trailing shape {tail} "
                f"(first-on: degree {r + 1}), got {z.shape[-4:]}."
            )
        out.append(jnp.broadcast_to(z, batch_shape + tail))

    return tuple(out)


__all__ = [
    "fssk_readout",
    "fssk_state",
    "fssk_state_from_coef",
    "fssk_vsig",
]
