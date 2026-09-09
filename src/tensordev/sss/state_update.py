from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from math import prod
from typing import Any

import jax
import jax.numpy as jnp

from tensordev._backend import (
    _resolve_seq_core,
    get_default_core,
    get_default_core_pair,
    get_default_seq_core,
)
from tensordev._wordwise.dispatch import _validate_execution
from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.jax import Jax
from tensordev.core.utils.pytrees import tree_first_leaf, tree_map
from tensordev.sss.coeffs import FSSKCoefficients
from tensordev.sss.kernel import FSSK
from tensordev.sss.recursion_scalar import update_state as update_state_scalar
from tensordev.sss.recursion_general import update_state as update_state_general
from tensordev.util.combinatorics import build_multiindex_layout

Array = jax.Array

_TOTAL_DEGREE_CORE = Jax()


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _PreparedFSSKStateCall:
    """Canonical state inputs shared by portable and wordwise executors."""

    y_time: Any
    coef: FSSKCoefficients
    seed: Any
    core: Any = field(metadata={"static": True})
    execution_core: Any = field(metadata={"static": True})
    seq_core: Any = field(metadata={"static": True})
    trunc: Any = field(metadata={"static": True})
    batch_shape: tuple[int, ...] = field(metadata={"static": True})
    axis: int = field(metadata={"static": True})
    block_size: int | None = field(metadata={"static": True})
    steps_per_block: int = field(metadata={"static": True})
    n_blocks: int = field(metadata={"static": True})
    accumulate: bool = field(metadata={"static": True})
    output_starting_state: bool = field(metadata={"static": True})


def _resolve_fssk_core_pair(core: Any, seq_core: Any) -> tuple[Any, Any]:
    if core is None:
        default_core, default_seq_core = get_default_core_pair()
        return default_core, default_seq_core if seq_core is None else seq_core
    if seq_core is None:
        seq_core = (
            get_default_seq_core()
            if core is get_default_core()
            else _resolve_seq_core(core, None)
        )
    return core, seq_core


def _maximum_order(trunc: Any) -> int:
    return sum(trunc) if isinstance(trunc, tuple) else int(trunc)


def _validate_fssk_core(core: Any, *, q: int, m: int, feature: str) -> None:
    grading = getattr(core, "grading", None)
    coordinates = getattr(core, "coordinates", None)
    if grading not in {"total_degree", "bidegree"}:
        raise TypeError(
            f"{feature} requires a total-degree or bidegree JAX core, got "
            f"{type(core).__name__}."
        )
    if coordinates not in {"standard", "shear"}:
        raise TypeError(
            f"{feature} requires standard or shear coordinates, got "
            f"{coordinates!r}."
        )
    if q > 1 and (grading != "total_degree" or coordinates != "standard"):
        raise RuntimeError(
            f"{feature} supports non-standard cores only for scalar FSSK "
            "kernels (kernel.q == 1)."
        )

    configured_m = (
        sum(core.dims)
        if grading == "bidegree"
        else getattr(core, "d", None)
    )
    if configured_m is not None and configured_m != m:
        raise ValueError(
            f"{feature} core alphabet dimension must equal m={m}, got "
            f"{configured_m}."
        )


@lru_cache(maxsize=128)
def _standard_execution_core(core: Any) -> Any:
    """Return a lightweight standard-coordinate view sharing core plans."""
    if getattr(core, "coordinates", None) == "standard":
        return core
    if getattr(core, "grading", None) == "total_degree":
        return Jax(
            d=core.d,
            max_trunc=core.max_truncation,
            default_trunc=core.default_truncation,
        )
    if getattr(core, "partially_symmetrized", False):
        return JaxPartiallySymmetrizedBigraded(
            plan_store=core.plan_store,
            bridge_plan_store=core.bridge_plan_store,
            shear_plan_store=core.shear_plan_store,
            default_trunc=core.default_truncation,
        )
    return JaxBigraded(
        plan_store=core.plan_store,
        default_trunc=core.default_truncation,
    )


def _truncate_scalar_coefficients(
        coef: FSSKCoefficients,
        maximum_order: int,
) -> FSSKCoefficients:
    if coef.trunc < maximum_order:
        raise ValueError(
            f"coefficients cover total order {coef.trunc}, but the active "
            f"truncation requires order {maximum_order}."
        )
    if coef.trunc == maximum_order:
        return coef
    if coef.q != 1:
        raise ValueError(
            "an explicit truncation below coef.trunc is currently supported "
            "only for scalar FSSK coefficients."
        )
    return replace(
        coef,
        layout=build_multiindex_layout(1, maximum_order - 1),
        trunc=maximum_order,
        psi=coef.psi[..., :maximum_order, :],
        phi=coef.phi[..., :, :max(maximum_order - 1, 0), :, :],
    )


def _fssk_q1_wordwise_eligible(
        *,
        core: Any,
        seq_core: Any,
        q: int,
        reference: Any,
        differentiable_inputs: Any,
        execution: str = "auto",
) -> bool:
    """Run the cheap device/transform check without loading an executor."""
    from tensordev._wordwise.dispatch import (
        fssk_q1_wordwise_device_eligible,
    )

    return fssk_q1_wordwise_device_eligible(
        core=core,
        seq_core=seq_core,
        q=q,
        reference=reference,
        differentiable_inputs=differentiable_inputs,
        execution=execution,
    )


def _fssk_accelerator_colocation_eligible(
        *,
        reference: Any,
        differentiable_inputs: Any,
) -> bool:
    from tensordev._wordwise.dispatch import (
        _eager_accelerator_colocation_eligible,
    )

    return _eager_accelerator_colocation_eligible(
        reference=reference,
        differentiable_inputs=differentiable_inputs,
    )


def _colocate_fssk_inputs(reference: Any, *values: Any) -> tuple[Any, ...]:
    """Place every eager native-path operand directly beside ``reference``."""
    from tensordev._wordwise.dispatch import _concrete_single_device

    device = _concrete_single_device(reference)
    if device is None:
        raise RuntimeError("eligible scalar-FSSK input has no concrete device.")
    return tuple(jax.device_put(values, device))


def _padded_batch_shape(
        source: tuple[int, ...],
        target: tuple[int, ...],
) -> tuple[int, ...]:
    return (1,) * (len(target) - len(source)) + tuple(source)


def _canonicalize_batch_array(
        value: Array,
        *,
        source_batch: tuple[int, ...],
        target_batch: tuple[int, ...],
        tail_ndim: int,
        has_time_axis: bool,
        force_full_batch: bool = False,
) -> Array:
    """Flatten broadcast axes while retaining wholly singleton operands."""
    prefix = (value.shape[0],) if has_time_axis else ()
    tail = value.shape[-tail_ndim:] if tail_ndim else ()
    padded = _padded_batch_shape(source_batch, target_batch)
    reshaped = value.reshape(prefix + padded + tail)
    singleton = all(extent == 1 for extent in padded)
    if singleton and not force_full_batch:
        return reshaped.reshape(prefix + (1,) + tail)

    expanded = jnp.broadcast_to(
        reshaped,
        prefix + target_batch + tail,
    )
    return expanded.reshape(prefix + (prod(target_batch),) + tail)


def _canonicalize_coefficients(
        coef: FSSKCoefficients,
        *,
        batch_shape: tuple[int, ...],
) -> FSSKCoefficients:
    coefficient_batch = tuple(coef.leading_shape[1:])
    return replace(
        coef,
        E=_canonicalize_batch_array(
            coef.E,
            source_batch=coefficient_batch,
            target_batch=batch_shape,
            tail_ndim=2,
            has_time_axis=True,
        ),
        psi=_canonicalize_batch_array(
            coef.psi,
            source_batch=coefficient_batch,
            target_batch=batch_shape,
            tail_ndim=2,
            has_time_axis=True,
        ),
        phi=_canonicalize_batch_array(
            coef.phi,
            source_batch=coefficient_batch,
            target_batch=batch_shape,
            tail_ndim=4,
            has_time_axis=True,
        ),
    )


def _state_blocks(state: Any) -> tuple[Array, ...]:
    return tuple(state.blocks) if isinstance(state, BigradedTensor) else tuple(state)


def _state_batch_shape(state: Any) -> tuple[int, ...]:
    shapes = tuple(tuple(block.shape[:-4]) for block in _state_blocks(state))
    return jnp.broadcast_shapes(*shapes) if shapes else ()


def _canonicalize_seed(
        seed: Any,
        *,
        batch_shape: tuple[int, ...],
) -> Any:
    block_batches = tuple(
        tuple(block.shape[:-4]) for block in _state_blocks(seed)
    )
    preserve_singleton = all(
        all(extent == 1 for extent in _padded_batch_shape(shape, batch_shape))
        for shape in block_batches
    )
    return tree_map(
        lambda block: _canonicalize_batch_array(
            block,
            source_batch=tuple(block.shape[:-4]),
            target_batch=batch_shape,
            tail_ndim=4,
            has_time_axis=False,
            force_full_batch=not preserve_singleton,
        ),
        seed,
    )


def _prepare_fssk_state_from_coef_call(
        y: Array,
        *,
        coef: FSSKCoefficients,
        trunc: Any,
        axis: int,
        block_size: int | None,
        accumulate: bool,
        initial_state: Any,
        output_starting_state: bool,
        core: Any,
        seq_core: Any,
) -> _PreparedFSSKStateCall:
    """Normalize one coefficient-level call for either state executor."""
    y = jnp.asarray(y)
    if y.ndim < 2:
        raise ValueError(
            "y must have at least a step axis and a trailing latent dimension."
        )

    axis_norm = axis % y.ndim
    if axis_norm == y.ndim - 1:
        raise ValueError(
            "axis must identify the step axis, not the trailing latent dimension."
        )

    dtype = y.dtype
    y_time = jnp.moveaxis(y, axis_norm, 0).astype(dtype)
    steps = y_time.shape[0]
    if steps == 0:
        raise ValueError("fssk_state_from_coef requires at least one increment.")
    y_time = _normalize_projected_y(y_time, coef)

    execution_core = (
        _standard_execution_core(core)
        if coef.q == 1
        else _TOTAL_DEGREE_CORE
    )
    if initial_state is not None:
        _validate_state(
            initial_state,
            core=core,
            trunc=trunc,
            q=coef.q,
            R=coef.R,
            m=coef.m,
            name="initial_state",
        )
        initial_state = core.tensor_to_standard_coordinates(
            initial_state,
            trunc=trunc,
            first_on=True,
        )
        initial_batch = _state_batch_shape(initial_state)
    else:
        initial_batch = ()

    coef = replace(
        coef,
        E=coef.E.astype(dtype),
        psi=coef.psi.astype(dtype),
        phi=coef.phi.astype(dtype),
    ).with_time_axis()
    coefficient_steps = coef.leading_shape[0]
    if coefficient_steps not in (1, steps):
        raise ValueError(
            "Coefficient time axis must have length 1 or S; "
            f"got {coefficient_steps} for S={steps}."
        )

    steps_per_block = steps if block_size is None else int(block_size)
    if steps_per_block <= 0:
        raise ValueError(f"block_size must be positive or None, got {block_size}.")
    n_blocks, remainder = divmod(steps, steps_per_block)
    if remainder:
        raise ValueError(
            f"block_size must divide S; got S={steps}, block_size={steps_per_block}."
        )

    y_batch = tuple(
        y_time.shape[1:-1] if coef.q == 1 else y_time.shape[1:-2]
    )
    batch_shape = jnp.broadcast_shapes(
        y_batch,
        tuple(coef.leading_shape[1:]),
        initial_batch,
    )
    if initial_state is None:
        seed = _zero_state(
            core=execution_core,
            trunc=trunc,
            q=coef.q,
            R=coef.R,
            m=coef.m,
            dtype=dtype,
        )
    else:
        seed = tree_map(lambda block: jnp.asarray(block, dtype=dtype), initial_state)

    return _PreparedFSSKStateCall(
        y_time=y_time,
        coef=coef,
        seed=seed,
        core=core,
        execution_core=execution_core,
        seq_core=seq_core,
        trunc=trunc,
        batch_shape=tuple(batch_shape),
        axis=axis_norm,
        block_size=block_size,
        steps_per_block=steps_per_block,
        n_blocks=n_blocks,
        accumulate=accumulate,
        output_starting_state=output_starting_state,
    )


def _prepare_fssk_state_call(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: Any,
        maximum_order: int,
        axis: int,
        block_size: int | None,
        accumulate: bool,
        initial_state: Any,
        output_starting_state: bool,
        increment_input: bool,
        core: Any,
        seq_core: Any,
) -> _PreparedFSSKStateCall:
    """Project a path and delegate to the shared coefficient preparation."""
    X = jnp.asarray(X)
    if X.ndim < 2:
        raise ValueError(
            "X must have at least a step axis and a trailing path dimension."
        )
    axis_norm = axis % X.ndim
    if axis_norm == X.ndim - 1:
        raise ValueError(
            "axis must identify the step axis, not the trailing path dimension."
        )
    if X.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"X trailing dimension must be {kernel.path_dim}, got {X.shape[-1]}."
        )

    dtype = X.dtype
    increments = (X if increment_input else jnp.diff(X, axis=axis_norm)).astype(dtype)
    steps = increments.shape[axis_norm]
    if steps == 0:
        raise ValueError("fssk_state requires at least one increment.")
    projected = jnp.einsum(
        "qmd,...d->...qm", kernel.A.astype(dtype), increments
    )
    y = projected[..., 0, :] if kernel.q == 1 else projected

    dt_arr = jnp.asarray(dt)
    if dt_arr.ndim == 0 or (dt_arr.ndim == 1 and dt_arr.shape[0] == 1):
        dt_for_coef = dt_arr.reshape(())
    else:
        dt_for_coef = _normalize_dt(
            dt,
            increment_shape=increments.shape,
            S=steps,
            axis_norm=axis_norm,
        )
    coef = kernel.coef(dt_for_coef, trunc=maximum_order, dtype=dtype)
    return _prepare_fssk_state_from_coef_call(
        y,
        coef=coef,
        trunc=trunc,
        axis=axis_norm,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        core=core,
        seq_core=seq_core,
    )


def _broadcast_prepared_seed(call: _PreparedFSSKStateCall) -> Any:
    return tree_map(
        lambda block: jnp.broadcast_to(
            block,
            call.batch_shape + block.shape[-4:],
        ),
        call.seed,
    )


def _run_portable_prepared_fssk_state(
        call: _PreparedFSSKStateCall,
) -> Any:
    """Run the established scan with its structured broadcast shapes."""
    steps = int(call.y_time.shape[0])
    coef = call.coef.broadcast_time(steps)
    seed = _broadcast_prepared_seed(call)

    y_blocks = call.y_time.reshape(
        (call.n_blocks, call.steps_per_block) + call.y_time.shape[1:]
    )
    E_blocks = coef.E.reshape(
        (call.n_blocks, call.steps_per_block) + coef.E.shape[1:]
    )
    psi_blocks = coef.psi.reshape(
        (call.n_blocks, call.steps_per_block) + coef.psi.shape[1:]
    )
    phi_blocks = coef.phi.reshape(
        (call.n_blocks, call.steps_per_block) + coef.phi.shape[1:]
    )

    def block_step(carry, block):
        y_block, E_block, psi_block, phi_block = block
        block_seed = carry if call.accumulate else seed

        def step(state, inputs):
            y_step, E_step, psi_step, phi_step = inputs
            coef_step = replace(
                coef,
                E=E_step,
                psi=psi_step,
                phi=phi_step,
            )
            return _update_state(
                state,
                y_step,
                coef_step,
                core=call.execution_core,
                trunc=call.trunc,
            )

        terminal = call.seq_core.tensor_reduce(
            (y_block, E_block, psi_block, phi_block),
            reduce_op=step,
            neutral=seed,
            seed=block_seed,
            axis=0,
        )
        return (terminal if call.accumulate else carry), terminal

    _, states = call.seq_core.tensor_scan(
        (y_blocks, E_blocks, psi_blocks, phi_blocks),
        initial=seed,
        scan_op=block_step,
        axis=0,
        out_axis=0,
    )
    return states


def _finalize_prepared_fssk_state(
        call: _PreparedFSSKStateCall,
        states: Any,
        *,
        flattened_batch: bool,
) -> Any:
    """Restore seed, batch, block-axis, container, and coordinates once."""
    if call.output_starting_state:
        seed = _broadcast_prepared_seed(call)
        if flattened_batch:
            flat_batch = prod(call.batch_shape)
            seed = tree_map(
                lambda block: block.reshape(
                    (flat_batch,) + block.shape[-4:]
                ),
                seed,
            )
        states = tree_map(
            lambda initial, values: jnp.concatenate(
                (initial[None], values), axis=0
            ),
            seed,
            states,
        )

    retain_block_axis = call.output_starting_state or call.n_blocks != 1
    block_count = call.n_blocks + int(call.output_starting_state)
    if flattened_batch:
        states = tree_map(
            lambda block: block.reshape(
                (block_count,) + call.batch_shape + block.shape[2:]
            ),
            states,
        )

    if retain_block_axis:
        states = tree_map(
            lambda block: jnp.moveaxis(block, 0, call.axis),
            states,
        )
    else:
        states = tree_map(lambda block: block[0], states)
    return call.core.tensor_from_standard_coordinates(
        states,
        trunc=call.trunc,
        first_on=True,
    )


@jax.jit
def _execute_portable_prepared_fssk_state(
        call: _PreparedFSSKStateCall,
) -> Any:
    """Execute and finalize a prepared FSSK call with portable JAX."""
    return _finalize_prepared_fssk_state(
        call,
        _run_portable_prepared_fssk_state(call),
        flattened_batch=False,
    )


@jax.jit
def _finalize_wordwise_prepared_fssk_state(
        call: _PreparedFSSKStateCall,
        states: Any,
) -> Any:
    return _finalize_prepared_fssk_state(
        call,
        states,
        flattened_batch=True,
    )


@jax.jit
def _execute_portable_prepared_fssk_vsig(
        call: _PreparedFSSKStateCall,
        kernel: FSSK,
        tau_dt: Array | float,
) -> Any:
    state = _finalize_prepared_fssk_state(
        call,
        _run_portable_prepared_fssk_state(call),
        flattened_batch=False,
    )
    return _fssk_readout_impl(
        state,
        kernel=kernel,
        tau_dt=tau_dt,
        core=call.core,
        trunc=call.trunc,
    )


@jax.jit
def _finalize_wordwise_prepared_fssk_vsig(
        call: _PreparedFSSKStateCall,
        states: Any,
        kernel: FSSK,
        tau_dt: Array | float,
) -> Any:
    state = _finalize_prepared_fssk_state(
        call,
        states,
        flattened_batch=True,
    )
    return _fssk_readout_impl(
        state,
        kernel=kernel,
        tau_dt=tau_dt,
        core=call.core,
        trunc=call.trunc,
    )


@jax.jit
def _finalize_wordwise_prepared_fssk_readout(
        call: _PreparedFSSKStateCall,
        standard_signature: Any,
) -> Any:
    """Restore the public batch shape and perform one coordinate conversion."""
    standard_signature = tree_map(
        lambda block: block.reshape(call.batch_shape + block.shape[1:]),
        standard_signature,
    )
    return call.core.tensor_from_standard_coordinates(
        standard_signature,
        trunc=call.trunc,
        first_on=False,
    )


@jax.jit
def _canonicalize_fssk_q1_wordwise_inputs(
        call: _PreparedFSSKStateCall,
) -> tuple[Array, FSSKCoefficients, Any]:
    """Flatten only the operands crossing the native executor boundary."""
    y_batch = tuple(call.y_time.shape[1:-1])
    y_time = _canonicalize_batch_array(
        call.y_time,
        source_batch=y_batch,
        target_batch=call.batch_shape,
        tail_ndim=1,
        has_time_axis=True,
        force_full_batch=True,
    )
    coef = _canonicalize_coefficients(
        call.coef,
        batch_shape=call.batch_shape,
    )
    seed = _canonicalize_seed(call.seed, batch_shape=call.batch_shape)
    return y_time, coef, seed


def _fssk_q1_wordwise_adapter(call: _PreparedFSSKStateCall) -> Any:
    """Build the native adapter only after eager GPU eligibility succeeds."""
    from tensordev._wordwise.fssk import PreparedFSSKQ1Call

    y_time, coef, seed = _canonicalize_fssk_q1_wordwise_inputs(call)
    return PreparedFSSKQ1Call(
        y_time=y_time,
        E_time=coef.E,
        psi_time=coef.psi,
        phi_time=coef.phi[..., 0, :, :, :],
        initial_standard=seed,
        core=call.core,
        trunc=call.trunc,
        batch_shape=call.batch_shape,
        axis=call.axis,
        block_size=call.block_size,
        accumulate=call.accumulate,
        output_starting_state=call.output_starting_state,
    )


def _try_fssk_q1_wordwise(call: _PreparedFSSKStateCall) -> Any | None:
    """Adapt shared preparation to the lazily imported state executor."""
    from tensordev._wordwise.fssk import try_fssk_q1_wordwise

    return try_fssk_q1_wordwise(_fssk_q1_wordwise_adapter(call))


def _try_fssk_q1_wordwise_readout(
        call: _PreparedFSSKStateCall,
        weights: Array,
) -> Any | None:
    """Run the fused terminal readout through the same native recurrence."""
    from tensordev._wordwise.fssk import try_fssk_q1_wordwise_readout

    return try_fssk_q1_wordwise_readout(
        _fssk_q1_wordwise_adapter(call),
        weights,
    )


def _prepared_fssk_q1_wordwise_eligible(
        call: _PreparedFSSKStateCall,
) -> bool:
    """Reject an outer JAX transform before loading plans or executors."""
    from tensordev._wordwise.dispatch import _contains_tracer

    return not _contains_tracer((call.y_time, call.coef, call.seed))


def _dispatch_fssk_q1_wordwise(
        call: _PreparedFSSKStateCall,
        *,
        execution: str,
        weights: Array | None = None,
) -> Any | None:
    """Execute an eligible call, allowing portable fallback only in auto mode."""
    if not _prepared_fssk_q1_wordwise_eligible(call):
        if execution == "wordwise":
            raise ValueError(
                "execution='wordwise' requires eager inputs outside JAX "
                "transformations (jit, grad, and vmap)."
            )
        return None
    if execution != "wordwise":
        if weights is None:
            return _try_fssk_q1_wordwise(call)
        return _try_fssk_q1_wordwise_readout(call, weights)

    from tensordev._wordwise.fssk import (
        run_fssk_q1_wordwise,
        run_fssk_q1_wordwise_readout,
    )

    adapted = _fssk_q1_wordwise_adapter(call)
    if weights is None:
        result = run_fssk_q1_wordwise(adapted)
    else:
        result = run_fssk_q1_wordwise_readout(adapted, weights)
    if result is None:
        raise RuntimeError("execution='wordwise' did not produce an FSSK result.")
    return result


@partial(jax.jit, static_argnames=("batch_shape", "dtype"))
def _canonicalize_fssk_q1_readout_weights(
        kernel: FSSK,
        tau_dt: Array | float,
        *,
        batch_shape: tuple[int, ...],
        dtype: Any,
) -> Array:
    tau_dt = jnp.asarray(tau_dt, dtype=dtype)
    E = _readout_expm(kernel, tau_dt, dtype=dtype)
    weights = jnp.einsum(
        "...rs,qs->...qr",
        E,
        kernel.b.astype(dtype),
    )[..., 0, :]
    return _canonicalize_batch_array(
        weights,
        source_batch=tuple(weights.shape[:-1]),
        target_batch=batch_shape,
        tail_ndim=1,
        has_time_axis=False,
    )


def _prepare_fssk_q1_wordwise_readout_weights(
        call: _PreparedFSSKStateCall,
        kernel: FSSK,
        tau_dt: Array | float,
) -> Array | None:
    """Return compact terminal weights, or ``None`` for a general readout."""
    if call.block_size is not None or call.output_starting_state:
        return None
    tau_shape = tuple(jnp.shape(tau_dt))
    try:
        output_batch = tuple(jnp.broadcast_shapes(call.batch_shape, tau_shape))
    except ValueError:
        return None
    if output_batch != call.batch_shape:
        return None
    return _canonicalize_fssk_q1_readout_weights(
        kernel,
        tau_dt,
        batch_shape=call.batch_shape,
        dtype=call.y_time.dtype,
    )


@partial(
    jax.jit,
    static_argnames=(
        "trunc",
        "maximum_order",
        "axis",
        "block_size",
        "accumulate",
        "output_starting_state",
        "increment_input",
        "core",
        "seq_core",
    ),
)
def _prepare_fssk_state_for_wordwise(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: Any,
        maximum_order: int,
        axis: int,
        block_size: int | None,
        accumulate: bool,
        initial_state: Any,
        output_starting_state: bool,
        increment_input: bool,
        core: Any,
        seq_core: Any,
) -> _PreparedFSSKStateCall:
    return _prepare_fssk_state_call(
        X,
        kernel=kernel,
        dt=dt,
        trunc=trunc,
        maximum_order=maximum_order,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        increment_input=increment_input,
        core=core,
        seq_core=seq_core,
    )


@partial(
    jax.jit,
    static_argnames=(
        "trunc",
        "axis",
        "block_size",
        "accumulate",
        "output_starting_state",
        "core",
        "seq_core",
    ),
)
def _prepare_fssk_state_from_coef_for_wordwise(
        y: Array,
        *,
        coef: FSSKCoefficients,
        trunc: Any,
        axis: int,
        block_size: int | None,
        accumulate: bool,
        initial_state: Any,
        output_starting_state: bool,
        core: Any,
        seq_core: Any,
) -> _PreparedFSSKStateCall:
    return _prepare_fssk_state_from_coef_call(
        y,
        coef=coef,
        trunc=trunc,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        core=core,
        seq_core=seq_core,
    )


def fssk_state(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: Any = None,
        axis: int = -2,
        block_size: int | None = None,
        accumulate: bool = True,
        initial_state: Any = None,
        output_starting_state: bool = False,
        increment_input: bool = False,
        core: Any = None,
        seq_core: Any = None,
        execution: str = "auto",
) -> Any:
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
        Positive total-degree level or nonzero bidegree rectangle. When
        omitted, the selected core's default truncation is used.
    axis:
        Step axis of ``X``.
    block_size:
        Steps per emitted block (``None`` = full sequence).
    accumulate:
        Carry hidden state across blocks.
    initial_state:
        Optional core-native seed in first-on format.
    output_starting_state:
        Prepend seed state to the output.
    increment_input:
        Treat ``X`` as increments rather than path nodes.
    core, seq_core:
        Algebra and sequential JAX cores. Both default to the configured core
        pair. Non-standard cores require ``kernel.q == 1``.
    execution:
        ``"auto"`` uses the default implementation; ``"jax"`` forces portable
        JAX. ``"wordwise"`` requests the alpha implementation for
        ``kernel.q == 1`` and raises if unsupported. It requires eager
        float32/float64 inputs on one NVIDIA GPU with CUDA compute capability
        8.0 or newer, outside ``jit``, ``grad``, and ``vmap``.
    """
    execution = _validate_execution(execution)
    core, seq_core = _resolve_fssk_core_pair(core, seq_core)
    active = core.normalize_truncation(trunc)
    _validate_fssk_core(
        core,
        q=kernel.q,
        m=kernel.m,
        feature="fssk_state",
    )
    maximum_order = _maximum_order(active)
    if maximum_order <= 0:
        raise ValueError(f"trunc must be positive, got {active}.")
    differentiable_inputs = (X, kernel, dt, initial_state)
    wordwise_eligible = _fssk_q1_wordwise_eligible(
        core=core,
        seq_core=seq_core,
        q=kernel.q,
        reference=X,
        differentiable_inputs=differentiable_inputs,
        execution=execution,
    )
    if wordwise_eligible or _fssk_accelerator_colocation_eligible(
        reference=X,
        differentiable_inputs=differentiable_inputs,
    ):
        kernel, dt, initial_state = _colocate_fssk_inputs(
            X,
            kernel,
            dt,
            initial_state,
        )
    if wordwise_eligible:
        call = _prepare_fssk_state_for_wordwise(
            X,
            kernel=kernel,
            dt=dt,
            trunc=active,
            maximum_order=maximum_order,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial_state,
            output_starting_state=output_starting_state,
            increment_input=increment_input,
            core=core,
            seq_core=seq_core,
        )
        states = _dispatch_fssk_q1_wordwise(call, execution=execution)
        if states is None:
            return _execute_portable_prepared_fssk_state(call)
        return _finalize_wordwise_prepared_fssk_state(call, states)
    return _fssk_state_impl(
        X,
        kernel=kernel,
        dt=dt,
        trunc=active,
        maximum_order=maximum_order,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        increment_input=increment_input,
        core=core,
        seq_core=seq_core,
    )


@partial(
    jax.jit,
    static_argnames=(
        "trunc",
        "maximum_order",
        "axis",
        "block_size",
        "accumulate",
        "output_starting_state",
        "increment_input",
        "core",
        "seq_core",
    ),
)
def _fssk_state_impl(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: Any,
        maximum_order: int,
        axis: int,
        block_size: int | None,
        accumulate: bool,
        initial_state: Any,
        output_starting_state: bool,
        increment_input: bool,
        core: Any,
        seq_core: Any,
) -> Any:
    call = _prepare_fssk_state_call(
        X,
        kernel=kernel,
        dt=dt,
        trunc=trunc,
        maximum_order=maximum_order,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        increment_input=increment_input,
        core=core,
        seq_core=seq_core,
    )
    return _finalize_prepared_fssk_state(
        call,
        _run_portable_prepared_fssk_state(call),
        flattened_batch=False,
    )


def fssk_state_from_coef(
        y: Array,
        *,
        coef: FSSKCoefficients,
        trunc: Any = None,
        axis: int = 0,
        block_size: int | None = None,
        accumulate: bool = True,
        initial_state: Any = None,
        output_starting_state: bool = False,
        core: Any = None,
        seq_core: Any = None,
        execution: str = "auto",
) -> Any:
    """
    Compute hidden FSSK recursion states from projected increments and coefficients.

    Parameters
    ----------
    y:
        Projected increments.  For ``q==1``, trailing shape ``(m,)`` or
        ``(1, m)``; for ``q>1``, trailing shape ``(q, m)``.
        ``axis`` is the step axis.
    coef:
        FSSK coefficients.  Leading axes are broadcast / time axes.
    trunc:
        Active total-degree level or bidegree rectangle. When omitted, a
        total-degree call uses ``coef.trunc`` and a bidegree call uses the core
        default.
    axis:
        Step axis of ``y``.
    block_size:
        Steps per emitted block (``None`` = full sequence).
    accumulate:
        Carry hidden state across blocks.
    initial_state:
        Optional core-native seed in first-on format.
    output_starting_state:
        Prepend seed state to the output.
    core, seq_core:
        Algebra and sequential JAX cores. Both default to the configured pair.
    execution:
        ``"auto"``, ``"jax"``, or ``"wordwise"``; see :func:`fssk_state`.
    """
    execution = _validate_execution(execution)
    core, seq_core = _resolve_fssk_core_pair(core, seq_core)
    if trunc is None and getattr(core, "grading", None) == "total_degree":
        trunc = coef.trunc
    active = core.normalize_truncation(trunc)
    _validate_fssk_core(
        core,
        q=coef.q,
        m=coef.m,
        feature="fssk_state_from_coef",
    )
    maximum_order = _maximum_order(active)
    if maximum_order <= 0:
        raise ValueError(f"trunc must be positive, got {active}.")
    differentiable_inputs = (y, coef, initial_state)
    wordwise_eligible = _fssk_q1_wordwise_eligible(
        core=core,
        seq_core=seq_core,
        q=coef.q,
        reference=y,
        differentiable_inputs=differentiable_inputs,
        execution=execution,
    )
    if wordwise_eligible or _fssk_accelerator_colocation_eligible(
        reference=y,
        differentiable_inputs=differentiable_inputs,
    ):
        coef, initial_state = _colocate_fssk_inputs(
            y,
            coef,
            initial_state,
        )
    coef = _truncate_scalar_coefficients(coef, maximum_order)
    if wordwise_eligible:
        call = _prepare_fssk_state_from_coef_for_wordwise(
            y,
            coef=coef,
            trunc=active,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial_state,
            output_starting_state=output_starting_state,
            core=core,
            seq_core=seq_core,
        )
        states = _dispatch_fssk_q1_wordwise(call, execution=execution)
        if states is None:
            return _execute_portable_prepared_fssk_state(call)
        return _finalize_wordwise_prepared_fssk_state(call, states)
    return _fssk_state_from_coef_impl(
        y,
        coef=coef,
        trunc=active,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        core=core,
        seq_core=seq_core,
    )


@partial(
    jax.jit,
    static_argnames=(
        "trunc",
        "axis",
        "block_size",
        "accumulate",
        "output_starting_state",
        "core",
        "seq_core",
    ),
)
def _fssk_state_from_coef_impl(
        y: Array,
        *,
        coef: FSSKCoefficients,
        trunc: Any,
        axis: int,
        block_size: int | None,
        accumulate: bool,
        initial_state: Any,
        output_starting_state: bool,
        core: Any,
        seq_core: Any,
) -> Any:
    call = _prepare_fssk_state_from_coef_call(
        y,
        coef=coef,
        trunc=trunc,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        core=core,
        seq_core=seq_core,
    )
    return _finalize_prepared_fssk_state(
        call,
        _run_portable_prepared_fssk_state(call),
        flattened_batch=False,
    )


def _update_state(
        Z: Any,
        y: Array,
        coef: FSSKCoefficients,
        *,
        core: Any,
        trunc: Any,
) -> Any:
    """Per-step state transition dispatcher."""
    if coef.q == 1:
        return update_state_scalar(Z, y, coef, core=core, trunc=trunc)
    return update_state_general(Z, y, coef, core=core)


def fssk_vsig(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: Any = None,
        axis: int = -2,
        block_size: int | None = None,
        accumulate: bool = True,
        initial_state: Any = None,
        output_starting_state: bool = False,
        tau_dt: Array | float = 0.0,
        increment_input: bool = False,
        core: Any = None,
        seq_core: Any = None,
        execution: str = "auto",
) -> Any:
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
        Positive total-degree level or nonzero bidegree rectangle. When
        omitted, the selected core's default truncation is used.
    axis:
        Step axis of ``X`` (default ``-2``).
    block_size:
        Steps per emitted block (``None`` = full sequence).
    accumulate:
        Carry hidden state across blocks (default ``True``).
    initial_state:
        Optional core-native seed in first-on format.
    output_starting_state:
        Include the readout of the seed state (default ``False``).
    tau_dt:
        Non-negative readout lag ``tau - t``; broadcasts against batch axes.
    increment_input:
        Treat ``X`` as increments rather than path nodes.
    core, seq_core:
        Algebra and sequential JAX cores. Both default to the configured pair.
        Non-standard cores require ``kernel.q == 1``.
    execution:
        ``"auto"``, ``"jax"``, or ``"wordwise"``; see :func:`fssk_state`.

    Returns
    -------
    tuple or BigradedTensor
        Volterra signature in the selected core's native layout and
        coordinates. With blocking, an emitted block axis appears at ``axis``.
    """
    execution = _validate_execution(execution)
    core, seq_core = _resolve_fssk_core_pair(core, seq_core)
    active = core.normalize_truncation(trunc)
    _validate_fssk_core(
        core,
        q=kernel.q,
        m=kernel.m,
        feature="fssk_vsig",
    )
    maximum_order = _maximum_order(active)
    if maximum_order <= 0:
        raise ValueError(f"trunc must be positive, got {active}.")
    differentiable_inputs = (
        X,
        kernel,
        dt,
        initial_state,
        tau_dt,
    )
    wordwise_eligible = _fssk_q1_wordwise_eligible(
        core=core,
        seq_core=seq_core,
        q=kernel.q,
        reference=X,
        differentiable_inputs=differentiable_inputs,
        execution=execution,
    )
    if wordwise_eligible or _fssk_accelerator_colocation_eligible(
        reference=X,
        differentiable_inputs=differentiable_inputs,
    ):
        kernel, dt, initial_state, tau_dt = _colocate_fssk_inputs(
            X,
            kernel,
            dt,
            initial_state,
            tau_dt,
        )
    if wordwise_eligible:
        call = _prepare_fssk_state_for_wordwise(
            X,
            kernel=kernel,
            dt=dt,
            trunc=active,
            maximum_order=maximum_order,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial_state,
            output_starting_state=output_starting_state,
            increment_input=increment_input,
            core=core,
            seq_core=seq_core,
        )
        weights = _prepare_fssk_q1_wordwise_readout_weights(
            call,
            kernel,
            tau_dt,
        )
        if weights is not None:
            signature = _dispatch_fssk_q1_wordwise(
                call, execution=execution, weights=weights
            )
            if signature is None:
                return _execute_portable_prepared_fssk_vsig(
                    call,
                    kernel,
                    tau_dt,
                )
            return _finalize_wordwise_prepared_fssk_readout(
                call,
                signature,
            )
        states = _dispatch_fssk_q1_wordwise(call, execution=execution)
        if states is None:
            return _execute_portable_prepared_fssk_vsig(
                call,
                kernel,
                tau_dt,
            )
        return _finalize_wordwise_prepared_fssk_vsig(
            call,
            states,
            kernel,
            tau_dt,
        )
    return _fssk_vsig_impl(
        X,
        kernel=kernel,
        dt=dt,
        trunc=active,
        maximum_order=maximum_order,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        tau_dt=tau_dt,
        increment_input=increment_input,
        core=core,
        seq_core=seq_core,
    )


@partial(
    jax.jit,
    static_argnames=(
        "trunc",
        "maximum_order",
        "axis",
        "block_size",
        "accumulate",
        "output_starting_state",
        "increment_input",
        "core",
        "seq_core",
    ),
)
def _fssk_vsig_impl(
        X: Array,
        *,
        kernel: FSSK,
        dt: Array | float,
        trunc: Any,
        maximum_order: int,
        axis: int,
        block_size: int | None,
        accumulate: bool,
        initial_state: Any,
        output_starting_state: bool,
        tau_dt: Array | float,
        increment_input: bool,
        core: Any,
        seq_core: Any,
) -> Any:
    hidden = _fssk_state_impl(
        X,
        kernel=kernel,
        dt=dt,
        trunc=trunc,
        maximum_order=maximum_order,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        increment_input=increment_input,
        core=core,
        seq_core=seq_core,
    )
    return _fssk_readout_impl(
        hidden,
        kernel=kernel,
        tau_dt=tau_dt,
        core=core,
        trunc=trunc,
    )


def fssk_readout(
        state: Any,
        *,
        kernel: FSSK,
        tau_dt: Array | float = 0.0,
        core: Any = None,
) -> Any:
    """
    Read out the truncated Volterra signature from hidden FSSK states.

    If ``state`` is the recursion state at time ``t``, this evaluates the
    linear readout

        ``1 + sum_p Z^p . exp(-Lambda * (tau - t)) b_p``.

    Parameters
    ----------
    state:
        Hidden FSSK state in first-on format. Total-degree states are tuples;
        bidegree states are scalar-omitting ``BigradedTensor`` objects.
    kernel:
        Finite-state-space Volterra kernel supplying ``Lambda`` and ``b``.
    tau_dt:
        Non-negative readout lag ``tau - t``.  Scalars and arbitrary array
        batch shapes are accepted; batch axes broadcast against the state
        leading axes.
    core:
        Core describing the state's layout and coordinates. With ``None``, a
        tuple state is interpreted as standard total degree without consulting
        the configured default core. A bidegree state requires its core
        explicitly.

    Returns
    -------
    tuple or BigradedTensor
        Truncated Volterra signature in the supplied core's native layout and
        coordinates, including the scalar unit.
    """
    if core is None:
        if isinstance(state, BigradedTensor):
            raise TypeError("a bidegree state requires core= for fssk_readout.")
        core = _TOTAL_DEGREE_CORE

    trunc = (
        state.truncation
        if isinstance(state, BigradedTensor)
        else len(tuple(state))
    )
    trunc = core.normalize_truncation(trunc)
    _validate_fssk_core(
        core,
        q=kernel.q,
        m=kernel.m,
        feature="fssk_readout",
    )
    if len(state) > 0:
        reference = tree_first_leaf(state)
        if _fssk_accelerator_colocation_eligible(
            reference=reference,
            differentiable_inputs=(state, kernel, tau_dt),
        ):
            state, kernel, tau_dt = _colocate_fssk_inputs(
                reference,
                state,
                kernel,
                tau_dt,
            )
    return _fssk_readout_impl(
        state,
        kernel=kernel,
        tau_dt=tau_dt,
        core=core,
        trunc=trunc,
    )


@partial(jax.jit, static_argnames=("core", "trunc"))
def _fssk_readout_impl(
        state: Any,
        *,
        kernel: FSSK,
        tau_dt: Array | float,
        core: Any,
        trunc: Any,
) -> Any:
    if len(state) == 0:
        raise ValueError("state must not be empty.")

    _validate_state(
        state,
        core=core,
        trunc=trunc,
        q=kernel.q,
        R=kernel.state_dim,
        m=kernel.m,
        name="state",
    )
    first = tree_first_leaf(state)
    dtype = first.dtype

    tau_dt = jnp.asarray(tau_dt, dtype=dtype)
    E = _readout_expm(kernel, tau_dt, dtype=dtype)
    weights = jnp.einsum("...rs,qs->...qr", E, kernel.b.astype(dtype))

    positive = tree_map(
        lambda z: jnp.sum(
            z[..., :, 0, :, :] * weights[..., :, :, None],
            axis=(-3, -2),
        ),
        state,
    )
    positive_first = tree_first_leaf(positive)
    unit = jnp.ones(positive_first.shape[:-1] + (1,), dtype=dtype)
    if isinstance(positive, BigradedTensor):
        return BigradedTensor(
            (unit,) + positive.blocks,
            positive.spec.with_scalar(True),
        )
    return (unit,) + tuple(positive)


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
            f"For q=1, y must have trailing shape (m,) or (1, m); "
            f"expected m={coef.m}, got shape {y_time.shape}."
        )
    if y_time.ndim < 3 or y_time.shape[-2:] != (coef.q, coef.m):
        raise ValueError(
            f"For q>1, y must have trailing shape ({coef.q}, {coef.m}), "
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


def _zero_state(
        *,
        core: Any,
        trunc: Any,
        q: int,
        R: int,
        m: int,
        batch_shape: tuple[int, ...] = (),
        dtype: Any = float,
) -> Any:
    if getattr(core, "grading", None) == "bidegree":
        layout = core.resolve_layout(trunc, include_scalar=False)
        return BigradedTensor(
            tuple(
                jnp.zeros(
                    batch_shape + (q, 1, R, layout.block_width(grade)),
                    dtype=dtype,
                )
                for grade in layout.grades
            ),
            layout.spec,
        )
    return tuple(
        jnp.zeros(batch_shape + (q, 1, R, m ** degree), dtype=dtype)
        for degree in range(1, _maximum_order(trunc) + 1)
    )


def _validate_state(
        state: Any,
        *,
        core: Any,
        trunc: Any,
        q: int,
        R: int,
        m: int,
        name: str,
) -> None:
    tail_prefix = (q, 1, R)
    if getattr(core, "grading", None) == "bidegree":
        if not isinstance(state, BigradedTensor):
            raise TypeError(
                f"{name} must be a BigradedTensor for a bidegree core, got "
                f"{type(state).__name__}."
            )
        spec = state.spec
        expected_partial = bool(getattr(core, "partially_symmetrized", False))
        if spec.dims != core.dims:
            raise ValueError(
                f"{name} uses alphabet dimensions {spec.dims}, expected "
                f"{core.dims}."
            )
        if spec.truncation != trunc:
            raise ValueError(
                f"{name} uses truncation {spec.truncation}, expected {trunc}."
            )
        if spec.coordinates != core.coordinates:
            raise ValueError(
                f"{name} uses coordinates {spec.coordinates!r}, expected "
                f"{core.coordinates!r}."
            )
        if spec.partially_symmetrized != expected_partial:
            raise ValueError(
                f"{name} has partially_symmetrized="
                f"{spec.partially_symmetrized!r}, expected {expected_partial!r}."
            )
        if spec.include_scalar:
            raise ValueError(f"{name} must use first-on format without a scalar block.")
        layout = core.resolve_layout(trunc, include_scalar=False)
        if len(state.blocks) != len(layout.grades):
            raise ValueError(
                f"{name} must have {len(layout.grades)} bidegree blocks, got "
                f"{len(state.blocks)}."
            )
        for grade, block in zip(state.grades, state.blocks):
            if block.shape[-4:-1] != tail_prefix:
                raise ValueError(
                    f"{name}[{grade}] must have trailing state axes "
                    f"{tail_prefix}, got {block.shape[-4:-1]}."
                )
            expected_width = layout.block_width(grade)
            if block.shape[-1] != expected_width:
                raise ValueError(
                    f"{name}[{grade}] must have coordinate width "
                    f"{expected_width}, got {block.shape[-1]}."
                )
        return

    levels = tuple(state)
    expected_levels = _maximum_order(trunc)
    if len(levels) != expected_levels:
        raise ValueError(
            f"{name} must have {expected_levels} levels (first-on), got "
            f"{len(levels)}."
        )
    for degree, level in enumerate(levels, start=1):
        expected = tail_prefix + (m ** degree,)
        if level.shape[-4:] != expected:
            raise ValueError(
                f"{name}[{degree - 1}] must have trailing shape {expected} "
                f"(first-on: degree {degree}), got {level.shape[-4:]}."
            )


__all__ = [
    "fssk_readout",
    "fssk_state",
    "fssk_state_from_coef",
    "fssk_vsig",
]
