"""Private eligibility checks for specialized wordwise execution.

The checks in this module are deliberately cheap and precede every host-plan
lookup.  In particular, ordinary CPU calls must not allocate wordwise metadata
or import Pallas merely because the specialized implementation is installed.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from tensordev.core.capabilities import _WORDWISE_SIGNATURE_PROTOCOL


_SUPPORTED_DTYPES = frozenset((np.dtype("float32"), np.dtype("float64")))


def _validate_execution(execution: str) -> str:
    """Validate the execution policy shared by signature entry points."""
    if not isinstance(execution, str) or execution not in (
        "auto", "jax", "wordwise"
    ):
        raise ValueError("execution must be 'auto', 'jax', or 'wordwise'.")
    return execution


def _select_wordwise_execution(
        execution: str,
        unsupported_reason: Callable[..., str | None],
        *args: Any,
        **kwargs: Any,
) -> bool:
    """Apply selection policy without checking candidates on portable calls."""
    _validate_execution(execution)
    if execution == "jax" or (
        execution == "auto" and not _automatic_wordwise_release_eligible()
    ):
        return False
    reason = unsupported_reason(*args, **kwargs)
    if reason is not None and execution == "wordwise":
        raise ValueError(
            f"execution='wordwise' {reason} "
            "Use execution='jax' or 'auto' for portable execution."
        )
    return reason is None


def _concrete_single_device(value: Any) -> Any | None:
    """Return the sole device of a concrete single-device array, if any."""
    if isinstance(value, jax.core.Tracer):
        return None
    devices_method = getattr(value, "devices", None)
    if devices_method is not None:
        try:
            devices = tuple(devices_method())
        except (AttributeError, TypeError):
            pass
        else:
            return devices[0] if len(devices) == 1 else None
    device = getattr(value, "device", None)
    if callable(device):
        device = device()
    return device


def _colocate_array(value: Any, reference: Any) -> Any:
    """Place host metadata beside a concrete single-device input when known.

    During tracing the metadata remains an ordinary JAX constant.  For eager
    calls this prevents a CPU process default from leaving decoder operands on
    the host when only the numerical input was explicitly placed on a GPU.
    """
    device = _concrete_single_device(reference)
    return (
        jnp.asarray(value)
        if device is None
        else jax.device_put(value, device)
    )


def _colocate_pytree(value: Any, reference: Any) -> Any:
    """Place an eager numerical PyTree beside ``reference`` when necessary."""
    device = _concrete_single_device(reference)
    if device is None or value is None or _contains_tracer(value):
        return value
    leaves = jax.tree.leaves(value)
    if all(_concrete_single_device(leaf) == device for leaf in leaves):
        return value
    return jax.device_put(value, device)


def _normalized_compute_capability(device: Any) -> tuple[int, int] | None:
    """Normalize the CUDA compute-capability forms exposed by JAX clients."""
    capability = getattr(device, "compute_capability", None)
    if callable(capability):
        capability = capability()
    if capability is None:
        return None
    if isinstance(capability, str):
        pieces = capability.strip().split(".", 1)
        if not all(piece.isdigit() for piece in pieces):
            return None
        return int(pieces[0]), int(pieces[1]) if len(pieces) == 2 else 0
    if (
        isinstance(capability, tuple)
        and len(capability) == 2
        and all(isinstance(value, Integral) for value in capability)
    ):
        return int(capability[0]), int(capability[1])
    if isinstance(capability, Integral) and not isinstance(capability, bool):
        capability = int(capability)
        return divmod(capability, 10)
    return None


def _supported_cuda_device(device: Any) -> bool:
    """Whether ``device`` satisfies the wordwise CUDA requirements."""
    if getattr(device, "platform", None) not in {"cuda", "gpu"}:
        return False
    kind = str(getattr(device, "device_kind", "")).lower()
    if kind and "nvidia" not in kind:
        return False
    capability = _normalized_compute_capability(device)
    return capability is not None and capability >= (8, 0)


def _automatic_wordwise_release_eligible() -> bool:
    """Whether automatic wordwise selection is enabled.

    Explicit ``execution="wordwise"`` is independent of automatic selection
    and retains all input and resource checks.
    """
    return False


def _eager_accelerator_colocation_eligible(
        reference: Any,
        differentiable_inputs: Any,
) -> bool:
    """Whether eager operands should follow a concrete accelerator input."""
    device = _concrete_single_device(reference)
    platform = getattr(device, "platform", None)
    if platform is None or platform == "cpu":
        return False
    return not _contains_tracer(differentiable_inputs)


def _wordwise_input_unsupported_reason(
        *,
        core: Any,
        seq_core: Any,
        reference: Any,
        differentiable_inputs: Any,
) -> str | None:
    """Check the core, transform, dtype, and device contract before planning."""
    if _contains_tracer((reference, differentiable_inputs)):
        return "does not support calls under jax.jit, jax.grad, or jax.vmap."
    if (
        type(core).__dict__.get("_wordwise_signature_protocol")
        is not _WORDWISE_SIGNATURE_PROTOCOL
    ):
        return "requires a supported JAX tensor core."
    if (
        type(seq_core).__dict__.get("_wordwise_signature_protocol")
        is not _WORDWISE_SIGNATURE_PROTOCOL
    ):
        return "requires a supported JAX sequential core."
    grading = getattr(core, "grading", None)
    if grading not in {"total_degree", "bidegree"}:
        return "requires total-degree or bidegree truncation."
    if grading == "bidegree" and getattr(core, "plan_store", None) is None:
        return "requires a bounded bidegree core with precomputed plans."
    try:
        dtype = np.dtype(reference.dtype)
    except (AttributeError, TypeError):
        return "requires float32 or float64 input."
    if dtype not in _SUPPORTED_DTYPES:
        return "requires float32 or float64 input."
    device = _concrete_single_device(reference)
    if device is None or not _supported_cuda_device(device):
        return (
            "requires input on a single NVIDIA GPU with CUDA compute "
            "capability 8.0 or newer."
        )
    return None


def _ordinary_wordwise_unsupported_reason(call: Any) -> str | None:
    if call.parallel or call.accumulate_in_tree:
        return "requires parallel=False and accumulate_in_tree=False."
    if len(call.increments) != 1:
        return "requires a single level-1 path."
    return _wordwise_input_unsupported_reason(
        core=call.core,
        seq_core=call.seq_core,
        reference=call.increments[0],
        differentiable_inputs=(
            call.increments, call.neutral, call.seed_policy.canonical_start
        ),
    )


def ordinary_wordwise_candidate_eligible(call: Any) -> bool:
    """Whether an ordinary call satisfies the wordwise execution contract."""
    return _ordinary_wordwise_unsupported_reason(call) is None


def ordinary_wordwise_device_eligible(
        call: Any, *, execution: str = "auto"
) -> bool:
    """Select ordinary wordwise execution, rejecting unsupported explicit use."""
    return _select_wordwise_execution(
        execution, _ordinary_wordwise_unsupported_reason, call
    )


def _contains_tracer(value: Any) -> bool:
    """Whether any differentiable leaf is currently under a JAX transform."""
    try:
        leaves = jax.tree.leaves(value)
    except (TypeError, ValueError):
        leaves = (value,)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _fssk_q1_wordwise_unsupported_reason(
        *,
        core: Any,
        seq_core: Any,
        q: int,
        reference: Any,
        differentiable_inputs: Any,
) -> str | None:
    if q != 1:
        return "requires an FSSK kernel or coefficients with q=1."
    return _wordwise_input_unsupported_reason(
        core=core,
        seq_core=seq_core,
        reference=reference,
        differentiable_inputs=differentiable_inputs,
    )


def fssk_q1_wordwise_candidate_eligible(
        *,
        core: Any,
        seq_core: Any,
        q: int,
        reference: Any,
        differentiable_inputs: Any,
) -> bool:
    """Whether a scalar-FSSK call satisfies the wordwise execution contract.

    This check intentionally consumes only the unresolved public inputs.  It
    therefore runs before coefficient construction, layout planning, or any
    Pallas import. Transformations with respect to paths, kernels,
    coefficients, seeds, or readout inputs are excluded.
    """
    return _fssk_q1_wordwise_unsupported_reason(
        core=core,
        seq_core=seq_core,
        q=q,
        reference=reference,
        differentiable_inputs=differentiable_inputs,
    ) is None


def fssk_q1_wordwise_device_eligible(
        *,
        core: Any,
        seq_core: Any,
        q: int,
        reference: Any,
        differentiable_inputs: Any,
        execution: str = "auto",
) -> bool:
    """Select scalar-FSSK wordwise execution under the shared policy."""
    return _select_wordwise_execution(
        execution,
        _fssk_q1_wordwise_unsupported_reason,
        core=core,
        seq_core=seq_core,
        q=q,
        reference=reference,
        differentiable_inputs=differentiable_inputs,
    )


__all__ = [
    "fssk_q1_wordwise_candidate_eligible",
    "fssk_q1_wordwise_device_eligible",
    "ordinary_wordwise_candidate_eligible",
    "ordinary_wordwise_device_eligible",
]
