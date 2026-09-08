"""Private eligibility checks for specialized wordwise execution.

The checks in this module are deliberately cheap and precede every host-plan
lookup.  In particular, ordinary CPU calls must not allocate wordwise metadata
or import Pallas merely because the specialized implementation is installed.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tensordev.core.capabilities import _WORDWISE_SIGNATURE_PROTOCOL


_SUPPORTED_DTYPES = frozenset((np.dtype("float32"), np.dtype("float64")))


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
    """Whether ``device`` can be considered for native validation."""
    if getattr(device, "platform", None) not in {"cuda", "gpu"}:
        return False
    kind = str(getattr(device, "device_kind", "")).lower()
    if kind and "nvidia" not in kind:
        return False
    capability = _normalized_compute_capability(device)
    return capability is not None and capability >= (8, 0)


def _automatic_wordwise_release_eligible() -> bool:
    """Whether benchmark-derived automatic-dispatch regions are installed.

    Automatic selection remains closed until real-GPU correctness and paired
    performance results identify exact software, architecture, and workload
    regions.  Private validation and benchmark runners bypass this gate while
    retaining every structural and resource check in their executor layers.
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


def ordinary_wordwise_candidate_eligible(call: Any) -> bool:
    """Return whether an ordinary call is a native-validation candidate.

    This is intentionally conservative.  Traced calls, sharded arrays, tree
    scans, non-JAX cores, and targets whose CUDA capability cannot be inspected
    are excluded before planning or lowering.
    """
    if call.parallel or call.accumulate_in_tree:
        return False
    if _contains_tracer(
        (
            call.increments,
            call.neutral,
            call.seed_policy.canonical_start,
        )
    ):
        return False
    if (
        type(call.core).__dict__.get("_wordwise_signature_protocol")
        is not _WORDWISE_SIGNATURE_PROTOCOL
    ):
        return False
    if (
        type(call.seq_core).__dict__.get("_wordwise_signature_protocol")
        is not _WORDWISE_SIGNATURE_PROTOCOL
    ):
        return False
    grading = getattr(call.core, "grading", None)
    if grading not in {"total_degree", "bidegree"}:
        return False
    if grading == "bidegree" and getattr(call.core, "plan_store", None) is None:
        return False
    if len(call.increments) != 1:
        return False
    increment = call.increments[0]
    if np.dtype(increment.dtype) not in _SUPPORTED_DTYPES:
        return False
    return _supported_cuda_device(_concrete_single_device(increment))


def ordinary_wordwise_device_eligible(call: Any) -> bool:
    """Whether a prepared ordinary signature may use automatic native code."""
    return (
        _automatic_wordwise_release_eligible()
        and ordinary_wordwise_candidate_eligible(call)
    )


def _contains_tracer(value: Any) -> bool:
    """Whether any differentiable leaf is currently under a JAX transform."""
    try:
        leaves = jax.tree.leaves(value)
    except (TypeError, ValueError):
        leaves = (value,)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _fssk_q1_candidate_eligible(
        *,
        core: Any,
        q: int,
        reference: Any,
        differentiable_inputs: Any,
) -> bool:
    if q != 1 or _contains_tracer(differentiable_inputs):
        return False
    if (
        type(core).__dict__.get("_wordwise_signature_protocol")
        is not _WORDWISE_SIGNATURE_PROTOCOL
    ):
        return False
    grading = getattr(core, "grading", None)
    if grading not in {"total_degree", "bidegree"}:
        return False
    if grading == "bidegree" and getattr(core, "plan_store", None) is None:
        return False
    try:
        dtype = np.dtype(reference.dtype)
    except (AttributeError, TypeError):
        return False
    if dtype not in _SUPPORTED_DTYPES:
        return False
    return _supported_cuda_device(_concrete_single_device(reference))


def fssk_q1_wordwise_candidate_eligible(
        *,
        core: Any,
        seq_core: Any,
        q: int,
        reference: Any,
        differentiable_inputs: Any,
) -> bool:
    """Return whether a scalar-FSSK call is a native-validation candidate.

    This check intentionally consumes only the unresolved public inputs.  It
    therefore runs before coefficient construction, layout planning, or any
    Pallas import.  Every transformed call remains on the established portable
    implementation, including transformations with respect to kernel,
    coefficient, seed, or readout inputs rather than the path itself.
    """
    if not _fssk_q1_candidate_eligible(
        core=core,
        q=q,
        reference=reference,
        differentiable_inputs=differentiable_inputs,
    ):
        return False
    if (
        type(seq_core).__dict__.get("_wordwise_signature_protocol")
        is not _WORDWISE_SIGNATURE_PROTOCOL
    ):
        return False
    grading = getattr(core, "grading", None)
    return not (
        grading == "bidegree" and getattr(core, "plan_store", None) is None
    )


def fssk_q1_wordwise_device_eligible(
        *,
        core: Any,
        seq_core: Any,
        q: int,
        reference: Any,
        differentiable_inputs: Any,
) -> bool:
    """Whether an eager scalar-FSSK call may use automatic native code."""
    return (
        _automatic_wordwise_release_eligible()
        and fssk_q1_wordwise_candidate_eligible(
            core=core,
            seq_core=seq_core,
            q=q,
            reference=reference,
            differentiable_inputs=differentiable_inputs,
        )
    )


__all__ = [
    "fssk_q1_wordwise_candidate_eligible",
    "fssk_q1_wordwise_device_eligible",
    "ordinary_wordwise_candidate_eligible",
    "ordinary_wordwise_device_eligible",
]
