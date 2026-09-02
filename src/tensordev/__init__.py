"""Public package exports."""

from typing import Literal

from .core import (
    BigradedSpec,
    BigradedTensor,
    Jax,
    JaxBigraded,
    JaxPartiallySymmetrizedBigraded,
    JaxPartiallySymmetrizedShearBigraded,
    JaxSequentialCore,
    bigraded_core,
    shear_core,
    symmetrized_core,
    total_degree_core,
)
from .development import FreeDevelopment, free_development, Signature, path_signature
from ._backend import (
    _resolve_core_configuration,
    get_default_core,
    get_default_core_pair,
    get_default_seq_core,
    register_default_core_callback as _register_default_core_callback,
    reset_default_core,
    set_default_core,
)

# --- volterra ---
from .volterra import (
    vsig,
    VolterraSignature,
    FractionalKernel,
    GammaKernel,
    FSSKConvolutionKernel,
)

# --- sss ---
from .sss import (
    fssk_vsig,
    fssk_state,
    StateSpaceSignature,
    FSSK,
)

# --- kernel ---
from .kernel import (
    FreeKernel,
    free_kernel,
    SigKernel,
    HigherOrderKernel,
    higher_order_kernel,
    FSSKSigKernel,
    fssk_sigkernel,
    LinearKernel,
    RBFKernel,
)


def core_expected_memory(
    *,
    dims: int | tuple[int, int],
    max_trunc: int | tuple[int, int],
    representation: Literal[
        "ordered", "partially_symmetrized"
    ] = "ordered",
    coordinates: Literal["standard", "shear"] = "standard",
    unit: str = "MiB",
    precompute_shuffle: bool | Literal["generator"] = False,
    breakdown: bool = False,
) -> float | dict[str, float]:
    """Estimate exact eager plan-buffer payload for the public core factories.

    Dispatch matches :func:`set_default_core`: standard coordinates use an
    integer/integer total configuration or pair/pair bidegree configuration.
    Bidegree cores accept ``representation="ordered"`` or
    ``"partially_symmetrized"``.  Shear coordinates require pair-valued
    ``dims`` and use the shape of
    ``max_trunc`` to select total degree or bidegree.  Results default to
    ``"MiB"``; ``unit`` may instead be ``"bytes"``, ``"KiB"``, or ``"GiB"``.
    Set ``breakdown=True`` to return the category mapping including ``total``.
    The exact count is derived without constructing a core or allocating any
    payload-sized plan arrays.  It excludes Python-container overhead,
    runtime temporaries, and compiled executable storage; high-order symbolic
    counting may still take time.
    """
    if not isinstance(unit, str):
        raise TypeError(f"unit must be a string, got {unit!r}.")
    unit_divisors = {
        "bytes": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    if unit not in unit_divisors:
        raise ValueError(
            "unit must be exactly one of 'bytes', 'KiB', 'MiB', or 'GiB', "
            f"got {unit!r}."
        )
    if not isinstance(breakdown, bool):
        raise TypeError(f"breakdown must be a boolean, got {breakdown!r}.")

    configuration = _resolve_core_configuration(
        dims=dims,
        max_trunc=max_trunc,
        representation=representation,
        coordinates=coordinates,
        precompute_shuffle=precompute_shuffle,
    )
    bytes_by_category = dict(
        configuration.expected_memory_bytes_by_category()
    )

    divisor = unit_divisors[unit]
    result = {
        name: value / divisor
        for name, value in bytes_by_category.items()
    }
    result["total"] = sum(bytes_by_category.values()) / divisor
    return result if breakdown else result["total"]


__all__ = [
    # core
    "Jax",
    "JaxSequentialCore",
    "JaxBigraded",
    "JaxPartiallySymmetrizedBigraded",
    "JaxPartiallySymmetrizedShearBigraded",
    "BigradedSpec",
    "BigradedTensor",
    "total_degree_core",
    "bigraded_core",
    "shear_core",
    "symmetrized_core",
    "core_expected_memory",
    "get_default_core",
    "get_default_core_pair",
    "get_default_seq_core",
    "set_default_core",
    "reset_default_core",
    # core operations
    "tensor_summation",
    "tensor_scalar_multiply",
    "tensor_dilation",
    "tensor_product",
    "tensor_product_homogeneous",
    "tensor_shuffle_vector",
    "tensor_shuffle_vector_homogeneous",
    "tensor_shuffle_product",
    "tensor_shuffle_product_homogeneous",
    "tensor_exponential",
    "tensor_logarithm",
    "tensor_fmexp",
    "tensor_inner_product",
    "tensor_inner_product_homogeneous",
    "tensor_signature_inner_product",
    "tensor_signature_inner_product_homogeneous",
    "tensor_shear_inner_product",
    "tensor_shear_inner_product_homogeneous",
    "tensor_adjoint_product",
    "tensor_adjoint_left_homogeneous",
    "tensor_adjoint_right_homogeneous",
    "tensor_matrix_product",
    "tensor_matrix_product_homogeneous",
    "tensor_matrix_product_left",
    "tensor_matrix_product_left_homogeneous",
    "tensor_matrix_product_right",
    "tensor_matrix_product_right_homogeneous",
    "tensor_stack",
    "tensor_moveaxis",
    "tensor_densify",
    "tensor_from_flat",
    "tensor_to_flat",
    "tensor_flatten",
    "tensor_slice",
    "tensor_from_standard_coordinates",
    "tensor_to_standard_coordinates",
    "tensor_partially_symmetrize",
    "tensor_partially_symmetrize_homogeneous",
    "tensor_to_ordered",
    # development
    "FreeDevelopment",
    "free_development",
    "Signature",
    "path_signature",
    # volterra
    "vsig",
    "VolterraSignature",
    "FractionalKernel",
    "GammaKernel",
    "FSSKConvolutionKernel",
    # sss
    "fssk_vsig",
    "fssk_state",
    "StateSpaceSignature",
    "FSSK",
    # kernel
    "FreeKernel",
    "free_kernel",
    "SigKernel",
    "HigherOrderKernel",
    "higher_order_kernel",
    "FSSKSigKernel",
    "fssk_sigkernel",
    "LinearKernel",
    "RBFKernel",
]

_CORE_METHOD_EXPORTS = (
    # summation / scaling
    "tensor_summation",
    "tensor_scalar_multiply",
    "tensor_dilation",
    # tensor (Chen) product
    "tensor_product",
    "tensor_product_homogeneous",
    # shuffle product
    "tensor_shuffle_vector",
    "tensor_shuffle_vector_homogeneous",
    "tensor_shuffle_product",
    "tensor_shuffle_product_homogeneous",
    # exponential / logarithm
    "tensor_exponential",
    "tensor_logarithm",
    "tensor_fmexp",
    # inner product / adjoint
    "tensor_inner_product",
    "tensor_inner_product_homogeneous",
    "tensor_signature_inner_product",
    "tensor_signature_inner_product_homogeneous",
    "tensor_shear_inner_product",
    "tensor_shear_inner_product_homogeneous",
    "tensor_adjoint_product",
    "tensor_adjoint_left_homogeneous",
    "tensor_adjoint_right_homogeneous",
    # matrix-valued tensor algebra
    "tensor_matrix_product",
    "tensor_matrix_product_homogeneous",
    "tensor_matrix_product_left",
    "tensor_matrix_product_left_homogeneous",
    "tensor_matrix_product_right",
    "tensor_matrix_product_right_homogeneous",
    # layout utilities
    "tensor_stack",
    "tensor_moveaxis",
    "tensor_densify",
    "tensor_from_flat",
    "tensor_to_flat",
    "tensor_slice",
    "tensor_from_standard_coordinates",
    "tensor_to_standard_coordinates",
    "tensor_partially_symmetrize",
    "tensor_partially_symmetrize_homogeneous",
    "tensor_to_ordered",
)


def _rebind_default_core(core, seq_core) -> None:
    """Bind top-level operations directly to one resolved core instance."""
    # Resolve the complete mapping before mutating module globals.  A core with
    # an incomplete public contract therefore cannot leave a partial rebind.
    methods = {name: getattr(core, name) for name in _CORE_METHOD_EXPORTS}
    methods["tensor_flatten"] = methods["tensor_to_flat"]
    methods["_core"] = core
    methods["_seq_core"] = seq_core
    globals().update(methods)


# Direct bound methods avoid a proxy lookup per operation.  Changing the
# default rebinds these names once.
_unregister_default_core_callback = _register_default_core_callback(
    _rebind_default_core,
    notify=True,
)
