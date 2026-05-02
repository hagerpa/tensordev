"""Public package exports."""

from math import comb

from .core import Einsum, Jax, Universal
from .development import FreeDevelopment, free_development, Signature, path_signature
from ._backend import get_default_core, get_default_seq_core, shuffle_core


def shuffle_core_expected_memory(d: int, trunc: int) -> float:
    """Upper bound on the memory (in MB) that ``shuffle_core(d, trunc)`` will allocate.

    Uses the formula ``d^(i+j) * C(i+j, i)`` entries per operator pair ``(i, j)``,
    which is exact for ``j=0`` and otherwise an upper bound — the actual memory
    is typically 20–40% less at small ``d`` due to repeated-letter collisions.
    Each entry costs 32 bytes (three int64 indices + one float64 coefficient).
    """
    total_entries = 0
    for t in range(trunc + 1):
        for i in range(t, -1, -1):
            j = t - i
            if j > i:
                break
            total_entries += d ** t * comb(t, i)
    return total_entries * 32 / 1024 ** 2

__all__ = [
    "Einsum",
    "Jax",
    "Universal",
    "FreeDevelopment",
    "free_development",
    "Signature",
    "path_signature",
]

_core = get_default_core()

# Explicit module-level forwarding so IDEs surface these via td.<tab>.
# Grouped by family; all delegate to the default backend core singleton.

# --- summation / scaling ---
tensor_summation = _core.tensor_summation
tensor_scalar_multiply = _core.tensor_scalar_multiply
tensor_dilation = _core.tensor_dilation

# --- tensor (Chen) product ---
tensor_product = _core.tensor_product
tensor_product_homogeneous = _core.tensor_product_homogeneous

# --- shuffle product ---
tensor_shuffle_vector = _core.tensor_shuffle_vector
tensor_shuffle_vector_homogeneous = _core.tensor_shuffle_vector_homogeneous

# --- exponential / logarithm ---
tensor_exponential = _core.tensor_exponential
tensor_logarithm = _core.tensor_logarithm
tensor_fmexp = _core.tensor_fmexp

# --- inner product / adjoint ---
tensor_inner_product = _core.tensor_inner_product
tensor_inner_product_homogeneous = _core.tensor_inner_product_homogeneous
tensor_adjoint_product = _core.tensor_adjoint_product
tensor_adjoint_left_homogeneous = _core.tensor_adjoint_left_homogeneous
tensor_adjoint_right_homogeneous = _core.tensor_adjoint_right_homogeneous

# --- matrix-valued tensor algebra ---
tensor_matrix_product = _core.tensor_matrix_product
tensor_matrix_product_homogeneous = _core.tensor_matrix_product_homogeneous
tensor_matrix_product_left = _core.tensor_matrix_product_left
tensor_matrix_product_left_homogeneous = _core.tensor_matrix_product_left_homogeneous
tensor_matrix_product_right = _core.tensor_matrix_product_right
tensor_matrix_product_right_homogeneous = _core.tensor_matrix_product_right_homogeneous

# --- layout utilities ---
tensor_stack = _core.tensor_stack
tensor_moveaxis = _core.tensor_moveaxis
tensor_densify = _core.tensor_densify
tensor_from_flat = _core.tensor_from_flat
tensor_to_flat = _core.tensor_to_flat
tensor_flatten = _core.tensor_to_flat
tensor_slice = _core.tensor_slice
