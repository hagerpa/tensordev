# Low-level utilities for the tensordev.core package.
#
# Public surface:
#   annotations              — @dummy_jit decorator and iter_class_jittables
#   shuffle_precalculation   — assemble_shuffle_algebra / _homogeneous
#   shuffle_basics           — numba primitives (factorial_nb, shuffle_nb, …)
#
# End-user shuffle algebra is accessed via Universal.precompute_shuffle().

from .shuffle_precalculation import assemble_shuffle_algebra, assemble_shuffle_algebra_homogeneous

__all__ = ["assemble_shuffle_algebra", "assemble_shuffle_algebra_homogeneous"]

