"""Advanced explicit JAX core classes for ordered shear coordinates.

Most users should select these through ``tensordev.set_default_core`` with
``coordinates="shear"``.  ``JaxShearTotal`` stores dense total-degree levels;
``JaxShearBigraded`` stores true placement-factored bidegree levels.
"""

from .bigraded import JaxShearBigraded
from .jax import JaxShearTotal
from .symmetrized import JaxPartiallySymmetrizedShearBigraded
from .factory import shear_core

__all__ = [
    "JaxShearTotal",
    "JaxShearBigraded",
    "JaxPartiallySymmetrizedShearBigraded",
    "shear_core",
]
