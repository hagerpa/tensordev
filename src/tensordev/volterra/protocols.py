from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


Array = Any
ArrayNamespace = Any
DType = Any


@runtime_checkable
class VolterraKernel(Protocol):
    """
    Protocol for Volterra kernel specifications.

    The kernel itself is the mathematical object. It produces
    discretization-dependent coefficient objects, which may be cached
    internally by the concrete implementation.

    There are two entry points:

    - ``coef(dt=..., trunc=...)`` for uniform grids
    - ``coef_grid(time=..., trunc=...)`` for arbitrary grids
    """

    path_dim: int

    def coef(
        self,
        dt: float,
        *,
        trunc: int,
        xp: Optional[ArrayNamespace] = None,
        dtype: Optional[DType] = None,
    ) -> "VolterraCoefficients":
        """
        Return coefficients for a uniform grid with step size ``dt``.

        This is the important fast path for convolution-type kernels and
        the natural place for internal caching.
        """
        ...


@runtime_checkable
class VolterraCoefficients(Protocol):
    """
    Protocol for discretization-dependent Volterra coefficient objects.

    A VolterraCoefficients object is bound to:
    - a kernel,
    - a truncation level,
    - and either a uniform step size ``dt`` or a general time grid.
    """

    kernel: VolterraKernel
    trunc: int

    @property
    def uniform(self) -> bool:
        """
        Whether these coefficients correspond to a uniform grid.
        """
        ...

    def sig(self, x: Array) -> Any:
        """
        Compute the truncated Volterra signature of ``x``.
        """
        ...

    def sigkernel(self, x: Array, y: Array) -> Array:
        """
        Compute the Volterra signature kernel between ``x`` and ``y``.
        """
        ...

    def state(self, x: Array) -> Any:
        """
        Return the terminal internal state associated with ``x``.
        """
        ...

    def readout(self, x: Array) -> Any:
        """
        Return the terminal readout associated with ``x``.
        """
        ...


__all__ = [
    "VolterraKernel",
    "VolterraCoefficients",
]