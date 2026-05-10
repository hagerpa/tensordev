"""VolterraSignature — composition wrapper around VolterraKernel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax

from tensordev.core.universal import DenseElem
from tensordev.volterra.kernel import VolterraKernel, FractionalKernel, GammaKernel
from tensordev.volterra.iteration import vsig as _vsig

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class VolterraSignature:
    """
    Thin wrapper around :class:`VolterraKernel` that bundles a truncation level.

    Parameters
    ----------
    kernel:
        The underlying Volterra kernel.
    trunc:
        Tensor truncation level. Required. Static (changes cause retracing).
    """

    kernel: VolterraKernel
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
            times: Optional[Array] = None,
            dt: Optional[Array | float] = None,
            axis: int = -2,
            output_starting_point: bool = False,
            increment_input: bool = False,
    ) -> DenseElem:
        """Compute the truncated Volterra signature of ``X``.

        Thin wrapper around :func:`vsig`. ``self.kernel`` and ``self.trunc`` are
        forwarded automatically.

        Parameters
        ----------
        X:
            Path nodes or increments ``(..., S, d)``; step axis ``axis``.
        times:
            Optional one-dimensional node times of shape ``(S + 1,)``.
            Mutually exclusive with ``dt``.
        dt:
            Optional scalar uniform step size.  If both ``times`` and ``dt``
            are omitted, ``dt=1`` is used.
        axis:
            Step axis of ``X`` (default ``-2``).
        output_starting_point:
            If ``True``, return the full trajectory ``[1, V_0, ..., V_{S-1}]``
            with the trajectory axis at ``axis``.
        increment_input:
            Treat ``X`` as increments rather than path nodes.

        Returns
        -------
        DenseElem
            Terminal Volterra signature.  With ``output_starting_point=True``,
            each level carries an additional trajectory axis at ``axis``.
        """
        return _vsig(
            X,
            kernel=self.kernel,
            trunc=self.trunc,
            times=times,
            dt=dt,
            axis=axis,
            output_starting_point=output_starting_point,
            increment_input=increment_input,
        )


__all__ = ["VolterraSignature"]