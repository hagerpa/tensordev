from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from tensordev.util.combinatorics import MultiIndexLayout


Array = jax.Array


def _leading_index(item, tail_ndim: int):
    if not isinstance(item, tuple):
        item = (item,)
    return item + (slice(None),) * tail_ndim


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class FSSKCoefficients:
    r"""
    Packed coefficients for finite-state-space kernels.

    Leading axes are generic broadcast axes.  For state recursion, the first
    leading axis is interpreted as the time axis.  Scalar coefficients may have
    no leading axis and can be converted to time length one using
    ``with_time_axis``.
    """

    layout: MultiIndexLayout
    trunc: int = field(metadata={"static": True})
    m: int = field(metadata={"static": True})
    q: int = field(metadata={"static": True})
    R: int = field(metadata={"static": True})
    E: Array                # leading + (R, R)
    psi: Array              # leading + (M, R)
    phi: Array              # leading + (q, Mphi, R, R)

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.E.shape[:-2]

    @property
    def leading_ndim(self) -> int:
        return len(self.leading_shape)

    @property
    def M(self) -> int:
        return int(self.psi.shape[-2])

    @property
    def Mphi(self) -> int:
        return int(self.phi.shape[-3])

    def __getitem__(self, item) -> "FSSKCoefficients":
        """Slice the leading coefficient axes only."""
        return replace(
            self,
            E=self.E[_leading_index(item, 2)],
            psi=self.psi[_leading_index(item, 2)],
            phi=self.phi[_leading_index(item, 4)],
        )

    def with_time_axis(self) -> "FSSKCoefficients":
        """Ensure a leading time axis exists.

        If the object already has leading axes, it is returned unchanged.
        If it is scalar in time, a length-one leading axis is inserted.
        """
        if self.leading_ndim > 0:
            return self
        return replace(
            self,
            E=self.E[None, ...],
            psi=self.psi[None, ...],
            phi=self.phi[None, ...],
        )

    def broadcast_to_leading_shape(
        self,
        leading_shape: tuple[int, ...],
    ) -> "FSSKCoefficients":
        """Broadcast all coefficient arrays to a common leading shape."""
        leading_shape = tuple(int(s) for s in leading_shape)
        return replace(
            self,
            E=jnp.broadcast_to(self.E, leading_shape + self.E.shape[-2:]),
            psi=jnp.broadcast_to(self.psi, leading_shape + self.psi.shape[-2:]),
            phi=jnp.broadcast_to(self.phi, leading_shape + self.phi.shape[-4:]),
        )

    def broadcast_time(self, steps: int) -> "FSSKCoefficients":
        """Broadcast the first leading axis to ``steps``.

        This is the convention used by state recursion.
        """
        coef = self.with_time_axis()
        steps = int(steps)

        if coef.leading_shape[0] not in (1, steps):
            raise ValueError(
                "Coefficient time axis must have length 1 or S; "
                f"got {coef.leading_shape[0]} for S={steps}."
            )

        return coef.broadcast_to_leading_shape(
            (steps,) + coef.leading_shape[1:]
        )


__all__ = ["FSSKCoefficients"]