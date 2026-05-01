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
class VolterraCoefficients:
    r"""
    Packed interval coefficients for the quadratic Volterra-signature scheme.

    This object stores the coefficient arrays consumed by the symmetric
    shuffle/Horner evaluation routines.  The public package name deliberately
    does not include ``symmetric``; the implementation assumes the coefficient
    symmetry hypothesis from Part II, i.e. the coefficient of a word ``w p``
    depends on the prefix word ``w`` only through its multi-index ``ell``.

    For a triple ``(s,t,tau)`` and a prefix multi-index ``ell`` with
    ``|ell| <= trunc - 1``, ``alpha[..., p, ell_idx]`` stores the normalized
    coefficient

        K_{s,t}^{w(ell) p, tau} / (t - s)^{|ell| + 1}.

    Consequently, ``alpha`` has trailing shape ``(q, M)``, where
    ``M = #{ell in N^q : |ell| <= trunc - 1}``.  All axes before ``(q, M)``
    are generic leading axes.  For grid precomputation these leading axes are
    usually ``(source_interval, readout_index)``.
    """

    layout: MultiIndexLayout
    trunc: int = field(metadata={"static": True})
    m: int = field(metadata={"static": True})
    q: int = field(metadata={"static": True})
    alpha: Array        # leading + (q, M)
    valid: Array        # leading, boolean mask for Delta^3-valid triples

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.alpha.shape[:-2]

    @property
    def leading_ndim(self) -> int:
        return len(self.leading_shape)

    @property
    def M(self) -> int:
        return int(self.alpha.shape[-1])

    @property
    def dtype(self) -> jnp.dtype:
        return self.alpha.dtype

    def __getitem__(self, item) -> "VolterraCoefficients":
        """Slice the leading coefficient axes only."""
        if not isinstance(item, tuple):
            item = (item,)
        return replace(
            self,
            alpha=self.alpha[_leading_index(item, 2)],
            valid=self.valid[item],
        )

    def with_leading_axis(self) -> "VolterraCoefficients":
        """Ensure that a leading axis exists.

        If the object already has leading axes, it is returned unchanged.  If it
        is scalar in the leading axes, a length-one leading axis is inserted.
        """
        if self.leading_ndim > 0:
            return self
        return replace(
            self,
            alpha=self.alpha[None, ...],
            valid=self.valid[None, ...],
        )

    def broadcast_to_leading_shape(
        self,
        leading_shape: tuple[int, ...],
    ) -> "VolterraCoefficients":
        """Broadcast ``alpha`` and ``valid`` to a common leading shape."""
        leading_shape = tuple(int(s) for s in leading_shape)
        return replace(
            self,
            alpha=jnp.broadcast_to(self.alpha, leading_shape + self.alpha.shape[-2:]),
            valid=jnp.broadcast_to(self.valid, leading_shape),
        )

    def broadcast_source_readout(
        self,
        steps: int,
        readouts: int | None = None,
    ) -> "VolterraCoefficients":
        """Broadcast the first two leading axes to a grid shape.

        This is the convention used by the quadratic Volterra recursion:
        leading axes ``(S, R)`` correspond to source intervals and readout
        indices.  Existing axes must either be singleton axes or already have
        the requested length.
        """
        coeffs = self
        if coeffs.leading_ndim == 0:
            coeffs = replace(
                coeffs,
                alpha=coeffs.alpha[None, None, ...],
                valid=coeffs.valid[None, None, ...],
            )
        elif coeffs.leading_ndim == 1:
            coeffs = replace(
                coeffs,
                alpha=coeffs.alpha[:, None, ...],
                valid=coeffs.valid[:, None],
            )

        steps = int(steps)
        readouts = steps if readouts is None else int(readouts)
        if coeffs.leading_shape[0] not in (1, steps):
            raise ValueError(
                "Coefficient source axis must have length 1 or S; "
                f"got {coeffs.leading_shape[0]} for S={steps}."
            )
        if coeffs.leading_shape[1] not in (1, readouts):
            raise ValueError(
                "Coefficient readout axis must have length 1 or R; "
                f"got {coeffs.leading_shape[1]} for R={readouts}."
            )
        return coeffs.broadcast_to_leading_shape(
            (steps, readouts) + coeffs.leading_shape[2:]
        )


def validate_volterra_coefficients(coeffs: VolterraCoefficients) -> None:
    """Host-side shape validation for packed Volterra coefficients."""
    expected = (coeffs.q, coeffs.layout.size)
    if coeffs.alpha.shape[-2:] != expected:
        raise ValueError(
            "Volterra coefficients must satisfy "
            f"alpha.shape[-2:] == {expected}, got {coeffs.alpha.shape[-2:]}"
        )
    if coeffs.valid.shape != coeffs.leading_shape:
        raise ValueError(
            "Volterra coefficient mask must have shape equal to leading_shape; "
            f"got valid.shape={coeffs.valid.shape} and leading_shape={coeffs.leading_shape}."
        )


__all__ = ["VolterraCoefficients", "validate_volterra_coefficients"]
