from __future__ import annotations

from dataclasses import dataclass, field

import jax

from tensordev.volterra.combinatorics import MultiIndexLayout


Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class FSSKCoefficients:
    r"""
    Packed coefficients for finite-state-space kernels.

    Let ``batch_shape = dt.shape``. The packed multi-index layout stores all
    ``ell in N^q`` with ``|ell| <= trunc - 1``. Then:

    - ``psi[..., idx, :] = \widehat\psi_{ell(idx)}`` for all packed ``idx``,
    - ``phi[..., p, idx, :, :] = \widehat\Phi_{p + 1, ell(idx)}`` for packed
      indices with ``|ell| <= trunc - 2``.

    Thus the coefficient arrays have shapes:

    - ``E``:   ``batch_shape + (R, R)``
    - ``psi``: ``batch_shape + (M, R)``
    - ``phi``: ``batch_shape + (q, Mphi, R, R)``

    where:

    - ``M = #{ell in N^q : |ell| <= trunc - 1}``,
    - ``Mphi = #{ell in N^q : |ell| <= trunc - 2}``.

    No distinguished meaning is attached here to the leading batch axes; they
    simply correspond to whatever batch of time increments was used to build the
    coefficients. A caller may later interpret one of those axes as time, but
    that interpretation is external to this class.
    """

    layout: MultiIndexLayout
    trunc: int = field(metadata={"static": True})
    m: int = field(metadata={"static": True})
    q: int = field(metadata={"static": True})
    R: int = field(metadata={"static": True})
    E: Array                # shape batch_shape + (R, R)
    psi: Array              # shape batch_shape + (M, R)
    phi: Array              # shape batch_shape + (q, Mphi, R, R)


__all__ = ["FSSKCoefficients"]