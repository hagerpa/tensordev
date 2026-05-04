from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from tensordev.kernel.free import free_kernel
from tensordev.kernel.base_kernel import BaseKernel
from tensordev.kernel.static_kernels import LinearKernel, StaticKernel
from tensordev.util.path_preprocessing import DyadicOrder

Array = jnp.ndarray


@dataclass(frozen=True)
class SigKernel(BaseKernel):
    """
    Classical linear signature kernel for Euclidean paths.

    This class evaluates the signature kernel for paths in ``R^d`` and provides
    empirical kernel statistics such as batchwise kernel values, Gram matrices,
    MMD, and scoring rules.

    Input convention
    ----------------
    Inputs are paths of shape ``(batch, length, dim)`` or a single path of shape
    ``(length, dim)``. A single path is promoted to batch size ``1``.

    The parameter ``dyadic_order`` controls optional dyadic refinement of the
    time grid before evaluation.  It can be a single int (same for both paths)
    or a tuple ``(order_x, order_y)`` for asymmetric refinement.

    The parameter ``static_kernel`` replaces the default ``⟨dx_i, dy_j⟩``
    increment inner product with the discrete mixed-difference formula::

        G[i,j] = k(x_{i+1}, y_{j+1}) - k(x_i, y_{j+1})
               - k(x_{i+1}, y_j)   + k(x_i, y_j)

    where ``x_i`` are the path nodes.  The default ``LinearKernel(scale=1.0)``
    reproduces the classical signature kernel exactly.
    """

    backend: str = "scan"
    dyadic_order: DyadicOrder = 0
    num_devices: int = 1
    static_kernel: StaticKernel = LinearKernel(scale=1.0)
    increment_input: bool = False

    def _compute(
            self,
            X,
            Y,
            *,
            evaluate: str = "terminal",
            return_fg: bool = False,
            pairwise: bool = False,
            increment_input: bool = False,
    ):
        return free_kernel(
            (X,),
            (Y,),
            evaluate=evaluate,
            return_fg=return_fg,
            pairwise=pairwise,
            backend=self.backend,
            dyadic_order=self.dyadic_order,
            increment_in=increment_input,
            static_kernel=self.static_kernel,
        )

    def _as_sample_batch(self, X):
        """
        Normalize path input to shape ``(batch, length, dim)``.

        Parameters
        ----------
        X :
            Path batch of shape ``(batch, length, dim)`` or a single path of shape
            ``(length, dim)``.

        Returns
        -------
        Array
            Array of shape ``(batch, length, dim)``.

        Raises
        ------
        ValueError
            If the input does not have path shape or contains fewer than two time
            points.
        """
        X = jnp.asarray(X)
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim < 3:
            raise ValueError("Expected shape (..., length, dim) with at least one batch axis.")
        if X.shape[-2] < 2:
            raise ValueError("Each path must contain at least two time points.")
        return X
