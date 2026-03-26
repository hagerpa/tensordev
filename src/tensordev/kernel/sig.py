from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from tensordev import Jax
from tensordev.kernel.free import free_kernel
from tensordev.kernel.base_kernel import BaseKernel

Array = jnp.ndarray
JaxCore = Jax()


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
    time grid before evaluation.
    """

    backend: str = "scan"
    dyadic_order: int = 0
    core: object = None

    def __call__(
            self,
            X,
            Y,
            *,
            evaluate: str = "terminal",
            return_fg: bool = False,
            pairwise: bool = False,
    ):
        """
        Evaluate the configured classical signature kernel.

        Parameters
        ----------
        X, Y :
            Normalized batched path inputs of shape ``(batch, length, dim)``.
        evaluate : {"terminal", "grid"}, default="terminal"
            Whether to return only the terminal kernel values or the full discrete
            solution.
        return_fg : bool, default=False
            Whether to additionally return the tensor-valued components ``f`` and
            ``g`` inherited from the underlying free-kernel solver.
        pairwise : bool, default=False
            Whether to evaluate batchwise or pairwise over the empirical samples.

        Returns
        -------
        Array or tuple
            Output of the underlying ``free_kernel`` call.
        """
        return free_kernel(
            (X,),
            (Y,),
            evaluate=evaluate,
            return_fg=return_fg,
            pairwise=pairwise,
            backend=self.backend,
            dyadic_order=self.dyadic_order,
            core=self.core,
            increment_in=False,
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
        if X.ndim != 3:
            raise ValueError("Expected shape (batch, length, dim) or (length, dim).")
        if X.shape[-2] < 2:
            raise ValueError("Each path must contain at least two time points.")
        return X
