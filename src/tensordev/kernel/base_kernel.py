from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp

from tensordev.kernel.parallel import pmap_batch

Array = jnp.ndarray


class BaseKernel(ABC):
    """
    Abstract base class for configured kernel objects.

    A ``BaseKernel`` instance is a callable object representing a kernel with
    fixed hyperparameters. Subclasses implement ``_compute`` for the actual
    single-device kernel evaluation and ``_as_sample_batch`` for normalization
    of public inputs into the batched empirical format expected by the empirical
    helper methods.

    Parallelism and chunking
    ------------------------
    ``__call__`` centralises both concerns:

    - **Chunking** (``max_batch``): a Python loop over blocks of at most
      ``max_batch`` samples is applied before any JAX work.  This is a
      memory guard — it limits peak device memory independently of the
      number of accelerators.
    - **Device parallelism** (``num_devices``): after chunking, each block is
      dispatched via ``pmap_batch`` across ``num_devices`` devices and
      delegated to ``_compute`` on each shard.

    Empirical convention
    --------------------
    The empirical helpers (``compute_kernel``, ``compute_Gram``, …) are thin
    wrappers around ``__call__`` that perform input normalisation, size
    validation, and — for the symmetric Gram case — the triangle optimisation.

    Input format
    ------------
    ``increment_input`` controls whether inputs are interpreted as path values
    (``False``, the default) or as pre-computed interval increments (``True``).
    It can be set once at construction time (via a subclass field) or overridden
    on a per-call basis by passing it explicitly to ``__call__`` or any empirical
    helper.
    """

    num_devices: int = 1
    increment_input: bool = False

    def __call__(
        self,
        X,
        Y,
        *,
        evaluate: str = "terminal",
        return_fg: bool = False,
        pairwise: bool = False,
        max_batch: Optional[int] = None,
        increment_input: Optional[bool] = None,
    ):
        inc = self.increment_input if increment_input is None else increment_input
        X = self._as_sample_batch(X)
        Y = self._as_sample_batch(Y)

        if max_batch is not None and max_batch > 0:
            bx = self._batch_size(X)
            by = self._batch_size(Y)

            if not pairwise and bx > max_batch:
                out = []
                for i in range(0, bx, max_batch):
                    out.append(self._dispatch(
                        self._slice_batch(X, i, i + max_batch),
                        self._slice_batch(Y, i, i + max_batch),
                        evaluate=evaluate, return_fg=return_fg, pairwise=False,
                        increment_input=inc,
                    ))
                return jnp.concatenate(out, axis=0)

            if pairwise and (bx > max_batch or by > max_batch):
                row_blocks = []
                for i in range(0, bx, max_batch):
                    col_blocks = []
                    for j in range(0, by, max_batch):
                        col_blocks.append(self._dispatch(
                            self._slice_batch(X, i, i + max_batch),
                            self._slice_batch(Y, j, j + max_batch),
                            evaluate=evaluate, return_fg=return_fg, pairwise=True,
                            increment_input=inc,
                        ))
                    row_blocks.append(jnp.concatenate(col_blocks, axis=1))
                return jnp.concatenate(row_blocks, axis=0)

        return self._dispatch(X, Y, evaluate=evaluate, return_fg=return_fg,
                              pairwise=pairwise, increment_input=inc)

    def _dispatch(self, X, Y, *, evaluate, return_fg, pairwise, increment_input):
        """pmap dispatch on already-normalised data."""
        if self.num_devices > 1:
            if pairwise:
                X, Y = self._broadcast_pairwise(X, Y)
                return pmap_batch(
                    lambda x: self._compute(x, Y, evaluate=evaluate, return_fg=return_fg,
                                            pairwise=False, increment_input=increment_input),
                    X,
                    num_devices=self.num_devices,
                )
            return pmap_batch(
                lambda x, y: self._compute(x, y, evaluate=evaluate, return_fg=return_fg,
                                           pairwise=False, increment_input=increment_input),
                X, Y,
                num_devices=self.num_devices,
            )
        return self._compute(X, Y, evaluate=evaluate, return_fg=return_fg,
                             pairwise=pairwise, increment_input=increment_input)

    @abstractmethod
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
        """Single-device kernel evaluation. Called by ``_dispatch`` on each device shard."""
        raise NotImplementedError

    def _broadcast_pairwise(self, X: Array, Y: Array):
        """
        Insert singleton batch axes so X and Y broadcast to an outer-product batch.

        Default implementation for plain arrays with batch prefix on ``shape[:-2]``.
        """
        batch_x = X.shape[:-2]
        batch_y = Y.shape[:-2]
        nx, ny = len(batch_x), len(batch_y)
        X = X.reshape(batch_x + (1,) * ny + X.shape[-2:])
        Y = Y.reshape((1,) * nx + batch_y + Y.shape[-2:])
        return X, Y

    @abstractmethod
    def _as_sample_batch(self, X):
        """
        Normalize a public input into the subclass-specific batched sample format.

        Returns a batched representation with the empirical sample axis on axis ``0``.
        """
        raise NotImplementedError

    def _batch_size(self, X: Array) -> int:
        """Return the empirical sample size on axis ``0`` of a plain array."""
        return int(jnp.asarray(X).shape[0])

    def _slice_batch(self, X: Array, start: int, stop: int) -> Array:
        """Slice the leading sample axis of a plain array."""
        return X[start:stop]

    @staticmethod
    def _off_diagonal_mean(K: Array) -> Array:
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("_off_diagonal_mean expects a square matrix.")
        n = K.shape[0]
        if n < 2:
            raise ValueError("Need at least two samples for off-diagonal averaging.")
        return (jnp.sum(K) - jnp.sum(jnp.diag(K))) / (n * (n - 1))

    def compute_kernel(self, X, Y, *, max_batch: Optional[int] = 100,
                       increment_input: Optional[bool] = None):
        """
        Compute batchwise kernel values ``(k(x_i, y_i))_i``.

        Parameters
        ----------
        X, Y :
            Public inputs accepted by the concrete kernel wrapper. After
            normalization, both inputs must have the same empirical sample size.
        max_batch : int, optional
            Passed to ``__call__`` to limit peak memory.
        increment_input : bool, optional
            If provided, overrides the instance's ``increment_input`` for this call.

        Returns
        -------
        Array
            Array of shape ``(n,)`` containing the batchwise kernel values.

        Raises
        ------
        ValueError
            If ``X`` and ``Y`` do not have the same empirical sample size.
        """
        X = self._as_sample_batch(X)
        Y = self._as_sample_batch(Y)
        if self._batch_size(X) != self._batch_size(Y):
            raise ValueError("compute_kernel expects matching batch sizes for X and Y.")
        return self(X, Y, evaluate="terminal", return_fg=False, pairwise=False,
                    max_batch=max_batch, increment_input=increment_input)

    def compute_Gram(self, X, Y=None, *, sym: bool = False,
                     max_batch: Optional[int] = 100,
                     increment_input: Optional[bool] = None):
        """
        Compute the Gram matrix ``(k(x_i, y_j))_{i,j}``.

        Parameters
        ----------
        X, Y :
            Public inputs accepted by the concrete kernel wrapper. If ``Y is None``,
            then ``Y = X`` and ``sym`` is forced to ``True``.
        sym : bool, default=False
            If ``True`` and the two empirical sample sizes agree, only the upper
            triangle is computed blockwise and then mirrored.
        max_batch : int, optional
            Passed to ``__call__`` (non-sym) or used as block size (sym triangle).
        increment_input : bool, optional
            If provided, overrides the instance's ``increment_input`` for this call.

        Returns
        -------
        Array
            Gram matrix of shape ``(n_x, n_y)``.

        Raises
        ------
        ValueError
            If ``sym=True`` and the two empirical sample sizes do not agree.
        """
        inc = self.increment_input if increment_input is None else increment_input
        X = self._as_sample_batch(X)
        if Y is None:
            Y = X
            sym = True
        else:
            Y = self._as_sample_batch(Y)

        if not sym:
            return self(X, Y, evaluate="terminal", return_fg=False, pairwise=True,
                        max_batch=max_batch, increment_input=inc)

        bx = self._batch_size(X)
        by = self._batch_size(Y)
        if bx != by:
            raise ValueError("sym=True requires X and Y to have the same batch size.")

        if max_batch is None or max_batch <= 0:
            return self._dispatch(X, Y, evaluate="terminal", return_fg=False,
                                  pairwise=True, increment_input=inc)

        G = None
        for i in range(0, bx, max_batch):
            i1 = min(i + max_batch, bx)
            Xi = self._slice_batch(X, i, i1)
            for j in range(i, by, max_batch):
                j1 = min(j + max_batch, by)
                Yj = self._slice_batch(Y, j, j1)
                block = self._dispatch(Xi, Yj, evaluate="terminal", return_fg=False,
                                       pairwise=True, increment_input=inc)
                if G is None:
                    G = jnp.zeros((bx, by), dtype=block.dtype)
                G = G.at[i:i1, j:j1].set(block)
                if i != j:
                    G = G.at[j:j1, i:i1].set(block.T)

        if G is None:
            raise ValueError("Cannot compute a Gram matrix from an empty batch.")
        return G

    def compute_mmd(self, X, Y, *, max_batch: Optional[int] = 100,
                    increment_input: Optional[bool] = None):
        """
        Compute the empirical maximum mean discrepancy induced by the kernel.

        The within-sample terms use the off-diagonal average of the corresponding
        Gram matrices.

        Parameters
        ----------
        X, Y :
            Public inputs accepted by the concrete kernel wrapper.
        max_batch : int, optional
            Block size used internally for Gram matrix computation.
        increment_input : bool, optional
            If provided, overrides the instance's ``increment_input`` for this call.

        Returns
        -------
        Array
            Scalar array containing the empirical MMD.
        """
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch, increment_input=increment_input)
        Kyy = self.compute_Gram(Y, Y, sym=True, max_batch=max_batch, increment_input=increment_input)
        Kxy = self.compute_Gram(X, Y, sym=False, max_batch=max_batch, increment_input=increment_input)
        return self._off_diagonal_mean(Kxx) + self._off_diagonal_mean(Kyy) - 2.0 * jnp.mean(Kxy)

    def compute_scoring_rule(self, X, y, *, max_batch: Optional[int] = 100,
                             increment_input: Optional[bool] = None):
        """
        Compute the empirical kernel scoring rule

            S(X, y) = E_offdiag[k(X, X)] - 2 E[k(X, y)],

        where ``X`` is a sample batch and ``y`` is a single sample.

        Parameters
        ----------
        X :
            Public input representing a batch of samples.
        y :
            Public input representing a single sample.
        max_batch : int, optional
            Block size used internally for Gram matrix computation.
        increment_input : bool, optional
            If provided, overrides the instance's ``increment_input`` for this call.

        Returns
        -------
        Array
            Scalar array containing the empirical scoring rule.

        Raises
        ------
        ValueError
            If ``y`` does not represent exactly one sample.
        """
        yb = self._as_sample_batch(y)
        if self._batch_size(yb) != 1:
            raise ValueError("compute_scoring_rule expects a single sample y.")
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch, increment_input=increment_input)
        Kxy = self.compute_Gram(X, yb, sym=False, max_batch=max_batch, increment_input=increment_input)
        return self._off_diagonal_mean(Kxx) - 2.0 * jnp.mean(Kxy)

    def compute_expected_scoring_rule(self, X, Y, *, max_batch: Optional[int] = 100,
                                      increment_input: Optional[bool] = None):
        """
        Compute the empirical expected kernel scoring rule

            E_Y[S(X, Y)] = E_offdiag[k(X, X)] - 2 E[k(X, Y)],

        where ``X`` and ``Y`` are sample batches.

        Parameters
        ----------
        X, Y :
            Public inputs accepted by the concrete kernel wrapper.
        max_batch : int, optional
            Block size used internally for Gram matrix computation.
        increment_input : bool, optional
            If provided, overrides the instance's ``increment_input`` for this call.

        Returns
        -------
        Array
            Scalar array containing the empirical expected scoring rule.
        """
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch, increment_input=increment_input)
        Kxy = self.compute_Gram(X, Y, sym=False, max_batch=max_batch, increment_input=increment_input)
        return self._off_diagonal_mean(Kxx) - 2.0 * jnp.mean(Kxy)

