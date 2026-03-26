from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp

Array = jnp.ndarray


class BaseKernel(ABC):
    """
    Abstract base class for configured kernel objects.

    A ``BaseKernel`` instance is a callable object representing a kernel with
    fixed hyperparameters. Subclasses implement ``__call__`` for the actual
    kernel evaluation and ``_as_sample_batch`` for normalization of public
    inputs into the batched empirical format expected by the empirical helper
    methods.

    Empirical convention
    --------------------
    The empirical methods in this class interpret axis ``0`` as the sample axis.
    They provide generic implementations of

    - batchwise kernel evaluation,
    - Gram matrix computation,
    - MMD,
    - scoring rules,

    by repeatedly calling the configured kernel object on blocks of samples.

    Design note
    -----------
    The raw kernel implementation may support richer batch conventions than a
    single leading sample axis. The empirical helpers here intentionally use the
    simpler convention that axis ``0`` indexes samples.
    """

    @abstractmethod
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
        Evaluate the configured kernel on two batched inputs.

        Parameters
        ----------
        X, Y :
            Batched kernel inputs in the subclass-specific normalized format.
        evaluate : {"terminal", "grid"}, default="terminal"
            Whether to return only the terminal kernel values or the full
            discrete solution, if supported by the concrete kernel.
        return_fg : bool, default=False
            Whether to additionally return auxiliary tensor-valued quantities,
            if supported by the concrete kernel.
        pairwise : bool, default=False
            If ``False``, evaluate the kernel batchwise:
            ``(k(x_i, y_i))_i``.
            If ``True``, evaluate the pairwise Gram block:
            ``(k(x_i, y_j))_{i,j}``.

        Returns
        -------
        Array or tuple
            Output of the concrete kernel implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def _as_sample_batch(self, X):
        """
        Normalize a public input into the subclass-specific batched sample format.

        Parameters
        ----------
        X :
            Public input accepted by the concrete kernel wrapper.

        Returns
        -------
        object
            Batched representation with sample axis on axis ``0``.
        """
        raise NotImplementedError

    def _batch_size(self, X) -> int:
        """
        Return the empirical sample size carried on axis ``0``.

        Parameters
        ----------
        X :
            A normalized batched input. This may be a single array or a tuple/list
            of arrays sharing the same leading sample axis.

        Returns
        -------
        int
            Number of empirical samples.

        Raises
        ------
        ValueError
            If the input is empty or if tensor levels disagree on the leading
            sample axis.
        """
        if isinstance(X, (tuple, list)):
            if not X:
                raise ValueError("Expected at least one tensor level.")
            n = int(X[0].shape[0])
            for k, level in enumerate(X, start=1):
                if int(level.shape[0]) != n:
                    raise ValueError(
                        "All tensor levels must have the same leading sample axis. "
                        f"Level 1 has {n}, level {k} has {level.shape[0]}."
                    )
            return n

        X = jnp.asarray(X)
        if X.ndim < 1:
            raise ValueError("Expected an input with a leading sample axis.")
        return int(X.shape[0])

    def _slice_batch(self, X, start: int, stop: int):
        """
        Slice the empirical sample axis on a normalized batched input.

        Parameters
        ----------
        X :
            Normalized batched input.
        start, stop : int
            Slice bounds along the leading sample axis.

        Returns
        -------
        object
            Same container structure as ``X``, restricted to samples
            ``start:stop``.
        """
        if isinstance(X, tuple):
            return tuple(level[start:stop] for level in X)
        if isinstance(X, list):
            return [level[start:stop] for level in X]
        return X[start:stop]

    @staticmethod
    def _off_diagonal_mean(K: Array) -> Array:
        """
        Compute the mean of the off-diagonal entries of a square Gram matrix.

        Parameters
        ----------
        K : Array
            Square matrix of shape ``(n, n)``.

        Returns
        -------
        Array
            Scalar array containing the off-diagonal average.

        Raises
        ------
        ValueError
            If ``K`` is not square or if ``n < 2``.
        """
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("_off_diagonal_mean expects a square matrix.")
        n = K.shape[0]
        if n < 2:
            raise ValueError("Need at least two samples for off-diagonal averaging.")
        return (jnp.sum(K) - jnp.sum(jnp.diag(K))) / (n * (n - 1))

    def compute_kernel(self, X, Y, *, max_batch: Optional[int] = 100):
        """
        Compute batchwise kernel values ``(k(x_i, y_i))_i``.

        Parameters
        ----------
        X, Y :
            Public inputs accepted by the concrete kernel wrapper. After
            normalization, both inputs must have the same empirical sample size.
        max_batch : int, optional
            If given and positive, evaluate the kernel in blocks of at most
            ``max_batch`` samples along the empirical sample axis. If ``None`` or
            non-positive, evaluate in a single call.

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

        bx = self._batch_size(X)
        by = self._batch_size(Y)
        if bx != by:
            raise ValueError("compute_kernel expects matching batch sizes for X and Y.")

        if max_batch is None or max_batch <= 0 or bx <= max_batch:
            return self(X, Y, evaluate="terminal", return_fg=False, pairwise=False)

        out = []
        for i in range(0, bx, max_batch):
            out.append(
                self(
                    self._slice_batch(X, i, i + max_batch),
                    self._slice_batch(Y, i, i + max_batch),
                    evaluate="terminal",
                    return_fg=False,
                    pairwise=False,
                )
            )
        return jnp.concatenate(out, axis=0)

    def compute_Gram(self, X, Y=None, *, sym: bool = False, max_batch: Optional[int] = 100):
        """
        Compute the Gram matrix ``(k(x_i, y_j))_{i,j}``.

        Parameters
        ----------
        X, Y :
            Public inputs accepted by the concrete kernel wrapper. If ``Y is None``,
            then ``Y = X`` and ``sym`` is forced to ``True``.
        sym : bool, default=False
            If ``True`` and the two empirical sample sizes agree, only the upper
            triangle is computed blockwise and then mirrored. This is useful when
            computing a symmetric Gram matrix.
        max_batch : int, optional
            If given and positive, evaluate the Gram matrix blockwise using blocks
            of size at most ``max_batch``. If ``None`` or non-positive, evaluate
            in a single call.

        Returns
        -------
        Array
            Gram matrix of shape ``(n_x, n_y)``.

        Raises
        ------
        ValueError
            If ``sym=True`` and the two empirical sample sizes do not agree.
        """
        X = self._as_sample_batch(X)
        if Y is None:
            Y = X
            sym = True
        else:
            Y = self._as_sample_batch(Y)

        bx = self._batch_size(X)
        by = self._batch_size(Y)

        if max_batch is None or max_batch <= 0:
            return self(X, Y, evaluate="terminal", return_fg=False, pairwise=True)

        if not sym:
            row_blocks = []
            for i in range(0, bx, max_batch):
                Xi = self._slice_batch(X, i, i + max_batch)
                col_blocks = []
                for j in range(0, by, max_batch):
                    Yj = self._slice_batch(Y, j, j + max_batch)
                    col_blocks.append(
                        self(Xi, Yj, evaluate="terminal", return_fg=False, pairwise=True)
                    )
                row_blocks.append(jnp.concatenate(col_blocks, axis=1))
            return jnp.concatenate(row_blocks, axis=0)

        if bx != by:
            raise ValueError("sym=True requires X and Y to have the same batch size.")

        G = None
        for i in range(0, bx, max_batch):
            i1 = min(i + max_batch, bx)
            Xi = self._slice_batch(X, i, i1)
            for j in range(i, by, max_batch):
                j1 = min(j + max_batch, by)
                Yj = self._slice_batch(Y, j, j1)
                block = self(Xi, Yj, evaluate="terminal", return_fg=False, pairwise=True)
                if G is None:
                    G = jnp.zeros((bx, by), dtype=block.dtype)
                G = G.at[i:i1, j:j1].set(block)
                if i != j:
                    G = G.at[j:j1, i:i1].set(block.T)

        if G is None:
            raise ValueError("Cannot compute a Gram matrix from an empty batch.")
        return G

    def compute_mmd(self, X, Y, *, max_batch: Optional[int] = 100):
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

        Returns
        -------
        Array
            Scalar array containing the empirical MMD.
        """
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        Kyy = self.compute_Gram(Y, Y, sym=True, max_batch=max_batch)
        Kxy = self.compute_Gram(X, Y, sym=False, max_batch=max_batch)
        return self._off_diagonal_mean(Kxx) + self._off_diagonal_mean(Kyy) - 2.0 * jnp.mean(Kxy)

    def compute_scoring_rule(self, X, y, *, max_batch: Optional[int] = 100):
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
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        Kxy = self.compute_Gram(X, yb, sym=False, max_batch=max_batch)
        return self._off_diagonal_mean(Kxx) - 2.0 * jnp.mean(Kxy)

    def compute_expected_scoring_rule(self, X, Y, *, max_batch: Optional[int] = 100):
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

        Returns
        -------
        Array
            Scalar array containing the empirical expected scoring rule.
        """
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        Kxy = self.compute_Gram(X, Y, sym=False, max_batch=max_batch)
        return self._off_diagonal_mean(Kxx) - 2.0 * jnp.mean(Kxy)