from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


# ===========================================================================================================
# Abstract base class
# ===========================================================================================================


class StaticKernel(ABC):
    """Abstract base class for static (pointwise) kernels k: R^d × R^d → R.

    A ``StaticKernel`` represents a kernel function that operates on individual
    points (or sequences of points), supporting both batchwise and pairwise
    (Gram matrix) evaluation.

    Shape convention
    ----------------
    The **last two** axes of every input are ``(length, dim)``.  All preceding
    axes form an **arbitrary** batch shape (possibly empty).  Outputs follow the
    same convention on the batch prefix:

    batch_kernel
        X: ``batch + (length_X, dim)``,  Y: ``batch + (length_Y, dim)``
        → ``batch + (length_X, length_Y)``

    Gram_matrix
        X: ``batch_X + (length_X, dim)``,  Y: ``batch_Y + (length_Y, dim)``
        → ``batch_X + batch_Y + (length_X, length_Y)``
    """

    @abstractmethod
    def batch_kernel(self, X: Array, Y: Array) -> Array:
        """Evaluate batchwise kernel values k(X^i_s, Y^i_t).

        Parameters
        ----------
        X : ``batch + (length_X, dim)``
        Y : ``batch + (length_Y, dim)``

        Returns
        -------
        ``batch + (length_X, length_Y)``
        """
        ...

    @abstractmethod
    def Gram_matrix(self, X: Array, Y: Array) -> Array:
        """Evaluate the full pairwise Gram block k(X^i_s, Y^j_t).

        Parameters
        ----------
        X : ``batch_X + (length_X, dim)``
        Y : ``batch_Y + (length_Y, dim)``

        Returns
        -------
        ``batch_X + batch_Y + (length_X, length_Y)``
        """
        ...


# ---------------------------------------------------------------------------
# Internal broadcast helper (mirrors free._broadcast_pairwise)
# ---------------------------------------------------------------------------

def _expand_for_gram(X: Array, Y: Array, n_trailing: int = 2) -> tuple[Array, Array]:
    """Expand X and Y batch axes into outer-product position.

    Given
        X.shape == batch_X + trailing,
        Y.shape == batch_Y + trailing   (``n_trailing`` trailing axes),
    returns
        X_b.shape == batch_X + (1,)*ny + trailing,
        Y_b.shape == (1,)*nx + batch_Y + trailing,
    so that elementwise operations broadcast to ``batch_X + batch_Y + trailing``.
    """
    batch_X = X.shape[:-n_trailing]
    batch_Y = Y.shape[:-n_trailing]
    nx, ny = len(batch_X), len(batch_Y)
    X_b = X.reshape(batch_X + (1,) * ny + X.shape[-n_trailing:])
    Y_b = Y.reshape((1,) * nx + batch_Y + Y.shape[-n_trailing:])
    return X_b, Y_b


# ===========================================================================================================
# Concrete static kernels
# ===========================================================================================================


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearKernel(StaticKernel):
    """Linear kernel k(x, y) = <scale·x, scale·y>.

    Parameters
    ----------
    scale : float, default=1.0
        Multiplicative scaling applied to both inputs before computing the
        inner product.
    """

    scale: float = 1.0

    def batch_kernel(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch + (length_X, dim)``  →  ``batch + (length_X, length_Y)``
        """
        # jnp.matmul broadcasts freely over any number of leading batch axes
        return jnp.matmul(self.scale * X, (self.scale * Y).swapaxes(-1, -2))

    def Gram_matrix(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch_X + (length_X, dim)``,  Y: ``batch_Y + (length_Y, dim)``
        →  ``batch_X + batch_Y + (length_X, length_Y)``
        """
        X_b, Y_b = _expand_for_gram(self.scale * X, self.scale * Y)
        return jnp.matmul(X_b, Y_b.swapaxes(-1, -2))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RBFKernel(StaticKernel):
    """RBF kernel k(x, y) = exp(-‖x − y‖² / sigma).

    Parameters
    ----------
    sigma : float, default=1.0
        Bandwidth parameter of the RBF kernel.
    """

    sigma: float = 1.0

    def batch_kernel(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch + (length_X, dim)``  →  ``batch + (length_X, length_Y)``
        """
        Xs = jnp.sum(X ** 2, axis=-1)                       # batch + (length_X,)
        Ys = jnp.sum(Y ** 2, axis=-1)                       # batch + (length_Y,)
        dist = -2.0 * jnp.matmul(X, Y.swapaxes(-1, -2))    # batch + (length_X, length_Y)
        dist = dist + Xs[..., None] + Ys[..., None, :]
        return jnp.exp(-dist / self.sigma)

    def Gram_matrix(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch_X + (length_X, dim)``,  Y: ``batch_Y + (length_Y, dim)``
        →  ``batch_X + batch_Y + (length_X, length_Y)``
        """
        X_b, Y_b = _expand_for_gram(X, Y)
        Xs = jnp.sum(X_b ** 2, axis=-1)                            # batch_X + (1,)*ny + (length_X,)
        Ys = jnp.sum(Y_b ** 2, axis=-1)                            # (1,)*nx + batch_Y + (length_Y,)
        dist = -2.0 * jnp.matmul(X_b, Y_b.swapaxes(-1, -2))       # batch_X + batch_Y + (length_X, length_Y)
        dist = dist + Xs[..., None] + Ys[..., None, :]
        return jnp.exp(-dist / self.sigma)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RBF_CEXP_Kernel(StaticKernel):
    """RBF CEXP kernel k: H × H → R.

    First maps each function through the integral operator induced by the
    cos-exp kernel (parameterised by ``sigma1`` and ``n_freqs``), then applies
    an RBF kernel with bandwidth ``sigma2`` on the resulting feature vectors.

    Input shape: ``batch + (length_t, length_x, dim)``

    Parameters
    ----------
    sigma1 : float, default=1.0
        Bandwidth of the cos-exp integral operator.
    sigma2 : float, default=1.0
        Bandwidth of the outer RBF kernel.
    n_freqs : int, default=20
        Number of frequencies used in the cos-exp kernel.
    """

    sigma1: float = 1.0
    sigma2: float = 1.0
    n_freqs: int = 20

    def batch_kernel(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch + (length_X_t, length_x, dim)``
        Y: ``batch + (length_Y_t, length_x, dim)``
        →  ``batch + (length_X_t, length_Y_t)``
        """
        CX = cexp(X, self.n_freqs, self.sigma1)   # batch + (length_X_t, length_x, dim)
        CY = cexp(Y, self.n_freqs, self.sigma1)   # batch + (length_Y_t, length_x, dim)
        # flatten last two axes: batch + (length_t, length_x × dim)
        CX = CX.reshape(CX.shape[:-2] + (-1,))
        CY = CY.reshape(CY.shape[:-2] + (-1,))
        return RBFKernel(self.sigma2).batch_kernel(CX, CY)

    def Gram_matrix(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch_X + (length_X_t, length_x, dim)``
        Y: ``batch_Y + (length_Y_t, length_x, dim)``
        →  ``batch_X + batch_Y + (length_X_t, length_Y_t)``
        """
        CX = cexp(X, self.n_freqs, self.sigma1)
        CY = cexp(Y, self.n_freqs, self.sigma1)
        CX = CX.reshape(CX.shape[:-2] + (-1,))
        CY = CY.reshape(CY.shape[:-2] + (-1,))
        return RBFKernel(self.sigma2).Gram_matrix(CX, CY)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RBF_SQR_Kernel(StaticKernel):
    """RBF SQR kernel k: H × H → R.

    Product of two RBF kernels: one applied to the flattened inputs and one
    applied to their element-wise squares.

    Input shape: ``batch + (length_t, length_x, dim)``

    Parameters
    ----------
    sigma1 : float, default=1.0
        Bandwidth for the RBF kernel applied to the raw flattened inputs.
    sigma2 : float, default=1.0
        Bandwidth for the RBF kernel applied to the squared flattened inputs.
    """

    sigma1: float = 1.0
    sigma2: float = 1.0

    def batch_kernel(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch + (length_X_t, length_x, dim)``
        Y: ``batch + (length_Y_t, length_x, dim)``
        →  ``batch + (length_X_t, length_Y_t)``
        """
        X = X.reshape(X.shape[:-2] + (-1,))  # batch + (length_X_t, length_x × dim)
        Y = Y.reshape(Y.shape[:-2] + (-1,))  # batch + (length_Y_t, length_x × dim)
        rbf1 = RBFKernel(self.sigma1)
        rbf2 = RBFKernel(self.sigma2)
        return rbf1.batch_kernel(X, Y) * rbf2.batch_kernel(X ** 2, Y ** 2)

    def Gram_matrix(self, X: Array, Y: Array) -> Array:
        """
        X: ``batch_X + (length_X_t, length_x, dim)``
        Y: ``batch_Y + (length_Y_t, length_x, dim)``
        →  ``batch_X + batch_Y + (length_X_t, length_Y_t)``
        """
        X = X.reshape(X.shape[:-2] + (-1,))
        Y = Y.reshape(Y.shape[:-2] + (-1,))
        rbf1 = RBFKernel(self.sigma1)
        rbf2 = RBFKernel(self.sigma2)
        return rbf1.Gram_matrix(X, Y) * rbf2.Gram_matrix(X ** 2, Y ** 2)


# ===========================================================================================================
# Helper functions
# ===========================================================================================================


def cexp(X: Array, n_freqs: int = 20, sigma: float = float(np.sqrt(10))) -> Array:
    """Transform function values through the integral operator of the cos-exp kernel.

    The function values are assumed to lie on the uniform grid over ``[0, 1]``,
    sampled along the ``-2`` axis (second-to-last).

    Parameters
    ----------
    X : ``batch + (length_t, length_x, dim)``
        Array of function values for arbitrary batch prefix.
    n_freqs : int, default=20
        Number of cosine frequencies to include in the sum.
    sigma : float, default=sqrt(10)
        Bandwidth of the cos-exp kernel.

    Returns
    -------
    ``batch + (length_t, length_x, dim)``
    """
    length_x = X.shape[-2]
    obs_grid = jnp.linspace(0, 1, length_x)           # (length_x,)
    x_y = obs_grid[:, None] - obs_grid[None, :]       # (length_x, length_x)

    T_mat = cos_exp_kernel(x_y, n_freqs=n_freqs, sigma=sigma)  # (length_x, length_x)

    # Swap last two axes: batch + (length_t, dim, length_x) then matmul with T_mat
    X_t = jnp.swapaxes(X, -1, -2)                             # batch + (length_t, dim, length_x)
    cos_exp_X = (1.0 / length_x) * jnp.matmul(X_t, T_mat)    # batch + (length_t, dim, length_x)

    return jnp.swapaxes(cos_exp_X, -1, -2)                    # batch + (length_t, length_x, dim)


def cos_exp_kernel(x_y: Array, n_freqs: int = 5, sigma: float = 1.0) -> Array:
    """Evaluate the cos-exp kernel.

    Parameters
    ----------
    x_y : Array
        Square matrix with entries ``x_y[i, j] = x_i - y_j``.
    n_freqs : int, default=5
        Number of cosine frequencies.
    sigma : float, default=1.0
        Bandwidth of the Gaussian envelope.

    Returns
    -------
    Array of the same shape as ``x_y``.
    """
    freqs = jnp.arange(n_freqs)                                              # (n_freqs,)
    cos_term = jnp.cos(
        2 * jnp.pi * x_y[:, :, None] * freqs[None, None, :]
    ).sum(axis=-1)                                                            # (length_x, length_x)

    return cos_term * jnp.exp(-x_y ** 2 / sigma)


__all__ = [
    "StaticKernel",
    "LinearKernel",
    "RBFKernel",
    "RBF_CEXP_Kernel",
    "RBF_SQR_Kernel",
    "cexp",
    "cos_exp_kernel",
]
