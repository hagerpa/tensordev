r"""Reference values of :math:`\varphi_k(-hM)=\sum_{j\ge0}(-hM)^j/(j+k)!` and their derivatives.

:math:`\varphi_k(-hM)`, with :math:`\varphi_0=\exp`, is the top-right :math:`R\times R` block of
the exponential of the :math:`(k+1)R` square block matrix with :math:`-hM` in the top-left block
and identities on the first block superdiagonal.
"""
import numpy as np
from scipy.linalg import expm, expm_frechet

KINDS = [("expm", 0), ("phi1", 1), ("phi2", 2)]


def phi(matrix, dt, order, matrix_direction=None, dt_direction=0.0):
    r""":math:`\varphi_k(-dt\,M)` for ``dt`` of shape ``()`` or ``(m,)``.

    With ``matrix_direction`` or ``dt_direction`` (scalar ``dt`` only), also return the derivative
    in :math:`s` at 0 of :math:`\varphi_k(-(dt+s\,\delta t)(M+s\,\delta M))`.
    """
    matrix = np.asarray(matrix, dtype=float)
    if np.ndim(dt):
        return np.stack([phi(matrix, h, order) for h in np.asarray(dt)])
    size = matrix.shape[0]
    augmented = np.kron(np.eye(order + 1, k=1), np.eye(size))
    augmented[:size, :size] = -dt * matrix
    block = (slice(0, size), slice(order * size, None))
    if matrix_direction is None and dt_direction == 0:
        return expm(augmented)[block]
    direction = np.zeros_like(augmented)
    direction[:size, :size] = -dt_direction * matrix
    if matrix_direction is not None:
        direction[:size, :size] -= dt * np.asarray(matrix_direction)
    value, derivative = expm_frechet(augmented, direction)
    return value[block], derivative[block]
