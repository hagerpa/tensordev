"""
helpers.py — Shared parameter-generation helpers for the validation scripts.
"""

import numpy as np
from scipy.linalg import expm


def random_fssk(
    *,
    q: int,
    R: int,
    m: int,
    d: int,
    seed: int = 0,
    eig_min: float = 1.0,
    eig_max: float = 2.0,
    jordan_alpha: float = 0.25,
    spectral_radius: float | None = None,
    normalise_b: bool = True,
    dtype=np.float64,
):
    """Return random (Lambda, A, b) parameters for an FSSK kernel.

    Lambda : (R, R)  near-diagonal matrix with random eigenvalues in
             [eig_min, eig_max] and a small Jordan super-diagonal.
             If *spectral_radius* is not None, Lambda is rescaled to that
             spectral radius (default: no rescaling).

    A      : (q, m, d)  semi-orthogonal projection matrices (rows or
             columns orthonormal depending on m vs d).

    b      : (q, R)  random coefficients.  If *normalise_b* is True
             (default), b is rescaled so that

                 ‖e^{-Λ} bᵀ‖_F = 1,

             i.e. the output weight at unit lag has unit Frobenius norm.
             This ties the b-scale to Λ so that different seeds produce
             kernels with comparable absolute output magnitudes.

    Parameters
    ----------
    spectral_radius : float or None
        If given, rescale Lambda to this spectral radius before normalising b.
    normalise_b : bool
        Normalise b as described above.  Default True.
    """
    rng = np.random.default_rng(seed)

    eigs = rng.uniform(eig_min, eig_max, size=R)
    J = np.diag(eigs)
    for i in range(R - 1):
        J[i, i + 1] = jordan_alpha

    G = rng.normal(size=(R, R))
    Q, _ = np.linalg.qr(G)
    Lambda = Q @ J @ Q.T

    if spectral_radius is not None:
        sr = np.max(np.abs(np.linalg.eigvals(Lambda)))
        if sr > 0:
            Lambda *= spectral_radius / sr

    A = np.empty((q, m, d), dtype=dtype)
    for p in range(q):
        if m <= d:
            G = rng.normal(size=(d, m))
            Qp, _ = np.linalg.qr(G)
            A[p] = Qp.T
        else:
            G = rng.normal(size=(m, d))
            Qp, _ = np.linalg.qr(G)
            A[p] = Qp

    b = rng.normal(size=(q, R))

    if normalise_b:
        # e^{-Λ} bᵀ  has shape (R, q); normalise b so its Frobenius norm = 1.
        #eb = expm(-Lambda) @ b.T      # (R, q)
        #frob = np.linalg.norm(eb)
        #if frob > 0:
        #    b /= frob
        L = np.linalg.norm(Lambda, ord=2)
        bnorm = np.linalg.norm(b, ord="fro")

        if bnorm > 0:
            b *= L / bnorm
    else:
        # Legacy column-wise L1 normalisation.
        b /= np.sum(np.abs(b), axis=0, keepdims=True)

    return Lambda.astype(dtype), A.astype(dtype), b.astype(dtype)

