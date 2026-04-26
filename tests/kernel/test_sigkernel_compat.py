from jax import config

config.update("jax_enable_x64", True)

import inspect

import numpy as np
import pytest
import jax.random as jr

import tensordev.kernel.sig as td_sigkernel
from tensordev.kernel.static_kernels import RBFKernel
from random_paths import random_trigonometric_polynomial_paths

REF_SIGKERNEL = pytest.importorskip("sigkernel")
TORCH = pytest.importorskip("torch")


def _reference_signature_kernel(dyadic_order):
    if not hasattr(REF_SIGKERNEL, "SigKernel"):
        pytest.skip("sigkernel.SigKernel not available in this installation")
    if not hasattr(REF_SIGKERNEL, "LinearKernel"):
        pytest.skip("sigkernel.LinearKernel not available in this installation")

    return REF_SIGKERNEL.SigKernel(
        REF_SIGKERNEL.LinearKernel(),
        dyadic_order=dyadic_order,
    )


def _ours_signature_kernel(dyadic_order):

    return td_sigkernel.SigKernel(
        dyadic_order=dyadic_order,
        backend="wavefront",
    )


def _call_with_supported_kwargs(fun, *args, **kwargs):
    """
    Call a sigkernel method while filtering keyword arguments unsupported by the
    installed version.
    """
    params = inspect.signature(fun).parameters
    supported = {k: v for k, v in kwargs.items() if k in params}
    return fun(*args, **supported)


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_compute_kernel_matches_sigkernel(dim, dyadic_order):
    key = jr.PRNGKey(1000 + 10 * dim + dyadic_order)

    X = random_trigonometric_polynomial_paths(
        key,
        batch=4,
        steps=100,
        dim=dim,
    )
    Y = random_trigonometric_polynomial_paths(
        jr.fold_in(key, 1),
        batch=4,
        steps=100,
        dim=dim,
    )

    ours_kernel = _ours_signature_kernel(dyadic_order)
    ref_kernel = _reference_signature_kernel(dyadic_order)

    ours = ours_kernel.compute_kernel(
        X,
        Y,
        max_batch=2,
    )
    ref = _call_with_supported_kwargs(
        ref_kernel.compute_kernel,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(Y), dtype=TORCH.float64),
        max_batch=2,
    )

    np.testing.assert_allclose(
        _to_numpy(ours),
        _to_numpy(ref),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_compute_Gram_matches_sigkernel(dim, dyadic_order):
    key = jr.PRNGKey(2000 + 10 * dim + dyadic_order)

    X = random_trigonometric_polynomial_paths(
        key,
        batch=4,
        steps=36,
        dim=dim,
    )
    Y = random_trigonometric_polynomial_paths(
        jr.fold_in(key, 1),
        batch=5,
        steps=36,
        dim=dim,
    )

    ours_kernel = _ours_signature_kernel(dyadic_order)
    ref_kernel = _reference_signature_kernel(dyadic_order)

    ours = ours_kernel.compute_Gram(
        X,
        Y,
        sym=False,
        max_batch=2,
    )
    ref = _call_with_supported_kwargs(
        ref_kernel.compute_Gram,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(Y), dtype=TORCH.float64),
        sym=False,
        max_batch=2,
    )

    np.testing.assert_allclose(
        _to_numpy(ours),
        _to_numpy(ref),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_compute_Gram_symmetric_matches_sigkernel(dim, dyadic_order):
    key = jr.PRNGKey(3000 + 10 * dim + dyadic_order)

    X = random_trigonometric_polynomial_paths(
        key,
        batch=5,
        steps=32,
        dim=dim,
    )

    ours_kernel = _ours_signature_kernel(dyadic_order)
    ref_kernel = _reference_signature_kernel(dyadic_order)

    ours = ours_kernel.compute_Gram(
        X,
        sym=True,
        max_batch=2,
    )
    ref = _call_with_supported_kwargs(
        ref_kernel.compute_Gram,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        sym=True,
        max_batch=2,
    )

    np.testing.assert_allclose(
        _to_numpy(ours),
        _to_numpy(ref),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_compute_mmd_matches_sigkernel(dyadic_order):
    key = jr.PRNGKey(4000 + dyadic_order)

    X = random_trigonometric_polynomial_paths(
        key,
        batch=4,
        steps=32,
        dim=2,
    )
    Y = random_trigonometric_polynomial_paths(
        jr.fold_in(key, 1),
        batch=5,
        steps=32,
        dim=2,
    )

    ours_kernel = _ours_signature_kernel(dyadic_order)
    ref_kernel = _reference_signature_kernel(dyadic_order)

    if not hasattr(ref_kernel, "compute_mmd"):
        pytest.skip("sigkernel compute_mmd API not available in this installation")

    ours = ours_kernel.compute_mmd(
        X,
        Y,
        max_batch=2,
    )
    ref = _call_with_supported_kwargs(
        ref_kernel.compute_mmd,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(Y), dtype=TORCH.float64),
        max_batch=2,
    )

    np.testing.assert_allclose(
        _to_numpy(ours),
        _to_numpy(ref),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_compute_scoring_rule_matches_sigkernel(dyadic_order):
    key = jr.PRNGKey(5000 + dyadic_order)

    X = random_trigonometric_polynomial_paths(
        key,
        batch=4,
        steps=32,
        dim=2,
    )
    y = random_trigonometric_polynomial_paths(
        jr.fold_in(key, 1),
        batch=1,
        steps=32,
        dim=2,
    )

    ours_kernel = _ours_signature_kernel(dyadic_order)
    ref_kernel = _reference_signature_kernel(dyadic_order)

    if not hasattr(ref_kernel, "compute_scoring_rule"):
        pytest.skip("sigkernel compute_scoring_rule API not available in this installation")

    ours = ours_kernel.compute_scoring_rule(
        X,
        y,
        max_batch=2,
    )
    ref = _call_with_supported_kwargs(
        ref_kernel.compute_scoring_rule,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(y), dtype=TORCH.float64),
        max_batch=2,
    )

    np.testing.assert_allclose(
        _to_numpy(ours),
        _to_numpy(ref),
        rtol=1e-8,
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# Static-kernel tests (RBF, RBF_SQR)
# ---------------------------------------------------------------------------

# Each entry: (our_kernel_factory(sigma), ref_class_name, kwargs)
_STATIC_KERNEL_CASES = [
    pytest.param(
        lambda sigma: RBFKernel(sigma=sigma),
        "RBFKernel",
        dict(sigma=1.0),
        id="RBF-sigma1",
    ),
    pytest.param(
        lambda sigma: RBFKernel(sigma=sigma),
        "RBFKernel",
        dict(sigma=0.5),
        id="RBF-sigma0.5",
    ),
]


def _ref_static_kernel(ref_class_name, sigma):
    ref_cls = getattr(REF_SIGKERNEL, ref_class_name, None)
    if ref_cls is None:
        pytest.skip(f"sigkernel.{ref_class_name} not available")
    return ref_cls(sigma)


def _ours_sig_kernel_static(our_static_kernel, dyadic_order):
    return td_sigkernel.SigKernel(
        dyadic_order=dyadic_order,
        backend="wavefront",
        static_kernel=our_static_kernel,
    )


def _ref_sig_kernel_static(ref_class_name, dyadic_order, sigma):
    ref_static = _ref_static_kernel(ref_class_name, sigma)
    if not hasattr(REF_SIGKERNEL, "SigKernel"):
        pytest.skip("sigkernel.SigKernel not available")
    return REF_SIGKERNEL.SigKernel(ref_static, dyadic_order=dyadic_order)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("our_factory,ref_class_name,kw", _STATIC_KERNEL_CASES)
def test_static_kernel_compute_kernel(dim, dyadic_order, our_factory, ref_class_name, kw):
    key = jr.PRNGKey(7000 + 10 * dim + dyadic_order)
    X = random_trigonometric_polynomial_paths(key, batch=4, steps=50, dim=dim)
    Y = random_trigonometric_polynomial_paths(jr.fold_in(key, 1), batch=4, steps=50, dim=dim)

    sigma = kw["sigma"]
    ours = _ours_sig_kernel_static(our_factory(sigma), dyadic_order).compute_kernel(X, Y, max_batch=2)
    ref = _call_with_supported_kwargs(
        _ref_sig_kernel_static(ref_class_name, dyadic_order, sigma).compute_kernel,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(Y), dtype=TORCH.float64),
        max_batch=2,
    )
    np.testing.assert_allclose(_to_numpy(ours), _to_numpy(ref), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("our_factory,ref_class_name,kw", _STATIC_KERNEL_CASES)
def test_static_kernel_compute_Gram(dim, dyadic_order, our_factory, ref_class_name, kw):
    key = jr.PRNGKey(8000 + 10 * dim + dyadic_order)
    X = random_trigonometric_polynomial_paths(key, batch=4, steps=30, dim=dim)
    Y = random_trigonometric_polynomial_paths(jr.fold_in(key, 1), batch=5, steps=30, dim=dim)

    sigma = kw["sigma"]
    ours = _ours_sig_kernel_static(our_factory(sigma), dyadic_order).compute_Gram(X, Y, sym=False, max_batch=2)
    ref = _call_with_supported_kwargs(
        _ref_sig_kernel_static(ref_class_name, dyadic_order, sigma).compute_Gram,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(Y), dtype=TORCH.float64),
        sym=False,
        max_batch=2,
    )
    np.testing.assert_allclose(_to_numpy(ours), _to_numpy(ref), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("our_factory,ref_class_name,kw", _STATIC_KERNEL_CASES)
def test_static_kernel_compute_Gram_symmetric(dim, dyadic_order, our_factory, ref_class_name, kw):
    key = jr.PRNGKey(9000 + 10 * dim + dyadic_order)
    X = random_trigonometric_polynomial_paths(key, batch=5, steps=28, dim=dim)

    sigma = kw["sigma"]
    ours = _ours_sig_kernel_static(our_factory(sigma), dyadic_order).compute_Gram(X, sym=True, max_batch=2)
    ref = _call_with_supported_kwargs(
        _ref_sig_kernel_static(ref_class_name, dyadic_order, sigma).compute_Gram,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        sym=True,
        max_batch=2,
    )
    np.testing.assert_allclose(_to_numpy(ours), _to_numpy(ref), rtol=1e-7, atol=1e-7)
    key = jr.PRNGKey(6000 + dyadic_order)

    X = random_trigonometric_polynomial_paths(
        key,
        batch=4,
        steps=32,
        dim=2,
    )
    Y = random_trigonometric_polynomial_paths(
        jr.fold_in(key, 1),
        batch=5,
        steps=32,
        dim=2,
    )

    ours_kernel = _ours_signature_kernel(dyadic_order)
    ref_kernel = _reference_signature_kernel(dyadic_order)

    if not hasattr(ref_kernel, "compute_expected_scoring_rule"):
        pytest.skip("sigkernel compute_expected_scoring_rule API not available in this installation")

    ours = ours_kernel.compute_expected_scoring_rule(
        X,
        Y,
        max_batch=2,
    )
    ref = _call_with_supported_kwargs(
        ref_kernel.compute_expected_scoring_rule,
        TORCH.as_tensor(np.asarray(X), dtype=TORCH.float64),
        TORCH.as_tensor(np.asarray(Y), dtype=TORCH.float64),
        max_batch=2,
    )

    np.testing.assert_allclose(
        _to_numpy(ours),
        _to_numpy(ref),
        rtol=1e-8,
        atol=1e-8,
    )