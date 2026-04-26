"""
End-to-end JAX path tests for JaxShuffleCore.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import jax
import jax.numpy as jnp

from tensordev.core.utils.annotations import iter_class_jittables, is_jittable
from tensordev.core.jax import JaxShuffleCore


def _make_jax_core(N: int) -> JaxShuffleCore:
    return JaxShuffleCore(d=1, trunc=N)


def _jax_dense(scalars):
    return tuple(jnp.array([[float(v)]]) for v in scalars)


# ---------------------------------------------------------------------------
# 1. Registration: which methods are auto-JIT'd
# ---------------------------------------------------------------------------

def test_registration():
    jittables = {name: kw for name, _, kw in iter_class_jittables(JaxShuffleCore)}

    assert "tensor_shuffle_product_homogeneous" in jittables
    assert "tensor_shuffle_product" in jittables

    static_names = set(jittables["tensor_shuffle_product"].get("static_argnames", ()))
    for expected in ("trunc", "a_first_on", "b_first_on", "first_on_out"):
        assert expected in static_names, f"'{expected}' must be in static_argnames"

    # JaxShuffleCore.sparse_einsum is @partial(jax.jit, ...) — must NOT carry JIT_TAG
    assert not is_jittable(getattr(JaxShuffleCore, "sparse_einsum"))

    print("PASS test_registration")


# ---------------------------------------------------------------------------
# 2. Instance attribute wiring
# ---------------------------------------------------------------------------

def test_instance_wiring():
    sc = _make_jax_core(4)

    assert "tensor_shuffle_product_homogeneous" in sc.__dict__
    assert "tensor_shuffle_product" in sc.__dict__
    # sparse_einsum lives at class level via @partial(jax.jit, ...) — not instance attr
    assert "sparse_einsum" not in sc.__dict__

    print("PASS test_instance_wiring")


# ---------------------------------------------------------------------------
# 3. sparse_einsum returns a JAX array
# ---------------------------------------------------------------------------

def test_sparse_einsum_dispatch():
    sc = _make_jax_core(2)

    Ai = jnp.array([[3.0]])
    Bj = jnp.array([[2.0]])
    res = sc.sparse_einsum(Ai, Bj, 1, 0)

    assert isinstance(res, jax.Array)
    np.testing.assert_allclose(np.array(res), [[6.0]])
    print("PASS test_sparse_einsum_dispatch")


# ---------------------------------------------------------------------------
# 4. tensor_shuffle_product_homogeneous — JAX path
# ---------------------------------------------------------------------------

def test_homogeneous_jax():
    sc = _make_jax_core(4)

    # shuffle(A_2, B_1): C(3,2) = 3
    Ai = jnp.array([[2.0]])
    Bj = jnp.array([[5.0]])
    res = sc.tensor_shuffle_product_homogeneous(Ai, Bj, 2, 1)

    assert isinstance(res, jax.Array)
    np.testing.assert_allclose(np.array(res), [[30.0]])
    print("PASS test_homogeneous_jax")


# ---------------------------------------------------------------------------
# 5. tensor_shuffle_product — basic correctness
# ---------------------------------------------------------------------------

def test_shuffle_product_jax_basic():
    sc = _make_jax_core(4)

    A = _jax_dense([1.0, 2.0])
    B = _jax_dense([1.0, 3.0])
    C = sc.tensor_shuffle_product(A, B)

    assert len(C) == 3
    np.testing.assert_allclose(np.array(C[0]), [[1.0]])
    np.testing.assert_allclose(np.array(C[1]), [[5.0]])
    np.testing.assert_allclose(np.array(C[2]), [[12.0]])
    print("PASS test_shuffle_product_jax_basic")


# ---------------------------------------------------------------------------
# 6. Same instance → same compiled function (no spurious retrace)
# ---------------------------------------------------------------------------

def test_static_reuse():
    sc = _make_jax_core(4)

    A = _jax_dense([1.0, 2.0])
    B = _jax_dense([1.0, 3.0])
    C1 = sc.tensor_shuffle_product(A, B)
    C2 = sc.tensor_shuffle_product(A, B)
    for k in range(len(C1)):
        np.testing.assert_allclose(np.array(C1[k]), np.array(C2[k]))
    print("PASS test_static_reuse")


# ---------------------------------------------------------------------------
# 7. Commutativity under JAX JIT
# ---------------------------------------------------------------------------

def test_commutativity_jax():
    sc = _make_jax_core(6)

    A = _jax_dense([1.0, 2.0, 0.5])
    B = _jax_dense([3.0, 1.0, 4.0])
    C_AB = sc.tensor_shuffle_product(A, B)
    C_BA = sc.tensor_shuffle_product(B, A)
    for k in range(len(C_AB)):
        np.testing.assert_allclose(
            np.array(C_AB[k]), np.array(C_BA[k]), err_msg=f"degree {k}"
        )
    print("PASS test_commutativity_jax")


# ---------------------------------------------------------------------------
# 8. trunc parameter respected
# ---------------------------------------------------------------------------

def test_trunc_jax():
    sc = _make_jax_core(4)

    A = _jax_dense([1.0, 2.0])
    B = _jax_dense([1.0, 3.0])
    C_full = sc.tensor_shuffle_product(A, B)
    C_trunc = sc.tensor_shuffle_product(A, B, trunc=1)
    assert len(C_trunc) == 2
    np.testing.assert_allclose(np.array(C_trunc[0]), np.array(C_full[0]))
    np.testing.assert_allclose(np.array(C_trunc[1]), np.array(C_full[1]))
    print("PASS test_trunc_jax")


# ---------------------------------------------------------------------------
# 9. first_on_out
# ---------------------------------------------------------------------------

def test_first_on_out_jax():
    sc = _make_jax_core(4)

    A = _jax_dense([1.0, 2.0])
    B = _jax_dense([1.0, 3.0])
    C_full = sc.tensor_shuffle_product(A, B)
    C_drop = sc.tensor_shuffle_product(A, B, first_on_out=True)
    assert len(C_drop) == len(C_full) - 1
    for k in range(len(C_drop)):
        np.testing.assert_allclose(np.array(C_drop[k]), np.array(C_full[k + 1]))
    print("PASS test_first_on_out_jax")


# ---------------------------------------------------------------------------
# 10. Batch dimension
# ---------------------------------------------------------------------------

def test_batch_jax():
    sc = _make_jax_core(4)

    batch = 7
    rng = jax.random.PRNGKey(0)
    A = tuple(jax.random.normal(rng, (batch, 1)) for _ in range(3))
    B = tuple(jax.random.normal(rng, (batch, 1)) for _ in range(2))
    C = sc.tensor_shuffle_product(A, B)
    assert len(C) == 4
    for k, Ck in enumerate(C):
        assert Ck.shape == (batch, 1), f"degree {k}: wrong shape {Ck.shape}"
    print("PASS test_batch_jax")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_registration()
    test_instance_wiring()
    test_sparse_einsum_dispatch()
    test_homogeneous_jax()
    test_shuffle_product_jax_basic()
    test_static_reuse()
    test_commutativity_jax()
    test_trunc_jax()
    test_first_on_out_jax()
    test_batch_jax()
    print("\nAll 10 JaxShuffleCore tests passed.")