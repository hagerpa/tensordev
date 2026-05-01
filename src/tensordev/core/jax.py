# ---- JAX backend ----
from __future__ import annotations

import types
from functools import partial

import jax
from jax import lax
from jax import numpy as jnp

from tensordev.core.utils.annotations import iter_class_jittables
from .einsum import Einsum
from .sequential import SequentialCore, DenseElem
from .shuffle import ShuffleCore
from .universal import *

JAX_JIT_PARAMETERS = {
    "static_argnums", "static_argnames", "donate_argnums", "device",
    "backend", "inline", "abstracted_axes", "keep_unused",
}


class Jax(Einsum[jnp.ndarray]):

    def __init__(self):
        super().__init__(jnp)

        # Cache for JAX-converted shuffle operators: (d, i, j) -> (meta, jax_arrays)
        self._jax_operators_cache = {}

        for name, fn, kw in iter_class_jittables(type(self)):
            jax_kw = {k: v for k, v in kw.items() if k in JAX_JIT_PARAMETERS}
            setattr(self, name, types.MethodType(jax.jit(fn, **jax_kw), self))

    def _mapper(self, fun):
        vmap = jax.vmap(lambda *x: fun(x))
        return lambda x: vmap(*x)

    def _reducer(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],  # (acc, step) -> acc
            *,
            neutral: DenseElem,  # guaranteed; API symmetry
            seed: DenseElem,  # guaranteed
            associative: bool = False,
    ):
        if not associative:
            @jax.jit
            def reduce_fn(X):
                def body(carry, step):
                    return fun(carry, step), None

                final, _ = lax.scan(body, seed, X)
                return final

            return reduce_fn

        @jax.jit
        def reduce_fn(X):
            prefixes = lax.associative_scan(fun, X, axis=0)  # [x0, x0⊕x1, ...]
            last = jax.tree.map(lambda a: a[-1], prefixes)  # t = S-1
            return fun(seed, last)  # seed ⊕ (x0⊕...⊕x_{S-1})

        return reduce_fn

    def _accumulator(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],  # (carry, step) -> y ; we take carry' := y
            *,
            neutral: DenseElem,  # guaranteed; API symmetry
            seed: DenseElem,  # guaranteed
            associative: bool = False,
    ):
        if not associative:
            @jax.jit
            def scan_fn(X):
                def body(carry, step):
                    y = fun(carry, step)
                    return y, y

                final, ys_stacked = lax.scan(body, seed, X)
                return final, ys_stacked

            return scan_fn

        @jax.jit
        def scan_fn(X):
            prefixes = lax.associative_scan(fun, X, axis=0)  # [x0, x0⊕x1, ...]
            # Broadcast seed across time via vmap over the leading axis of the pytree:
            apply_seed = jax.vmap(lambda p: fun(seed, p))
            ys_stacked = apply_seed(prefixes)  # [seed⊕p_t]
            final = jax.tree.map(lambda a: a[-1], ys_stacked)
            return final, ys_stacked

        return scan_fn

    def _get_jax_operator(self, d: int, i: int, j: int):
        """
        Convert numpy shuffle operator to JAX arrays, with caching.

        Parameters
        ----------
        d : int
            Base dimension.
        i, j : int
            Operator degrees.

        Returns
        -------
        tuple
            ``(meta, (segment_ids, rows, cols, data))`` with JAX arrays.

        Notes
        -----
        Conversion happens once per ``(d, i, j)`` outside of JIT trace.
        Cached JAX arrays become compile-time constants when this method is
        called with static ``d, i, j`` from within a JIT-compiled function.
        """
        key = (d, i, j)
        if key not in self._jax_operators_cache:
            # Retrieve numpy operator from parent cache
            np_operator = self._shuffle_cache[d].operators[(i, j)]
            meta, (seg, rows, cols, data) = np_operator
            # Convert to JAX arrays and cache
            self._jax_operators_cache[key] = (
                meta,
                (
                    jnp.asarray(seg),
                    jnp.asarray(rows),
                    jnp.asarray(cols),
                    jnp.asarray(data),
                ),
            )
        return self._jax_operators_cache[key]

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, d, i, j are static
    def sparse_einsum(self, Ai, Bj, d: int, i: int, j: int):
        """
        JAX-optimized sparse bilinear product.

        Parameters
        ----------
        Ai : jnp.ndarray
            Left factor with shape ``(batch, d**i)``.
        Bj : jnp.ndarray
            Right factor with shape ``(batch, d**j)``.
        d : int (static)
            Base dimension of the tensor algebra.
        i : int (static)
            Degree of ``Ai`` (must satisfy ``i >= j``).
        j : int (static)
            Degree of ``Bj``.

        Returns
        -------
        jnp.ndarray
            Result with shape ``(batch, d**(i+j))``.

        Notes
        -----
        The operator arrays are retrieved via ``_get_jax_operator`` at trace time.
        Since ``d, i, j`` are static arguments, the operator arrays become
        compile-time constants in the JIT-compiled function.
        """
        # Retrieve JAX operator (happens once at trace time with static d, i, j)
        _, Q = self._get_jax_operator(d, i, j)
        idx, idy, idz, data = Q

        def single_batch_op(Ai_s, Bj_s):
            vals = data * Ai_s[idy] * Bj_s[idz]
            return jax.ops.segment_sum(vals, idx, num_segments=Ai.shape[1]*Bj.shape[1])

        # Use vmap to handle batch dimension
        return jax.vmap(single_batch_op)(Ai, Bj)


class JaxShuffleCore(ShuffleCore[jnp.ndarray]):
    """
    JAX-optimized shuffle product engine.

    Operators are precomputed as numpy arrays at construction, then immediately
    converted to JAX arrays.  All ``@dummy_jit``-decorated methods are
    auto-compiled with ``jax.jit`` using the same pattern as ``Jax``.
    """

    def __init__(self, d: int, trunc: int) -> None:
        super().__init__(jnp, d, trunc)
        # Convert numpy operators to JAX arrays once, fusing (rows, cols) into a
        # single flat index into the outer product (shape d^i × d^j) so that
        # sparse_einsum needs only one gather instead of two.
        self._jax_operators: dict = {
            (i, j): (
                meta,
                (
                    jnp.asarray(segment_ids),
                    jnp.asarray(rows * (d ** j) + cols),  # combined flat index
                    jnp.asarray(data),
                ),
            )
            for (i, j), (meta, (segment_ids, rows, cols, data)) in self.operators.items()
        }
        for name, fn, kw in iter_class_jittables(type(self)):
            jax_kw = {k: v for k, v in kw.items() if k in JAX_JIT_PARAMETERS}
            setattr(self, name, types.MethodType(jax.jit(fn, **jax_kw), self))

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def sparse_einsum(self, Ai, Bj, i: int, j: int):
        """JAX override: outer-product first, then one gather + segment_sum."""
        _, Q = self._jax_operators[(i, j)]
        idx, combined_ij, data = Q

        batch = jnp.broadcast_shapes(Ai.shape[:-1], Bj.shape[:-1])
        d_i, d_j = Ai.shape[-1], Bj.shape[-1]
        out_size = d_i * d_j

        Ai_flat = jnp.broadcast_to(Ai, batch + (d_i,)).reshape(-1, d_i)
        Bj_flat = jnp.broadcast_to(Bj, batch + (d_j,)).reshape(-1, d_j)
        B = Ai_flat.shape[0]

        # One fused XLA op replaces two length-nnz gathers.
        P = (Ai_flat[:, :, None] * Bj_flat[:, None, :]).reshape(B, -1)  # (B, d^{i+j})
        weighted = P[:, combined_ij] * data                               # (B, nnz)
        out_flat = jax.vmap(lambda w: jax.ops.segment_sum(w, idx, num_segments=out_size))(weighted)
        return out_flat.reshape(batch + (out_size,))


class JaxSequentialCore(SequentialCore[jnp.ndarray]):
    """
    JAX-native SequentialCore.

    Replaces every Python loop in the base class with a JAX primitive:
      - _mapper    → jax.vmap         (parallel map over the block axis)
      - _reducer   → lax.scan  or  lax.associative_scan  (in_tree=True)
      - _accumulator → lax.scan  or  lax.associative_scan + vmap seed broadcast
    """

    def __init__(self, default_time_axis: int = -2):
        super().__init__(jnp, default_time_axis)
        for name, fn, kw in iter_class_jittables(type(self)):
            jax_kw = {k: v for k, v in kw.items() if k in JAX_JIT_PARAMETERS}
            setattr(self, name, types.MethodType(jax.jit(fn, **jax_kw), self))

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def _mapper(self, fun: Callable[[DenseElem], DenseElem]) -> Callable[[DenseElem], DenseElem]:
        """vmap fun over the leading (block) axis of every level in the DenseElem."""
        vmapped = jax.vmap(fun)
        return vmapped

    def _reducer(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
            in_tree: bool = False,
    ) -> Callable[[DenseElem], DenseElem]:
        if not in_tree:
            @jax.jit
            def reduce_fn(X):
                def body(carry, step):
                    return fun(carry, step), None
                final, _ = lax.scan(body, seed, X)
                return final
            return reduce_fn

        @jax.jit
        def reduce_fn(X):
            # Parallel prefix, then fold seed into the global result.
            prefixes = lax.associative_scan(fun, X, axis=0)
            last = jax.tree.map(lambda a: a[-1], prefixes)
            return fun(seed, last)
        return reduce_fn

    def _accumulator(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
            in_tree: bool = False,
    ) -> Callable[[DenseElem], tuple[DenseElem, DenseElem]]:
        if not in_tree:
            @jax.jit
            def scan_fn(X):
                def body(carry, step):
                    y = fun(carry, step)
                    return y, y
                final, ys = lax.scan(body, seed, X)
                return final, ys
            return scan_fn

        @jax.jit
        def scan_fn(X):
            # All-parallel prefix scan; then broadcast seed into every prefix.
            prefixes = lax.associative_scan(fun, X, axis=0)
            ys = jax.vmap(lambda p: fun(seed, p))(prefixes)
            final = jax.tree.map(lambda a: a[-1], ys)
            return final, ys
        return scan_fn

