# ---- JAX backend ----
from __future__ import annotations

import types
from functools import partial

import jax
from jax import lax
from jax import numpy as jnp

from tensordev.core.utils.annotations import iter_class_jittables
from .einsum import Einsum
from .universal import *

JAX_JIT_PARAMETERS = {
    "static_argnums", "static_argnames", "donate_argnums", "device",
    "backend", "inline", "abstracted_axes", "keep_unused",
}


class Jax(Einsum[jnp.ndarray]):

    def __init__(self):
        super().__init__(jnp)

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

    @partial(jax.jit, static_argnums=(0,))
    def sparse_einsum(self, Ai, Bj, operator):
        # Unpack operator
        _, Q = operator
        idx, idy, idz, data = Q

        def single_batch_op(Ai_s, Bj_s):
            vals = data * Ai_s[idy] * Bj_s[idz]
            return jax.ops.segment_sum(vals, idx, num_segments=Ai.shape[1]*Bj.shape[1])

        # Use vmap to handle batch dimension
        return jax.vmap(single_batch_op)(Ai, Bj)
