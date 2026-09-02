# ---- JAX backend ----
from __future__ import annotations

import types
from dataclasses import dataclass
from functools import lru_cache, partial
from math import comb

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

from tensordev.core.utils.annotations import iter_class_jittables
from .einsum import Einsum
from .sequential import SequentialCore, DenseElem
from .shuffle import (
    TotalDegreeShufflePlanStore,
    _axis_dtype,
    _direct_homogeneous_shuffle,
    _homogeneous_outer_product,
    _non_negative_int,
    _normalize_precompute_shuffle,
    _prepare_homogeneous_shuffle_inputs,
    _sum_homogeneous_axis_permutations,
)
from .universal import *
from .utils.pytrees import PyTree, tree_index

JAX_JIT_PARAMETERS = {
    "static_argnums", "static_argnames", "donate_argnums", "device",
    "backend", "inline", "abstracted_axes", "keep_unused",
}


@lru_cache(maxsize=None)
def _compiled_jittables(core_type: type):
    """Compile unbound wrappers once per concrete JAX core class."""
    compiled = []
    compiled_by_function = {}
    for name, function, kwargs in iter_class_jittables(core_type):
        jax_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in JAX_JIT_PARAMETERS
        }
        compiled_function = compiled_by_function.get(function)
        if compiled_function is None:
            compiled_function = jax.jit(function, **jax_kwargs)
            compiled_by_function[function] = compiled_function
        compiled.append((name, compiled_function))
    return tuple(compiled)


class Jax(Einsum[jnp.ndarray]):

    def __init__(
            self,
            *,
            d: int | None = None,
            max_trunc: int | None = None,
            default_trunc: int | None = None,
            precompute_shuffle: bool = False,
            shuffle_plan_store=None,
    ):
        shuffle_scope = _normalize_precompute_shuffle(
            precompute_shuffle,
            allow_generator=False,
        )
        if shuffle_plan_store is not None:
            if shuffle_scope != "none":
                raise ValueError(
                    "precompute_shuffle and shuffle_plan_store are mutually "
                    "exclusive."
                )
            if not isinstance(
                shuffle_plan_store,
                JaxTotalDegreeShufflePlanStore,
            ):
                raise TypeError(
                    "shuffle_plan_store must be a JAX shuffle plan store, "
                    f"got {type(shuffle_plan_store).__name__}."
                )

        # Universal owns and validates the bounded total-degree configuration.
        # Passing an existing store lets it infer omitted bounds and reject
        # explicit bounds that disagree with the shared store.
        super().__init__(
            jnp,
            d=d,
            max_trunc=max_trunc,
            default_trunc=default_trunc,
            shuffle_plan_store=shuffle_plan_store,
        )
        if shuffle_scope == "full":
            if self.d is None or self.max_truncation is None:
                raise ValueError(
                    "precompute_shuffle requires both d and max_trunc."
                )
            self.shuffle_plan_store = JaxTotalDegreeShufflePlanStore(
                self.d,
                self.max_truncation,
            )

        for name, function in _compiled_jittables(type(self)):
            setattr(self, name, types.MethodType(function, self))

    def plan_strategy(self, i: int, j: int) -> str:
        """Return the execution strategy for one canonical shuffle pair."""
        return self._require_shuffle().plan_strategy(i, j)

    def execution_plan_memory_bytes(
            self,
            i: int | None = None,
            j: int | None = None,
    ) -> int:
        """Return JAX-only shuffle execution metadata memory."""
        return self._require_shuffle().execution_plan_memory_bytes(i, j)

    def _mapper(self, fun):
        vmap = jax.vmap(lambda *x: fun(x))
        return lambda x: vmap(*x)

    def _reducer(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
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
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
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


def total_degree_core(
        *,
        d: int,
        max_trunc: int,
        default_trunc: int | None = None,
        precompute_shuffle: bool = False,
) -> Jax:
    """Construct a bounded ordinary total-degree JAX core.

    Shuffle plans are optional and live on this same algebra core.  Omitting
    ``default_trunc`` makes the full capacity active, matching
    :func:`bigraded_core`.
    """
    return Jax(
        d=d,
        max_trunc=max_trunc,
        default_trunc=default_trunc,
        precompute_shuffle=precompute_shuffle,
    )


@dataclass(frozen=True, slots=True)
class _JaxShuffleRuntimePlan:
    """JAX-only execution metadata derived from an authoritative plan."""

    strategy: str
    valid_masks: object | None = None
    flat_indices: object | None = None
    coefficients: object | None = None
    digit_divisors: object | None = None

    def memory_bytes(self) -> int:
        return sum(
            int(array.nbytes)
            for array in (
                self.valid_masks,
                self.flat_indices,
                self.coefficients,
                self.digit_divisors,
            )
            if array is not None
        )

    def memory_bytes_by_category(self) -> dict[str, int]:
        def nbytes(array) -> int:
            return 0 if array is None else int(array.nbytes)

        return {
            "derived_execution_maps": nbytes(self.flat_indices),
            "derived_execution_coefficients": (
                nbytes(self.coefficients) + nbytes(self.digit_divisors)
            ),
            "derived_execution_masks": nbytes(self.valid_masks),
        }


class JaxTotalDegreeShufflePlanStore(TotalDegreeShufflePlanStore):
    """Internal JAX execution store for bounded total-degree shuffles.

    Small plans are emitted as a static sum of transposes. A materialized flat
    gather map executes in one gather-and-reduce when it fits the byte cap;
    beyond that cap, compact base-``d`` coefficients generate fixed-size
    gather chunks inside ``lax.scan``.
    """

    STATIC_PERMUTATION_THRESHOLD = 24
    FLAT_GATHER_CHUNK_SIZE = 24
    COEFFICIENT_GATHER_CHUNK_SIZE = 8
    FLAT_GATHER_MAX_BYTES = 256 * 1024
    FULL_GATHER_MAX_TEMPORARY_BYTES = 512 * 1024

    @classmethod
    def _runtime_plan_layout(
        cls,
        *,
        uses_direct_scaling: bool,
        permutation_count: int,
        output_width: int,
    ) -> tuple[str, int, int]:
        """Return ``(strategy, padded_count, flat_map_bytes)``."""
        if uses_direct_scaling:
            return "direct", 0, 0
        if permutation_count <= cls.STATIC_PERMUTATION_THRESHOLD:
            return "transpose", 0, 0
        flat_chunk_size = cls.FLAT_GATHER_CHUNK_SIZE
        padded_count = (
            (permutation_count + flat_chunk_size - 1) // flat_chunk_size
        ) * flat_chunk_size
        flat_map_bytes = (
            padded_count * output_width * np.dtype(np.int32).itemsize
        )
        if flat_map_bytes <= cls.FLAT_GATHER_MAX_BYTES:
            return "flat_gather", padded_count, flat_map_bytes

        coefficient_chunk_size = cls.COEFFICIENT_GATHER_CHUNK_SIZE
        padded_count = (
            (permutation_count + coefficient_chunk_size - 1)
            // coefficient_chunk_size
        ) * coefficient_chunk_size
        return "coefficient_gather", padded_count, flat_map_bytes

    @classmethod
    def _expected_memory_bytes_by_category(
        cls,
        d: int,
        max_trunc: int,
    ) -> dict[str, int]:
        """Predict every eagerly stored plan buffer without allocating plans."""
        d = _non_negative_int(d, name="d")
        if d == 0:
            raise ValueError("d must be strictly positive, got 0.")
        max_trunc = _non_negative_int(max_trunc, name="max_trunc")

        categories = {
            "axis_permutations": 0,
            "derived_execution_maps": 0,
            "derived_execution_coefficients": 0,
            "derived_execution_masks": 0,
        }
        index_itemsize = np.dtype(np.int32).itemsize
        mask_itemsize = np.dtype(np.bool_).itemsize
        for total_degree in range(max_trunc + 1):
            axis_itemsize = np.dtype(
                _axis_dtype(max(total_degree - 1, 0))
            ).itemsize
            for left_degree in range(total_degree, -1, -1):
                right_degree = total_degree - left_degree
                if right_degree > left_degree:
                    break
                permutation_count = comb(total_degree, left_degree)
                strategy, padded_count, flat_map_bytes = cls._runtime_plan_layout(
                    uses_direct_scaling=(d == 1 or right_degree == 0),
                    permutation_count=permutation_count,
                    output_width=d**total_degree,
                )
                if strategy == "direct":
                    continue

                categories["axis_permutations"] += (
                    permutation_count * total_degree * axis_itemsize
                )
                if strategy == "transpose":
                    continue

                categories["derived_execution_masks"] += (
                    padded_count * mask_itemsize
                )
                if strategy == "flat_gather":
                    categories["derived_execution_maps"] += flat_map_bytes
                else:
                    categories["derived_execution_coefficients"] += (
                        padded_count * total_degree * index_itemsize
                        + total_degree * index_itemsize
                    )
        return categories

    def __init__(self, d: int, max_trunc: int) -> None:
        super().__init__(d, max_trunc)
        self._runtime_plans = {
            degrees: self._build_runtime_plan(plan)
            for degrees, plan in self.plans.items()
        }

    def _build_runtime_plan(self, plan) -> _JaxShuffleRuntimePlan:
        strategy, padded_count, _ = self._runtime_plan_layout(
            uses_direct_scaling=plan.uses_direct_scaling,
            permutation_count=plan.permutation_count,
            output_width=plan.output_width,
        )
        if strategy in ("direct", "transpose"):
            return _JaxShuffleRuntimePlan(strategy)
        chunk_size = (
            self.FLAT_GATHER_CHUNK_SIZE
            if strategy == "flat_gather"
            else self.COEFFICIENT_GATHER_CHUNK_SIZE
        )
        valid = np.arange(padded_count) < plan.permutation_count
        valid = valid.reshape(-1, chunk_size)

        # If output digits are q_k, transposing by permutation p gathers the
        # raw outer product at sum_k q_k d^(n - 1 - p_k).
        permutations = np.asarray(plan.axis_permutations, dtype=np.int64)
        coefficients = np.asarray(
            plan.dimension ** (plan.output_degree - 1 - permutations),
            dtype=np.int32,
        )
        divisors = np.asarray(
            plan.dimension
            ** np.arange(plan.output_degree - 1, -1, -1, dtype=np.int64),
            dtype=np.int32,
        )
        if strategy == "flat_gather":
            positions = np.arange(plan.output_width, dtype=np.int32)
            digits = (positions[:, None] // divisors[None, :]) % plan.dimension
            flat_indices = np.zeros(
                (padded_count, plan.output_width),
                dtype=np.int32,
            )
            flat_indices[:plan.permutation_count] = (
                digits @ coefficients.T
            ).T
            return _JaxShuffleRuntimePlan(
                strategy="flat_gather",
                valid_masks=jnp.asarray(valid),
                flat_indices=jnp.asarray(
                    flat_indices.reshape(-1, chunk_size, plan.output_width)
                ),
            )

        padded_coefficients = np.zeros(
            (padded_count, plan.output_degree),
            dtype=np.int32,
        )
        padded_coefficients[:plan.permutation_count] = coefficients
        return _JaxShuffleRuntimePlan(
            strategy="coefficient_gather",
            valid_masks=jnp.asarray(valid),
            coefficients=jnp.asarray(
                padded_coefficients.reshape(
                    -1,
                    chunk_size,
                    plan.output_degree,
                )
            ),
            digit_divisors=jnp.asarray(divisors),
        )

    def plan_strategy(self, i: int, j: int) -> str:
        """Return the selected JAX execution strategy for one degree pair."""
        self.plan(i, j)
        return self._runtime_plans[(i, j)].strategy

    def execution_plan_memory_bytes(
        self,
        i: int | None = None,
        j: int | None = None,
    ) -> int:
        """Bytes occupied by JAX-only maps, coefficients, and validity masks."""
        if (i is None) != (j is None):
            raise ValueError("i and j must either both be provided or both omitted.")
        if i is not None:
            self.plan(i, j)
            return self._runtime_plans[(i, j)].memory_bytes()
        return sum(
            plan.memory_bytes() for plan in self._runtime_plans.values()
        )

    def memory_bytes_by_category(self) -> dict[str, int]:
        """Plan payload split into authoritative and derived JAX metadata."""
        categories = dict(super().memory_bytes_by_category())
        categories.update(
            {
                "derived_execution_maps": 0,
                "derived_execution_coefficients": 0,
                "derived_execution_masks": 0,
            }
        )
        for runtime_plan in self._runtime_plans.values():
            derived = runtime_plan.memory_bytes_by_category()
            for name, value in derived.items():
                categories[name] += value
        return categories

    def memory_bytes(self) -> int:
        """Total plan memory, including materialized JAX execution metadata."""
        return sum(self.memory_bytes_by_category().values())

    def plan_statistics(self) -> dict[str, int]:
        """Extend the shared plan summary with JAX execution strategies."""
        statistics = super().plan_statistics()
        strategies = tuple(
            plan.strategy for plan in self._runtime_plans.values()
        )
        statistics.update(
            {
                "transpose_plan_count": strategies.count("transpose"),
                "flat_gather_plan_count": strategies.count("flat_gather"),
                "coefficient_gather_plan_count": strategies.count(
                    "coefficient_gather"
                ),
                "derived_execution_bytes": self.execution_plan_memory_bytes(),
            }
        )
        return statistics

    @partial(jax.jit, static_argnums=(0, 1, 4, 5))
    def apply(self, xp, Ai, Bj, i: int, j: int):
        """Apply one homogeneous plan with the selected JAX strategy."""
        if xp is not jnp:
            raise TypeError("JAX shuffle plans can only be applied with jax.numpy.")
        plan = self.plan(i, j)
        runtime_plan = self._runtime_plans[(i, j)]
        batch_shape, left, right = _prepare_homogeneous_shuffle_inputs(
            jnp,
            Ai,
            Bj,
            plan,
        )
        if runtime_plan.strategy == "direct":
            return _direct_homogeneous_shuffle(left, right, plan)

        outer = _homogeneous_outer_product(
            jnp,
            left,
            right,
            batch_shape,
            plan,
        )
        if runtime_plan.strategy == "transpose":
            return _sum_homogeneous_axis_permutations(
                jnp,
                outer,
                batch_shape,
                plan,
            )

        flat_outer = jnp.reshape(outer, batch_shape + (plan.output_width,))

        def gathered_sum(gather_indices, valid):
            gathered = jnp.take(flat_outer, gather_indices.T, axis=-1)
            valid = jnp.reshape(
                valid,
                (1,) * (gathered.ndim - 1) + (valid.size,),
            )
            gathered = jnp.where(valid, gathered, jnp.zeros((), gathered.dtype))
            return jnp.sum(gathered, axis=-1, dtype=flat_outer.dtype)

        if runtime_plan.strategy == "flat_gather":
            full_gather_bytes = (
                int(np.prod(batch_shape, dtype=np.int64))
                * runtime_plan.flat_indices.size
                * np.dtype(flat_outer.dtype).itemsize
            )
            if full_gather_bytes <= self.FULL_GATHER_MAX_TEMPORARY_BYTES:
                return gathered_sum(
                    jnp.reshape(
                        runtime_plan.flat_indices,
                        (-1, plan.output_width),
                    ),
                    jnp.reshape(runtime_plan.valid_masks, (-1,)),
                )

            def add_flat_chunk(carry, scan_input):
                gather_indices, valid = scan_input
                return carry + gathered_sum(gather_indices, valid), None

            result, _ = lax.scan(
                add_flat_chunk,
                jnp.zeros_like(flat_outer),
                (runtime_plan.flat_indices, runtime_plan.valid_masks),
            )
            return result

        positions = jnp.arange(plan.output_width, dtype=jnp.int32)
        digits = (
            positions[:, None] // runtime_plan.digit_divisors[None, :]
        ) % plan.dimension

        def add_coefficient_chunk(carry, scan_input):
            coefficients, valid = scan_input
            gather_indices = digits @ coefficients.T
            return carry + gathered_sum(gather_indices.T, valid), None

        result, _ = lax.scan(
            add_coefficient_chunk,
            jnp.zeros_like(flat_outer),
            (runtime_plan.coefficients, runtime_plan.valid_masks),
        )
        return result


class JaxSequentialCore(SequentialCore[jnp.ndarray]):
    """Sequential core implemented with JAX map and scan primitives."""

    capabilities = SequentialCore.capabilities | {"functional_indexed_update"}

    def __init__(self, default_time_axis: int = -2):
        super().__init__(jnp, default_time_axis)
        for name, function in _compiled_jittables(type(self)):
            setattr(self, name, types.MethodType(function, self))

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def _mapper(self, fun: Callable[[PyTree], PyTree]) -> Callable[[PyTree], PyTree]:
        """vmap ``fun`` over the leading axis of every leaf in a PyTree."""
        vmapped = jax.vmap(fun)
        return vmapped

    def _scanner(
            self,
            fun: Callable[[PyTree, PyTree], tuple[PyTree, PyTree]],
            *,
            initial: PyTree,
    ) -> Callable[[PyTree], tuple[PyTree, PyTree]]:
        """Lower a heterogeneous leading-axis scan directly to ``lax.scan``."""
        return lambda X: lax.scan(fun, initial, X)

    def _update_index_leaf(
            self,
            array: jnp.ndarray,
            index,
            value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Functionally update one JAX array leaf."""
        return array.at[index].set(value)

    def _reducer(
            self,
            fun: Callable[[PyTree, PyTree], PyTree],
            *,
            neutral: PyTree,
            seed: Optional[PyTree],
            in_tree: bool = False,
    ) -> Callable[[PyTree], PyTree]:
        seed_ = neutral if seed is None else seed
        if not in_tree:
            @jax.jit
            def reduce_fn(X):
                def body(carry, step):
                    return fun(carry, step), None
                final, _ = lax.scan(body, seed_, X)
                return final
            return reduce_fn

        @jax.jit
        def reduce_fn(X):
            # Parallel prefix, then fold seed into the global result.
            prefixes = lax.associative_scan(fun, X, axis=0)
            last = tree_index(prefixes, -1)
            return last if seed is None else fun(seed, last)
        return reduce_fn

    def _accumulator(
            self,
            fun: Callable[[PyTree, PyTree], PyTree],
            *,
            neutral: PyTree,
            seed: Optional[PyTree],
            in_tree: bool = False,
    ) -> Callable[[PyTree], tuple[PyTree, PyTree]]:
        seed_ = neutral if seed is None else seed
        if not in_tree:
            @jax.jit
            def scan_fn(X):
                def body(carry, step):
                    y = fun(carry, step)
                    return y, y
                final, ys = lax.scan(body, seed_, X)
                return final, ys
            return scan_fn

        @jax.jit
        def scan_fn(X):
            # An absent seed means the algebraic neutral and needs no work.
            prefixes = lax.associative_scan(fun, X, axis=0)
            if seed is None:
                return tree_index(prefixes, -1), prefixes
            ys = jax.vmap(lambda p: fun(seed, p))(prefixes)
            final = tree_index(ys, -1)
            return final, ys
        return scan_fn
