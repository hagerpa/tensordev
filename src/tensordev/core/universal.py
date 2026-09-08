from __future__ import annotations

import functools
import itertools
from numbers import Integral
from typing import Optional, Sequence, Tuple, Union, Literal, List, TypeVar, Generic, Callable, Protocol, Any

from tensordev.core.grading import (
    GradedContractionSchedule,
    GradedConvolutionSchedule,
    GradedInnerProductSchedule,
    GradedMapSchedule,
    GradedSummationSchedule,
    formal_exponential_series,
    formal_logarithm_series,
    graded_contraction,
    graded_convolution,
    graded_horner_first_level,
    graded_inner_product,
    graded_summation,
    map_graded_blocks,
    total_degree_layout,
)
from tensordev.core.utils.annotations import jit as dummy_jit
from tensordev.core.utils.pytrees import (
    tree_first_leaf,
    tree_index,
    tree_leaves,
    tree_map,
    tree_moveaxis,
    tree_stack,
)


class _Array(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> Any: ...

    def __getitem__(self, key: Any) -> _Array: ...

    def __add__(self, other: Any) -> _Array: ...

    def __radd__(self, other: Any) -> _Array: ...

    def __mul__(self, other: Any) -> _Array: ...

    def __rmul__(self, other: Any) -> _Array: ...

    def __pow__(self, other: Any) -> _Array: ...

    def sum(self, *args: Any, **kwargs: Any) -> _Array: ...

    def reshape(self, *args: Any, **kwargs: Any) -> _Array: ...


class _ArrayNamespace(Protocol):
    def stack(self, arrays: Sequence[Any], axis: int = 0) -> _Array: ...

    def moveaxis(self, x: _Array, source: Any, destination: Any) -> _Array: ...

    def expand_dims(self, x: _Array, axis: int) -> _Array: ...

    def reshape(self, x: _Array, shape: tuple[int, ...]) -> _Array: ...

    def broadcast_shapes(self, *shapes: tuple[int, ...]) -> tuple[int, ...]: ...

    def broadcast_to(self, x: _Array, shape: tuple[int, ...]) -> _Array: ...

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> _Array: ...

    def zeros_like(self, x: _Array) -> _Array: ...

    def ones_like(self, x: _Array) -> _Array: ...

    def asarray(self, obj: Any, dtype: Any = None) -> _Array: ...

    def diff(self, x: _Array, axis: int) -> _Array: ...

    def concat(self, arrays: Sequence[Any], axis: int = 0) -> _Array: ...


Array = TypeVar("Array", bound=_Array)
Elem = Sequence[Optional[Array]]  # one tensor-algebra element (level-list)
DenseElem = Tuple[Array, ...]  # level k has last dim d**k stating at level k=0; no Nones; shared batch shape
DenseElemFirstOn = Tuple[Array, ...]  # level k has last dim d**k starting at level k=1 no Nones; shared batch shape,


def _canonicalize_level_list_output(source: Any, result: Any) -> Any:
    """Preserve tuple output for list-based total-degree inputs."""
    return tuple(result) if isinstance(source, list) else result


class _TensorSliceProxy:
    """Proxy returned by ``tensor_slice(A)`` to support ``A[key]`` syntax."""
    __slots__ = ("_element",)

    def __init__(self, element: Any) -> None:
        self._element = element

    def __getitem__(self, key: Any) -> Any:
        return _canonicalize_level_list_output(
            self._element,
            tree_index(self._element, key),
        )


class Universal(Generic[Array]):
    grading = "total_degree"
    partially_symmetrized = False
    coordinates = "standard"
    # Plain ``Jax()`` remains unbounded; configured total-degree cores replace
    # these instance attributes with a finite capacity and active default.
    max_truncation = None
    default_truncation = None

    def __init__(
            self,
            xp: _ArrayNamespace,
            *,
            d: Optional[int] = None,
            max_trunc: Optional[int] = None,
            default_trunc: Optional[int] = None,
            shuffle_plan_store: Any = None,
    ):
        self.xp = xp
        namespace = getattr(xp, "__name__", "")
        self.backend = namespace.partition(".")[0] or None

        if shuffle_plan_store is not None:
            store_d = getattr(shuffle_plan_store, "d", None)
            store_max = getattr(shuffle_plan_store, "max_truncation", None)
            if d is None:
                d = store_d
            elif store_d != d:
                raise ValueError(
                    "d disagrees with the supplied shuffle_plan_store."
                )
            if max_trunc is None:
                max_trunc = store_max
            elif store_max != max_trunc:
                raise ValueError(
                    "max_trunc disagrees with the supplied shuffle_plan_store."
                )

        if d is not None:
            if isinstance(d, bool) or not isinstance(d, Integral):
                raise TypeError(f"d must be a positive integer, got {d!r}.")
            d = int(d)
            if d <= 0:
                raise ValueError(f"d must be positive, got {d}.")

        if max_trunc is None:
            if default_trunc is not None:
                raise ValueError(
                    "default_trunc requires a finite max_trunc capacity."
                )
        else:
            if isinstance(max_trunc, bool) or not isinstance(max_trunc, Integral):
                raise TypeError(
                    "max_trunc must be a non-negative integer, "
                    f"got {max_trunc!r}."
                )
            max_trunc = int(max_trunc)
            if max_trunc < 0:
                raise ValueError(
                    f"max_trunc must be non-negative, got {max_trunc}."
                )
            if default_trunc is None:
                default_trunc = max_trunc
            elif isinstance(default_trunc, bool) or not isinstance(
                default_trunc, Integral
            ):
                raise TypeError(
                    "default_trunc must be a non-negative integer, "
                    f"got {default_trunc!r}."
                )
            else:
                default_trunc = int(default_trunc)
                if default_trunc < 0:
                    raise ValueError(
                        "default_trunc must be non-negative, "
                        f"got {default_trunc}."
                    )
                if default_trunc > max_trunc:
                    raise ValueError(
                        f"default_trunc={default_trunc} exceeds core capacity "
                        f"max_trunc={max_trunc}."
                    )

        self.d = d
        self.max_truncation = max_trunc
        self.default_truncation = default_trunc
        self.shuffle_plan_store = shuffle_plan_store
        self._truncation_views = (
            {}
            if self.default_truncation is None
            else {self.default_truncation: self}
        )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        if self.max_truncation is None and self.d is None:
            return f"{type(self).__name__}()"
        return (
            f"{type(self).__name__}(d={self.d}, "
            f"max_trunc={self.max_truncation}, "
            f"default_trunc={self.default_truncation})"
        )

    @property
    def capabilities(self) -> frozenset[str]:
        """Static algebra capabilities available to higher-level consumers."""
        capabilities = {
            "concatenation",
            "generator_action",
            "shuffle",
            "coordinate_conversion",
            "shear_pairing",
        }
        if self.shuffle_plan_store is not None:
            capabilities.add("shuffle_product")
        return frozenset(capabilities)

    def supports(self, capability: str) -> bool:
        """Return whether a named algebra capability is available."""
        return capability in self.capabilities

    def _validate_graded_element(self, tensor: Any, *, name: str) -> Any:
        """Storage hook for public PyTree-generic operations."""
        del name
        return tensor

    def _coordinate_identity(
            self,
            element: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        """Validated identity coordinate map for standard total coordinates."""
        element = tuple(self._validate_graded_element(element, name="element"))
        if not element:
            if trunc is not None:
                self.normalize_truncation(trunc)
            return tuple()
        start = 1 if first_on else 0
        natural = start + len(element) - 1
        active = self._effective_truncation(trunc, natural)
        return element[:max(active - start + 1, 0)]

    def _coordinate_forward(
            self,
            element: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        return self._coordinate_identity(element, trunc=trunc, first_on=first_on)

    def _coordinate_inverse(
            self,
            element: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        return self._coordinate_identity(element, trunc=trunc, first_on=first_on)

    def _coordinate_forward_transpose(
            self,
            element: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        return self._coordinate_identity(element, trunc=trunc, first_on=first_on)

    def _coordinate_inverse_transpose(
            self,
            element: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        return self._coordinate_identity(element, trunc=trunc, first_on=first_on)

    def _coordinate_forward_transpose_block(
            self,
            block: Any,
            grade: Any,
    ) -> Any:
        """Identity dual-coordinate action for a standard-coordinate block."""
        del grade
        return block

    @staticmethod
    def _coordinate_natural_truncation(
            element: Any,
            *,
            first_on: bool,
    ) -> Any:
        """Infer the active truncation carried by a graded word element."""
        truncation = getattr(element, "truncation", None)
        if truncation is not None:
            return truncation
        size = len(tuple(element))
        if size == 0:
            return None
        return size - (0 if first_on else 1)

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "first_on"),
        dynamic_batch=("A",),
    )
    def tensor_from_standard_coordinates(
            self,
            A: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        """Convert standard coefficients to this core's native coordinates.

        This changes coordinates only: total-degree tuples remain total-degree
        tuples, while bidegree tensors retain their grading, truncation, and
        scalar policy.  On a standard-coordinate core this is a validated
        identity operation.
        """
        return self._coordinate_forward(A, trunc=trunc, first_on=first_on)

    @dummy_jit(
        static_argnums=0,
        static_argnames=("trunc", "first_on"),
        dynamic_batch=("A",),
    )
    def tensor_to_standard_coordinates(
            self,
            A: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        """Convert this core's native coefficients to standard coordinates.

        This is the inverse coordinate map and does not change total versus
        bidegree layout.  A bidegree shear result remains a
        ``BigradedTensor``, with its coordinate metadata changed to
        ``"standard"``.  On a standard-coordinate core this is a validated
        identity operation.
        """
        return self._coordinate_inverse(A, trunc=trunc, first_on=first_on)

    def tensor_partially_symmetrize_homogeneous(
            self,
            block: Any,
            *,
            grade: Any = None,
    ) -> Any:
        """Partially symmetrize one ordered block when supported."""
        del block, grade
        raise RuntimeError(
            "tensor_partially_symmetrize_homogeneous requires a partially "
            "symmetrized bidegree core."
        )

    def tensor_partially_symmetrize(
            self,
            element: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        """Partially symmetrize one ordered tensor when supported."""
        del element, trunc, first_on
        raise RuntimeError(
            "tensor_partially_symmetrize requires a partially symmetrized "
            "bidegree core."
        )

    def tensor_to_ordered(
            self,
            element: Any,
            *,
            trunc: Any = None,
            first_on: bool = False,
    ) -> Any:
        """Lift a partially symmetrized word tensor when supported."""
        del element, trunc, first_on
        raise RuntimeError(
            "tensor_to_ordered requires a partially symmetrized bidegree core."
        )

    def normalize_truncation(self, trunc: Optional[int]) -> int:
        """Resolve a total-degree truncation against this core's capacity."""
        if trunc is None:
            if self.default_truncation is None:
                raise ValueError(
                    "An explicit integer truncation is required by the unbounded "
                    "total-degree core."
                )
            trunc = self.default_truncation
        if isinstance(trunc, bool) or not isinstance(trunc, Integral):
            raise TypeError(f"trunc must be a non-negative integer, got {trunc!r}.")
        trunc = int(trunc)
        if trunc < 0:
            raise ValueError(f"trunc must be non-negative, got {trunc}.")
        if self.max_truncation is not None and trunc > self.max_truncation:
            raise ValueError(
                f"active truncation {trunc} exceeds core capacity "
                f"{self.max_truncation}."
            )
        return trunc

    def _effective_truncation(
            self,
            trunc: Optional[int],
            natural_truncation: int,
    ) -> int:
        """Cap an input-inferred result without padding short tuples.

        An unbounded core infers its output degree when ``trunc`` is omitted.
        A bounded core instead applies its configured default while returning
        no levels beyond those naturally supplied by the operands.
        """
        if trunc is None and self.default_truncation is None:
            return natural_truncation
        return min(natural_truncation, self.normalize_truncation(trunc))

    def at_truncation(self, trunc: int):
        """Return a cheap bounded-core view with another active truncation."""
        if self.max_truncation is None:
            raise RuntimeError(
                "at_truncation requires a bounded total-degree core."
            )
        active = self.normalize_truncation(trunc)
        cached = self._truncation_views.get(active)
        if cached is not None:
            return cached
        view = type(self)(
            d=self.d,
            max_trunc=self.max_truncation,
            default_trunc=active,
            shuffle_plan_store=self.shuffle_plan_store,
        )
        view._truncation_views = self._truncation_views
        self._truncation_views[active] = view
        return view

    def memory_bytes_by_category(self) -> dict[str, int]:
        """Eager plan payload owned by this core, grouped by category."""
        if self.shuffle_plan_store is None:
            return {}
        return dict(self.shuffle_plan_store.memory_bytes_by_category())

    def memory_bytes(self) -> int:
        """Total eager plan payload owned by this core, in bytes."""
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        """Total eager plan payload owned by this core, in megabytes."""
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> dict[str, Any]:
        """Summarize this total-degree core and its optional shuffle plans."""
        statistics = (
            {}
            if self.shuffle_plan_store is None
            else dict(self.shuffle_plan_store.plan_statistics())
        )
        statistics.update(
            {
                "d": self.d,
                "max_truncation": self.max_truncation,
                "default_truncation": self.default_truncation,
                "shuffle_enabled": self.shuffle_plan_store is not None,
                "memory_bytes": self.memory_bytes(),
                "memory_mb": self.memory_mb(),
            }
        )
        return statistics

    def resolve_layout(
            self,
            trunc: Optional[int] = None,
            *,
            include_scalar: bool = True,
    ):
        """Resolve a finite total-degree layout for static graded schedules."""
        return total_degree_layout(
            self.normalize_truncation(trunc), include_scalar=include_scalar
        )

    # ------------------------------------------------------------------
    # Static layout protocol for grading-generic consumers
    # ------------------------------------------------------------------

    def _validate_alphabet_dim(self, alphabet_dim: int) -> int:
        if isinstance(alphabet_dim, bool) or not isinstance(alphabet_dim, Integral):
            raise TypeError(
                "alphabet_dim must be a positive integer, "
                f"got {alphabet_dim!r}."
            )
        alphabet_dim = int(alphabet_dim)
        if alphabet_dim <= 0:
            raise ValueError(
                f"alphabet_dim must be positive, got {alphabet_dim}."
            )
        if self.d is not None and alphabet_dim != self.d:
            raise ValueError(
                f"alphabet dimension {alphabet_dim} does not match configured "
                f"dimension {self.d}."
            )
        return alphabet_dim

    def _block_width_for_layout(self, layout, grade, *, alphabet_dim: int) -> int:
        alphabet_dim = self._validate_alphabet_dim(alphabet_dim)
        if not layout.contains(grade):
            raise KeyError(f"grade {grade!r} is not active in this layout.")
        return alphabet_dim ** layout.total_degree(grade)

    def _element_block(self, element, grade, *, layout):
        """Return one native block using the layout's scalar convention."""
        if not layout.contains(grade):
            raise KeyError(f"grade {grade!r} is not active in this layout.")
        element = tuple(element)
        if len(element) != len(layout.grades):
            raise ValueError(
                f"element has {len(element)} blocks, expected "
                f"{len(layout.grades)} for the resolved layout."
            )
        return element[layout.index(grade)]

    def _assemble_element(self, blocks, layout):
        blocks = tuple(blocks)
        if len(blocks) != len(layout.grades):
            raise ValueError(
                f"received {len(blocks)} blocks, expected {len(layout.grades)} "
                "for the resolved layout."
            )
        return blocks

    def _zero_block_for_layout(
            self,
            layout,
            grade,
            *,
            batch_shape: tuple[int, ...],
            dtype: Any,
            alphabet_dim: int,
    ):
        width = self._block_width_for_layout(
            layout, grade, alphabet_dim=alphabet_dim
        )
        return self.xp.zeros(tuple(batch_shape) + (width,), dtype=dtype)

    def _constant_element_for_layout(
            self,
            layout,
            *,
            batch_shape: tuple[int, ...],
            dtype: Any,
            alphabet_dim: int,
            scalar: float = 0.0,
    ):
        blocks = []
        for grade in layout.grades:
            if grade == layout.zero_grade:
                width = self._block_width_for_layout(
                    layout, grade, alphabet_dim=alphabet_dim
                )
                block = self.xp.full(
                    tuple(batch_shape) + (width,), scalar, dtype=dtype
                )
            else:
                block = self._zero_block_for_layout(
                    layout,
                    grade,
                    batch_shape=batch_shape,
                    dtype=dtype,
                    alphabet_dim=alphabet_dim,
                )
            blocks.append(block)
        return self._assemble_element(tuple(blocks), layout)

    def _generator_blocks(self, z, *, layout):
        """Decompose a first-level generator into native homogeneous blocks."""
        del layout
        z = self.xp.asarray(z)
        if z.ndim == 0 or z.shape[-1] <= 0:
            raise ValueError("a first-level generator must have positive width.")
        self._validate_alphabet_dim(z.shape[-1])
        return ((1, z),)

    @staticmethod
    def _validate_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        predecessor_blocks = tuple(predecessor_blocks)
        generator_blocks = tuple(generator_blocks)
        predecessor_grades = tuple(predecessor_grades)
        generator_grades = tuple(generator_grades)
        sizes = {
            len(predecessor_blocks),
            len(generator_blocks),
            len(predecessor_grades),
            len(generator_grades),
        }
        if len(sizes) != 1 or not predecessor_blocks:
            raise ValueError(
                "generator actions require equally sized, non-empty block and "
                "grade tuples."
            )
        return predecessor_blocks, generator_blocks

    def _right_multiply_generator_output_block(
            self,
            predecessor_blocks,
            generator_blocks,
            *,
            predecessor_grades,
            generator_grades,
            output_grade,
    ):
        predecessor_blocks, generator_blocks = self._validate_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )
        predecessor_grades = tuple(predecessor_grades)
        generator_grades = tuple(generator_grades)
        terms = []
        for source, generator, source_grade, generator_grade in zip(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        ):
            if source_grade + generator_grade != output_grade:
                raise ValueError("generator grades do not add to output_grade.")
            terms.append(self.tensor_product_homogeneous(source, generator))
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    def _shuffle_generator_output_block(
            self,
            predecessor_blocks,
            generator_blocks,
            *,
            predecessor_grades,
            generator_grades,
            output_grade,
    ):
        predecessor_blocks, generator_blocks = self._validate_generator_action_inputs(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )
        predecessor_grades = tuple(predecessor_grades)
        generator_grades = tuple(generator_grades)
        terms = []
        for source, generator, source_grade, generator_grade in zip(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        ):
            if source_grade + generator_grade != output_grade:
                raise ValueError("generator grades do not add to output_grade.")
            if generator_grade != 1:
                raise ValueError("total-degree generator grade must equal one.")
            terms.append(
                self.tensor_shuffle_vector_homogeneous(
                    source, generator, int(source_grade)
                )
            )
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    def prepare_development_input(
            self,
            X: DenseElemFirstOn,
            *,
            trunc: int,
            increment_input: bool,
            axis: int,
    ) -> DenseElemFirstOn:
        """Prepare positive total-degree levels for the development driver."""
        X = tuple(X)
        if not X:
            raise ValueError("free_development: X must contain at least one level.")
        if self.d is not None:
            for degree, level in enumerate(X, start=1):
                width = None if level.ndim == 0 else level.shape[-1]
                expected = self.d**degree
                if width != expected:
                    raise ValueError(
                        f"degree-{degree} development input has width {width}, "
                        f"expected {expected} for configured dimension {self.d}."
                    )
        selected = X[:trunc]
        return selected if increment_input else tuple(
            self.xp.diff(level, axis=axis) for level in selected
        )

    def development_neutral(
            self,
            increments: DenseElemFirstOn,
            *,
            trunc: int,
            axis: int,
    ) -> DenseElem:
        """Construct the identity carry matching a development's batch shape."""
        first = increments[0]
        time_axis = axis if axis >= 0 else first.ndim + axis
        index = tuple(0 if i == time_axis else slice(None) for i in range(first.ndim))
        prototype = first[index]
        layout = self.resolve_layout(trunc, include_scalar=True)
        return self._constant_element_for_layout(
            layout,
            batch_shape=prototype.shape[:-1],
            dtype=prototype.dtype,
            alphabet_dim=prototype.shape[-1],
            scalar=1.0,
        )

    # ----------------------------------------------------------------------
    # Axes iteration and reduction utilites
    # ----------------------------------------------------------------------

    @dummy_jit(static_argnums=0, static_argnames=("axis",), dynamic_batch=("X",))
    def tensor_stack(self, X: List[DenseElem], *, axis: int) -> tuple:
        X = [
            self._validate_graded_element(element, name=f"X[{index}]")
            for index, element in enumerate(X)
        ]
        result = tree_stack(self.xp, X, axis=axis)
        return _canonicalize_level_list_output(X[0], result)

    @dummy_jit(static_argnums=0, static_argnames=("source", "destination"), dynamic_batch=("X",))
    def tensor_moveaxis(
            self,
            X: DenseElem,
            *,
            source: int,
            destination: int
    ) -> DenseElem:
        X = self._validate_graded_element(X, name="X")
        result = tree_moveaxis(self.xp, X, source=source, destination=destination)
        return _canonicalize_level_list_output(X, result)

    def _mapper(self, fun: Callable[[DenseElem], DenseElem]):
        def map_fn(seq):
            size = int(tree_first_leaf(seq).shape[0])
            return self.tensor_stack([fun(tree_index(seq, i)) for i in range(size)], axis=0)

        return map_fn

    def _reducer(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
            associative: bool = False,
    ):
        """
        Maker: returns reduce_fn(X) -> single graded element.
        Sequential left-fold from `seed` over axis 0 using `fun`.
        """

        def reduce_fn(X):
            X = tuple(X)
            S = int(X[0].shape[0])
            step = lambda t: tuple(a[t] for a in X)
            steps = (step(t) for t in range(S))
            return functools.reduce(fun, steps, seed)

        return reduce_fn

    def _accumulator(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
            associative: bool = False,
    ):
        """
        Maker: returns scan_fn(X) -> (final, ys).

        Semantics:
          - The output of `fun` becomes the next carry.
          - Using itertools.accumulate over [seed, s0, s1, ...]:
              prefixes = [seed, fun(seed,s0), fun(fun(seed,s0),s1), ...]
            The initial `seed` is omitted from the emitted sequence.
          - final = ys[-1], ys are the prefixes without the bare seed.
        """

        def scan_fn(X):
            X = tuple(X)
            S = int(X[0].shape[0])
            step = lambda t: tuple(a[t] for a in X)
            prefixes = itertools.accumulate(
                itertools.chain([seed], (step(t) for t in range(S))),
                fun
            )
            ys = list(itertools.islice(prefixes, 1, None))  # drop the bare seed
            return ys[-1], self.tensor_stack(ys, axis=0)

        return scan_fn

    # ----------------------------------------------------------------------
    # Single & Binary Operations
    # ----------------------------------------------------------------------

    def _standard_summation_schedule(
            self,
            A: DenseElem,
            B: DenseElem,
            trunc: Optional[int],
    ) -> GradedSummationSchedule:
        """Resolve the total-degree tuple layout for addition."""
        A, B = tuple(A), tuple(B)
        NA, NB = len(A) - 1, len(B) - 1
        N = self._effective_truncation(trunc, max(NA, NB))
        return GradedSummationSchedule(
            grades=tuple(range(N + 1)),
            left_contains=lambda k: k <= NA,
            right_contains=lambda k: k <= NB,
            left_block=lambda k: A[k],
            right_block=lambda k: B[k],
            assemble=tuple,
        )

    def _summation_schedule(self, A, B, trunc) -> GradedSummationSchedule:
        return self._standard_summation_schedule(A, B, trunc)

    def _standard_tensor_summation(self, A, B, trunc=None):
        """Execute coordinatewise addition without coordinate redispatch."""
        schedule = self._standard_summation_schedule(A, B, trunc)
        blocks = graded_summation(
            schedule.grades,
            left_contains=schedule.left_contains,
            right_contains=schedule.right_contains,
            left_block=schedule.left_block,
            right_block=schedule.right_block,
            add=schedule.add,
            left_only=schedule.left_only,
            right_only=schedule.right_only,
            zero_block=schedule.zero_block,
        )
        return schedule.assemble(blocks)

    @dummy_jit(static_argnums=0, static_argnames=("trunc",), dynamic_batch=("A", "B"))
    def tensor_summation(self, A: DenseElem, B: DenseElem, trunc: Optional[int] = None) -> DenseElem:
        """
        Level-wise sum ``C = A + B`` (with implicit zero-padding if degrees differ).

        Parameters
        ----------
        A, B : sequence of ndarray
            Inputs with shapes ``A_k.shape == B_k.shape`` whenever both exist.
        trunc : int, optional
            If given, keep only degrees ``0..trunc``.

        Returns
        -------
        tuple of ndarray
            ``C_k = A_k + B_k`` for all available ``k`` (zeros used if one side
            lacks degree ``k``).
        """
        schedule = self._summation_schedule(A, B, trunc)
        blocks = graded_summation(
            schedule.grades,
            left_contains=schedule.left_contains,
            right_contains=schedule.right_contains,
            left_block=schedule.left_block,
            right_block=schedule.right_block,
            add=schedule.add,
            left_only=schedule.left_only,
            right_only=schedule.right_only,
            zero_block=schedule.zero_block,
        )
        return schedule.assemble(blocks)

    def _standard_product_homogeneous(self, Ai: Array, Bj: Array) -> Array:
        """Raw ordinary homogeneous concatenation without coordinate dispatch."""
        tmp = self.xp.expand_dims(Ai, -1) * self.xp.expand_dims(Bj, -2)
        return self.xp.reshape(
            tmp, tmp.shape[:-2] + (tmp.shape[-2] * tmp.shape[-1],)
        )

    @dummy_jit(static_argnums=0, dynamic_batch=("Ai", "Bj"))
    def tensor_product_homogeneous(self, Ai: Array, Bj: Array) -> Array:
        """
        Homogeneous-degree tensor (Chen) product on the last axis.

        Given degree-i and degree-j levels with shapes
            Ai.shape == batch + (d**i,),   Bj.shape == batch + (d**j,),
        return their (flattened) tensor product
            (Ai ⊗ Bj).shape == batch + (d**(i+j),).

        This is just a Kronecker on the last axis, preserving the batch shape.
        """
        return self._standard_product_homogeneous(Ai, Bj)

    def _standard_product_block(
            self,
            Ai: Array,
            Bj: Array,
            left_grade: int,
            right_grade: int,
            output_grade: int,
    ) -> Array:
        """Ordinary concatenation block used below coordinate transport."""
        del left_grade, right_grade, output_grade
        return self._standard_product_homogeneous(Ai, Bj)

    def _product_block(
            self,
            Ai: Array,
            Bj: Array,
            left_grade: int,
            right_grade: int,
            output_grade: int,
    ) -> Array:
        """Coordinate-native homogeneous product hook."""
        return self._standard_product_block(
            Ai, Bj, left_grade, right_grade, output_grade
        )

    @staticmethod
    def _sum_product_contributions(
            contributions,
            output_grade,
            product_block,
    ):
        contributions = tuple(contributions)
        if not contributions:
            raise ValueError(
                f"No product contribution for output grade {output_grade!r}."
            )
        left, right, left_grade, right_grade = contributions[0]
        result = product_block(
            left, right, left_grade, right_grade, output_grade
        )
        for left, right, left_grade, right_grade in contributions[1:]:
            result = result + product_block(
                left, right, left_grade, right_grade, output_grade
            )
        return result

    def _standard_product_output_block(self, contributions, output_grade):
        """Complete one ordinary-product output grade from resolved splits."""
        return self._sum_product_contributions(
            contributions, output_grade, self._standard_product_block
        )

    def _product_output_block(self, contributions, output_grade):
        """Complete one coordinate-native product grade for external consumers."""
        return self._sum_product_contributions(
            contributions, output_grade, self._product_block
        )

    def _standard_product_schedule(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            trunc: Optional[int],
            *,
            a_first_on: bool,
            b_first_on: bool,
            first_on_out: bool = False,
    ) -> GradedConvolutionSchedule:
        """Resolve total-degree levels for an ordinary graded product."""
        A, B = tuple(A), tuple(B)
        if len(A) == 0 or len(B) == 0:
            if trunc is not None:
                self.normalize_truncation(trunc)
            return GradedConvolutionSchedule(
                grades=tuple(),
                splits=lambda _grade: tuple(),
                left_contains=lambda _grade: False,
                right_contains=lambda _grade: False,
                left_block=lambda grade: A[grade],
                right_block=lambda grade: B[grade],
                assemble=tuple,
            )

        a0 = 1 if a_first_on else 0
        b0 = 1 if b_first_on else 0
        N = self._effective_truncation(
            trunc,
            len(A) + len(B) + a0 + b0 - 2,
        )

        start = 1 if first_on_out or a_first_on or b_first_on else 0
        prefix: tuple[Any, ...] = tuple()
        if a_first_on and b_first_on:
            if N < 1:
                return GradedConvolutionSchedule(
                    grades=tuple(),
                    splits=lambda _grade: tuple(),
                    left_contains=lambda _grade: False,
                    right_contains=lambda _grade: False,
                    left_block=lambda grade: A[grade],
                    right_block=lambda grade: B[grade],
                    assemble=tuple,
                )
            prefix = (self.xp.zeros_like(A[0]),)
            start = 2

        a_last = len(A) + a0 - 1
        b_last = len(B) + b0 - 1
        return GradedConvolutionSchedule(
            grades=tuple(range(start, N + 1)),
            splits=lambda n: (
                (i, n - i)
                for i in range(max(a0, n - b_last), min(a_last, n - b0) + 1)
            ),
            left_contains=lambda i: a0 <= i <= a_last,
            right_contains=lambda j: b0 <= j <= b_last,
            left_block=lambda i: A[i - a0],
            right_block=lambda j: B[j - b0],
            assemble=lambda blocks: prefix + tuple(blocks),
        )

    def _product_schedule(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            trunc: Optional[int],
            *,
            a_first_on: bool,
            b_first_on: bool,
            first_on_out: bool = False,
    ) -> GradedConvolutionSchedule:
        """Resolve total-degree levels for the coordinate-native product."""
        return self._standard_product_schedule(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )

    def _shuffle_schedule(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            trunc: Optional[int],
            *,
            a_first_on: bool,
            b_first_on: bool,
            first_on_out: bool,
    ) -> GradedConvolutionSchedule:
        """Resolve the total-degree tuple layout for full shuffle."""
        self._require_shuffle()
        return self._product_schedule(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )

    def _require_shuffle(self):
        """Return this core's full-shuffle plan store or fail explicitly."""
        if self.shuffle_plan_store is None:
            raise RuntimeError(
                "Full shuffle products require a core constructed with "
                "precompute_shuffle=True."
            )
        return self.shuffle_plan_store

    def _shuffle_block(
            self,
            left: Array,
            right: Array,
            left_grade: int,
            right_grade: int,
            output_grade: int,
    ) -> Array:
        """Apply one canonical ordinary homogeneous shuffle plan."""
        if left_grade + right_grade != output_grade:
            raise ValueError("shuffle plan/output grade mismatch")
        store = self._require_shuffle()
        self.normalize_truncation(output_grade)
        if left_grade < right_grade:
            left, right = right, left
            left_grade, right_grade = right_grade, left_grade
        return store.apply(
            self.xp,
            left,
            right,
            int(left_grade),
            int(right_grade),
        )

    @dummy_jit(static_argnums=(0, 3, 4), dynamic_batch=("Ai", "Bj"))
    def permutation_einsum(
            self,
            Ai: Array,
            Bj: Array,
            i: int,
            j: int,
    ) -> Array:
        """Apply the precomputed ordinary shuffle plan for degrees ``i,j``."""
        return self._shuffle_block(Ai, Bj, i, j, i + j)

    @dummy_jit(static_argnums=(0, 3, 4), dynamic_batch=("Ai", "Bj"))
    def tensor_shuffle_product_homogeneous(
            self,
            Ai: Array,
            Bj: Array,
            i: int,
            j: int,
    ) -> Array:
        """Compute the homogeneous ordinary shuffle product."""
        return self._shuffle_block(Ai, Bj, i, j, i + j)

    @dummy_jit(
        static_argnums=(0,),
        static_argnames=("trunc", "a_first_on", "b_first_on", "first_on_out"),
        dynamic_batch=("A", "B"),
    )
    def tensor_shuffle_product(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            *,
            trunc: Optional[int] = None,
            a_first_on: bool = False,
            b_first_on: bool = False,
            first_on_out: bool = False,
    ) -> Union[DenseElem, DenseElemFirstOn]:
        """Compute the truncated graded, commutative shuffle product."""
        schedule = self._shuffle_schedule(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )
        blocks = graded_convolution(
            schedule.grades,
            splits=schedule.splits,
            left_contains=schedule.left_contains,
            right_contains=schedule.right_contains,
            left_block=schedule.left_block,
            right_block=schedule.right_block,
            product_block=self._shuffle_block,
            zero_block=schedule.zero_block,
            product_grade=schedule.product_grade,
        )
        return schedule.assemble(blocks)

    @dummy_jit(static_argnums=(0, 3), dynamic_batch=("Ai", "v"))
    def tensor_shuffle_vector_homogeneous(self, Ai: Array, v: Array, i: int) -> Array:
        """
        Homogeneous shuffle product ``(A_i ⊔ v)_{i+1}`` where ``v`` is degree-1.

        For each output word ``(l_0, ..., l_i)`` the result is:
            sum_{p=0}^{i} A_i[l_0,...,l_{p-1},l_{p+1},...,l_i] * v[l_p]

        Parameters
        ----------
        Ai : Array, shape ``batch + (d**i,)``
        v : Array, shape ``batch + (d,)`` — batch shapes are broadcast.
        i : int
            Degree of ``Ai``.

        Returns
        -------
        Array, shape ``batch + (d**(i+1),)``
        """
        xp = self.xp
        d = v.shape[-1]
        batch = xp.broadcast_shapes(Ai.shape[:-1], v.shape[:-1])
        A = xp.broadcast_to(Ai, batch + (d ** i,)).reshape(batch + (d,) * i)
        v_ = xp.broadcast_to(v, batch + (d,))
        result = xp.zeros(batch + (d,) * (i + 1), dtype=Ai.dtype)
        for p in range(i + 1):
            A_exp = xp.expand_dims(A, axis=len(batch) + p)
            v_exp = v_.reshape(batch + (1,) * p + (d,) + (1,) * (i - p))
            result = result + A_exp * v_exp
        return result.reshape(batch + (d ** (i + 1),))

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "a_first_on"), dynamic_batch=("A", "v"))
    def tensor_shuffle_vector(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            v: Array,
            *,
            trunc: Optional[int] = None,
            a_first_on: bool = False,
    ) -> DenseElemFirstOn:
        """
        Graded shuffle product ``C = A ⊔ v`` where ``v`` is degree-1.

        Output is in first-on format (degrees 1, 2, ...):
            C_n = A_{n-1} ⊔ v   for n = 1, ..., min(NA + 1, trunc)

        Parameters
        ----------
        A : DenseElem or DenseElemFirstOn
            Graded element. If ``a_first_on=True``, starts at degree 1.
        v : Array, shape ``batch + (d,)``
            Degree-1 vector; batch shapes are broadcast per level.
        trunc : int, optional
            Maximum output degree.
        a_first_on : bool, default False
            Whether ``A`` starts at degree 1.

        Returns
        -------
        DenseElemFirstOn
            Output levels starting at degree 1.
        """
        A = tuple(A)
        if not A:
            return tuple()
        a0 = 1 if a_first_on else 0
        NA = len(A) + a0 - 1
        N = self._effective_truncation(trunc, NA + 1)
        out: List[Array] = []
        for n in range(1, N + 1):
            i = n - 1
            if i < a0 or i > NA:
                continue
            out.append(self.tensor_shuffle_vector_homogeneous(A[i - a0], v, i))
        return tuple(out)

    def _standard_tensor_product(
            self,
            A,
            B,
            trunc=None,
            *,
            a_first_on: bool = False,
            b_first_on: bool = False,
            first_on_out: bool = False,
    ):
        """Execute ordinary concatenation without coordinate redispatch."""
        schedule = self._standard_product_schedule(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
            first_on_out=first_on_out,
        )
        blocks = graded_convolution(
            schedule.grades,
            splits=schedule.splits,
            left_contains=schedule.left_contains,
            right_contains=schedule.right_contains,
            left_block=schedule.left_block,
            right_block=schedule.right_block,
            product_block=self._standard_product_block,
            zero_block=schedule.zero_block,
            product_grade=schedule.product_grade,
        )
        return schedule.assemble(blocks)

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "a_first_on", "b_first_on"),
               dynamic_batch=("A", "B"))
    def tensor_product(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            trunc: Optional[int] = None,
            *,
            a_first_on: bool = False,
            b_first_on: bool = False,
    ) -> Union[DenseElem, DenseElemFirstOn]:
        """
        Graded (Cauchy-type) product ``C = A ⊗ B`` in the free tensor algebra.

        For each degree ``n``,
            C_n = ∑_{i+j=n} flatten(tensor_product_homogeneous(A_i, B_j)).

        Parameters
        ----------
        A, B : tuple of ndarray
            Levels of the left/right factors, each of shape ``batch + (d**k,)``.

            By default, inputs are interpreted as
            ``(A_0, A_1, ..., A_N)`` and ``(B_0, B_1, ..., B_M)``.

            If ``a_first_on=True``, then ``A`` is interpreted as
            ``(A_1, A_2, ..., A_N)``.

            If ``b_first_on=True``, then ``B`` is interpreted analogously.

        trunc : int, optional
            If given, truncate the result to degrees up to ``trunc``.

        a_first_on, b_first_on : bool, default=False
            Whether the corresponding input starts at degree ``1`` rather than ``0``.

        Returns
        -------
        tuple of ndarray
            Product levels. If both flags are ``False``, returns
            ``(C_0, C_1, ..., C_N)``. Otherwise returns ``(C_1, C_2, ..., C_N)``.
        """
        return self._standard_tensor_product(
            A,
            B,
            trunc,
            a_first_on=a_first_on,
            b_first_on=b_first_on,
        )

    def _standard_tensor_scalar_multiply(self, A, alpha):
        """Coordinatewise scaling without coordinate-system validation."""
        try:
            first_leaf = tree_first_leaf(A)
        except ValueError:
            return tuple() if isinstance(A, list) else A
        a0 = self.xp.asarray(alpha, dtype=first_leaf.dtype)
        result = tree_map(lambda Ak: Ak * self.xp.expand_dims(a0, axis=-1), A)
        return _canonicalize_level_list_output(A, result)

    @dummy_jit(static_argnums=0, dynamic_batch=("A",), full_dynamic=("alpha",))
    def tensor_scalar_multiply(self, A: DenseElem, alpha: Union[Array, float]) -> DenseElem:
        """
        Uniform scalar multiply: ``Y = a * X``.

        Parameters
        ----------
        A : sequence of ndarray
            Input levels.
        alpha : float or ndarray
            Either a scalar, or a batch-shaped array broadcastable to the
            leading batch shape. When array-valued, scaling is done elementwise
            per batch example (broadcast over the last axis).

        Returns
        -------
        tuple of ndarray
            Levels ``B_k = alpha * A_k``.
        """
        A = self._validate_graded_element(A, name="A")
        return self._standard_tensor_scalar_multiply(A, alpha)

    def _dilation_schedule(self, A: DenseElem) -> GradedMapSchedule:
        """Resolve total-degree levels for the shared dilation map."""
        A = tuple(A)
        return GradedMapSchedule(
            grades=tuple(range(len(A))),
            block=lambda grade: A[grade],
            assemble=tuple,
        )

    def _grade_total_degree(self, grade: Any) -> int:
        """Return the dilation exponent associated with a grade."""
        return int(grade)

    @dummy_jit(static_argnums=0, dynamic_batch=("A",), full_dynamic=("c",))
    def tensor_dilation(self, A: DenseElem, c: Union[Array, float]) -> DenseElem:
        """Scale each homogeneous degree ``k`` by ``c**k``."""
        A = self._validate_graded_element(A, name="A")
        schedule = self._dilation_schedule(A)
        if not schedule.grades:
            return schedule.assemble(tuple())
        c0 = self.xp.asarray(c, dtype=schedule.block(schedule.grades[0]).dtype)
        blocks = map_graded_blocks(
            schedule.grades,
            schedule.block,
            lambda Xk, grade: Xk * self.xp.expand_dims(
                c0 ** self._grade_total_degree(grade), axis=-1
            ),
        )
        return schedule.assemble(blocks)

    @dummy_jit(static_argnums=0, dynamic_batch=("Ak", "Bk"))
    def tensor_inner_product_homogeneous(self, Ak: Array, Bk: Array) -> Array:
        """
        Level-wise (homogeneous) Euclidean inner product over the last axis.

        Parameters
        ----------
        Ak, Bk : ndarray
            Same-degree levels with shape ``batch + (d**k,)``.

        Returns
        -------
        ndarray
            Batch-shaped array ``batch`` with ⟨Ak, Bk⟩ = sum over the last axis.
        """
        return (Ak * Bk).sum(-1)

    def _standard_inner_product_schedule(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
    ) -> GradedInnerProductSchedule:
        """Resolve matching standard total-degree tuple levels."""
        A = self._validate_graded_element(A, name="A")
        B = self._validate_graded_element(B, name="B")
        return GradedInnerProductSchedule(
            grades=tuple(range(min(len(A), len(B)))),
            left_block=lambda grade: A[grade],
            right_block=lambda grade: B[grade],
            zero=lambda: self.xp.asarray(0.0),
        )

    def _standard_pairing_inner_product_schedule(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            *,
            a_first_on: bool,
            b_first_on: bool,
    ) -> GradedInnerProductSchedule:
        """Resolve actual total degrees for a mixed-origin pairing."""
        A = tuple(self._validate_graded_element(A, name="A"))
        B = tuple(self._validate_graded_element(B, name="B"))
        a_start = 1 if a_first_on else 0
        b_start = 1 if b_first_on else 0
        common_start = max(a_start, b_start)
        common_stop = min(a_start + len(A), b_start + len(B))
        return GradedInnerProductSchedule(
            grades=tuple(range(common_start, common_stop)),
            left_block=lambda grade: A[grade - a_start],
            right_block=lambda grade: B[grade - b_start],
            zero=lambda: self._standard_pairing_zero(A, B),
        )

    def _standard_pairing_zero(self, *tensors: Any) -> Array:
        """Return a broadcast zero with the homogeneous kernel's result type."""
        leaf_groups = tuple(tuple(tree_leaves(tensor)) for tensor in tensors)
        if not any(leaf_groups):
            return self.xp.asarray(0.0)
        leaves = tuple(leaf for group in leaf_groups for leaf in group)
        batch = self.xp.broadcast_shapes(
            *(leaf.shape[:-1] for leaf in leaves)
        )
        dtype = self.xp.result_type(*(leaf.dtype for leaf in leaves))
        zero_block = self.xp.zeros(batch + (1,), dtype=dtype)
        return self.tensor_inner_product_homogeneous(zero_block, zero_block)

    def _inner_product_schedule(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
    ) -> GradedInnerProductSchedule:
        """Resolve the native-coordinate inner-product schedule."""
        return self._standard_inner_product_schedule(A, B)

    def _evaluate_inner_product_schedule(
            self,
            schedule: GradedInnerProductSchedule,
    ) -> Array:
        """Evaluate one validated graded Euclidean contraction schedule."""
        return graded_inner_product(
            schedule.grades,
            left_block=schedule.left_block,
            right_block=schedule.right_block,
            inner_block=self.tensor_inner_product_homogeneous,
            zero=schedule.zero,
        )

    def _standard_tensor_inner_product(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            *,
            a_first_on: bool = False,
            b_first_on: bool = False,
    ) -> Array:
        """Euclidean pairing driver for operands in standard coordinates."""
        return self._evaluate_inner_product_schedule(
            self._standard_pairing_inner_product_schedule(
                A,
                B,
                a_first_on=a_first_on,
                b_first_on=b_first_on,
            )
        )

    @dummy_jit(static_argnums=0, dynamic_batch=("A", "B"))
    def tensor_inner_product(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
    ) -> Array:
        """
        Canonical Euclidean inner product, summing level-wise dot products over the
        last axis.

        Parameters
        ----------
        A, B : DenseElem or DenseElemFirstOn
            Graded elements with matching batch shape. If the inputs start at level 1,
            the inner product is taken over the positive levels only.

        Returns
        -------
        Array
            Batch-shaped array with the inner product.
        """
        return self._evaluate_inner_product_schedule(
            self._inner_product_schedule(A, B)
        )

    def _native_words_to_standard_block(
            self,
            words: Array,
            *,
            grade: Any,
    ) -> Array:
        """Apply the dual coordinate map without changing symmetrization."""
        return self._coordinate_forward_transpose_block(words, grade)

    def _native_words_to_standard(
            self,
            words: Union[DenseElem, DenseElemFirstOn],
            *,
            trunc: Any,
            first_on: bool,
    ) -> Union[DenseElem, DenseElemFirstOn]:
        """Apply the dual coordinate map without changing symmetrization."""
        return self._coordinate_forward_transpose(
            words,
            trunc=trunc,
            first_on=first_on,
        )

    def _validate_ordered_signature_pairing_block(
            self,
            block: Array,
            grade: Any,
            *,
            name: str,
    ) -> Array:
        """Validate one ordered standard-coordinate signature block."""
        del grade, name
        return block

    def _prepare_ordered_signature_pairing_operands(
            self,
            standard_words: Union[DenseElem, DenseElemFirstOn],
            ordered_standard_signature: Union[DenseElem, DenseElemFirstOn],
            *,
            words_first_on: bool = False,
            standard_first_on: bool = False,
    ) -> tuple[Any, Any]:
        """Validate operands before contraction with an ordered signature."""
        spec = getattr(ordered_standard_signature, "spec", None)
        include_scalar = getattr(spec, "include_scalar", None)
        if include_scalar is not None and standard_first_on == include_scalar:
            expected = "omit" if standard_first_on else "include"
            raise ValueError(
                f"standard_first_on={standard_first_on} requires its tensor "
                f"to {expected} the scalar block."
            )
        del words_first_on
        return standard_words, ordered_standard_signature

    def _pair_standard_block_with_ordered_signature(
            self,
            standard_words: Array,
            ordered_standard_signature: Array,
            *,
            grade: Any,
    ) -> Array:
        """Contract one explicit grade with an ordered signature block."""
        ordered_standard_signature = (
            self._validate_ordered_signature_pairing_block(
                ordered_standard_signature,
                grade,
                name="standard_tensor",
            )
        )
        return self.tensor_inner_product_homogeneous(
            standard_words,
            ordered_standard_signature,
        )

    def _pair_standard_block_with_standard_tensor(
            self,
            standard_words: Array,
            standard_tensor: Array,
            *,
            grade: Any,
            standard_partially_symmetrized: bool,
    ) -> Array:
        """Dispatch one standard-coordinate pairing by symmetrization."""
        if standard_partially_symmetrized:
            raise ValueError(
                "standard_partially_symmetrized=True requires "
                "a partially symmetrized core."
            )
        return self._pair_standard_block_with_ordered_signature(
            standard_words,
            standard_tensor,
            grade=grade,
        )

    def _pair_standard_words_with_ordered_signature(
            self,
            standard_words: Union[DenseElem, DenseElemFirstOn],
            ordered_standard_signature: Union[DenseElem, DenseElemFirstOn],
            *,
            words_first_on: bool,
            standard_first_on: bool,
    ) -> Array:
        """Contract standard-coordinate words with an ordered signature."""
        standard_words, ordered_standard_signature = (
            self._prepare_ordered_signature_pairing_operands(
                standard_words,
                ordered_standard_signature,
                words_first_on=words_first_on,
                standard_first_on=standard_first_on,
            )
        )
        return self._standard_tensor_inner_product(
            standard_words,
            ordered_standard_signature,
            a_first_on=words_first_on,
            b_first_on=standard_first_on,
        )

    def _pair_standard_words_with_standard_tensor(
            self,
            standard_words: Union[DenseElem, DenseElemFirstOn],
            standard_tensor: Union[DenseElem, DenseElemFirstOn],
            *,
            words_first_on: bool,
            standard_first_on: bool,
    ) -> Array:
        """Pair standard-coordinate words with an ordered tensor."""
        return self._pair_standard_words_with_ordered_signature(
            standard_words,
            standard_tensor,
            words_first_on=words_first_on,
            standard_first_on=standard_first_on,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("grade", "standard_partially_symmetrized"),
        dynamic_batch=("words", "standard_tensor"),
    )
    def tensor_shear_pairing_homogeneous(
            self,
            words: Array,
            standard_tensor: Array,
            *,
            grade: Any,
            standard_partially_symmetrized: bool = False,
    ) -> Array:
        """Pair one word block with a standard-coordinate block."""
        if grade is None:
            raise TypeError(
                "tensor_shear_pairing_homogeneous requires grade=."
            )
        if not isinstance(standard_partially_symmetrized, bool):
            raise TypeError(
                "standard_partially_symmetrized must be a boolean, got "
                f"{standard_partially_symmetrized!r}."
            )
        grade = self.normalize_truncation(grade)
        standard_words = (
            self._native_words_to_standard_block(
                words,
                grade=grade,
            )
        )
        return self._pair_standard_block_with_standard_tensor(
            standard_words,
            standard_tensor,
            grade=grade,
            standard_partially_symmetrized=standard_partially_symmetrized,
        )

    @dummy_jit(
        static_argnums=0,
        static_argnames=("words_first_on", "standard_first_on"),
        dynamic_batch=("words", "standard_tensor"),
    )
    def tensor_shear_pairing(
            self,
            words: Union[DenseElem, DenseElemFirstOn],
            standard_tensor: Union[DenseElem, DenseElemFirstOn],
            *,
            words_first_on: bool = False,
            standard_first_on: bool = False,
    ) -> Array:
        """Pair words with a compatible standard-coordinate tensor."""
        for name, value in (
            ("words_first_on", words_first_on),
            ("standard_first_on", standard_first_on),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean, got {value!r}.")
        natural = self._coordinate_natural_truncation(
            words,
            first_on=words_first_on,
        )
        standard_words = (
            self._native_words_to_standard(
                words,
                trunc=natural,
                first_on=words_first_on,
            )
        )
        return self._pair_standard_words_with_standard_tensor(
            standard_words,
            standard_tensor,
            words_first_on=words_first_on,
            standard_first_on=standard_first_on,
        )

    def _standard_adjoint_left_homogeneous(
            self, Ai: Array, Yni: Array
    ) -> Array:
        y = Yni.reshape((*Yni.shape[:-1], Ai.shape[-1], -1))
        return (Ai[..., :, None] * y).sum(axis=-2)

    def _standard_adjoint_right_homogeneous(
            self, Bj: Array, Ynj: Array
    ) -> Array:
        y = Ynj.reshape((*Ynj.shape[:-1], -1, Bj.shape[-1]))
        return (y * Bj[..., None, :]).sum(axis=-1)

    @dummy_jit(static_argnums=0, dynamic_batch=("Ai", "Yni"))
    def tensor_adjoint_left_homogeneous(self, Ai: Array, Yni: Array) -> Array:
        """
        Homogeneous left-adjoint contraction for degree i.

        Given
            Ai.shape  == batch + (d**i,)
            Yni.shape == batch + (d**(n+i),)
        reshape Yni as (..., d**i, -1) and contract the first width with Ai:
            out = Ai^T • Yni_(..., i, N)  ->  shape batch + (N,)
        """
        return self._standard_adjoint_left_homogeneous(Ai, Yni)

    @dummy_jit(static_argnums=0, dynamic_batch=("Bj", "Ynj"))
    def tensor_adjoint_right_homogeneous(self, Bj: Array, Ynj: Array) -> Array:
        """
        Homogeneous right-adjoint contraction for degree j.

        Given
            Bj.shape  == batch + (d**j,)
            Ynj.shape == batch + (d**(n+j),)
        reshape Ynj as (..., -1, d**j) and contract the last width with Bj:
            out = Ynj_(..., N, j) • Bj  ->  shape batch + (N,)
        """
        return self._standard_adjoint_right_homogeneous(Bj, Ynj)

    def _standard_adjoint_left_block(
            self,
            Wi: Array,
            Yni: Array,
            multiplier_grade: int,
            target_grade: int,
            output_grade: int,
    ) -> Array:
        del multiplier_grade, target_grade, output_grade
        return self._standard_adjoint_left_homogeneous(Wi, Yni)

    def _standard_adjoint_right_block(
            self,
            Wi: Array,
            Yni: Array,
            multiplier_grade: int,
            target_grade: int,
            output_grade: int,
    ) -> Array:
        del multiplier_grade, target_grade, output_grade
        return self._standard_adjoint_right_homogeneous(Wi, Yni)

    def _adjoint_left_block(
            self,
            Wi: Array,
            Yni: Array,
            multiplier_grade: int,
            target_grade: int,
            output_grade: int,
    ) -> Array:
        """Grade-aware left-adjoint hook for the shared contraction driver."""
        return self._standard_adjoint_left_block(
            Wi, Yni, multiplier_grade, target_grade, output_grade
        )

    def _adjoint_right_block(
            self,
            Wi: Array,
            Yni: Array,
            multiplier_grade: int,
            target_grade: int,
            output_grade: int,
    ) -> Array:
        """Grade-aware right-adjoint hook for the shared contraction driver."""
        return self._standard_adjoint_right_block(
            Wi, Yni, multiplier_grade, target_grade, output_grade
        )

    def _standard_adjoint_schedule(
            self,
            W: Union[DenseElem, DenseElemFirstOn],
            Y: Union[DenseElem, DenseElemFirstOn],
            trunc: Optional[int],
            *,
            contract: Callable[[Any, Any, Any, Any, Any], Any],
            w_first_on: bool,
            y_first_on: bool,
            first_on_out: bool,
    ) -> GradedContractionSchedule:
        """Resolve the total-degree tuple layout for an adjoint action."""
        W, Y = tuple(W), tuple(Y)
        if len(W) == 0 or len(Y) == 0:
            if trunc is not None:
                self.normalize_truncation(trunc)
            return GradedContractionSchedule(
                grades=tuple(),
                pairs=lambda _grade: tuple(),
                multiplier_block=lambda grade: W[grade],
                target_block=lambda grade: Y[grade],
                assemble=tuple,
            )

        w0 = 1 if w_first_on else 0
        y0 = 1 if y_first_on else 0
        w_last = w0 + len(W) - 1
        y_last = y0 + len(Y) - 1
        start = 1 if first_on_out else 0
        n_min = max(start, y0 - w_last)
        n_max = self._effective_truncation(trunc, y_last - w0)

        if n_max < start:
            return GradedContractionSchedule(
                grades=tuple(),
                pairs=lambda _grade: tuple(),
                multiplier_block=lambda grade: W[grade],
                target_block=lambda grade: Y[grade],
                assemble=tuple,
            )

        def zero_for_degree(n: int):
            i_min = max(w0, y0 - n)
            i_max = min(w_last, y_last - n)
            if i_min > i_max:
                raise ValueError(
                    f"No contributing term available to infer shape for degree {n}."
                )
            term = contract(
                W[i_min - w0], Y[n + i_min - y0], i_min, n + i_min, n
            )
            return self.xp.zeros_like(term)

        return GradedContractionSchedule(
            grades=tuple(range(start, n_max + 1)),
            pairs=lambda n: tuple() if n < n_min else (
                (i, n + i)
                for i in range(max(w0, y0 - n), min(w_last, y_last - n) + 1)
            ),
            multiplier_block=lambda i: W[i - w0],
            target_block=lambda j: Y[j - y0],
            assemble=tuple,
            zero_block=zero_for_degree,
        )

    def _adjoint_schedule(
            self,
            W,
            Y,
            trunc,
            *,
            contract,
            w_first_on: bool,
            y_first_on: bool,
            first_on_out: bool,
    ) -> GradedContractionSchedule:
        return self._standard_adjoint_schedule(
            W,
            Y,
            trunc,
            contract=contract,
            w_first_on=w_first_on,
            y_first_on=y_first_on,
            first_on_out=first_on_out,
        )

    def _standard_tensor_adjoint_product(
            self,
            W,
            Y,
            trunc=None,
            side: Literal["left", "right"] = "left",
            *,
            w_first_on: bool = False,
            y_first_on: bool = False,
            first_on_out: bool = False,
    ):
        """Execute an ordinary-product adjoint without coordinate redispatch."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        contract = (
            self._standard_adjoint_left_block
            if side == "left"
            else self._standard_adjoint_right_block
        )
        schedule = self._standard_adjoint_schedule(
            W,
            Y,
            trunc,
            contract=contract,
            w_first_on=w_first_on,
            y_first_on=y_first_on,
            first_on_out=first_on_out,
        )
        blocks = graded_contraction(
            schedule.grades,
            pairs=schedule.pairs,
            multiplier_block=schedule.multiplier_block,
            target_block=schedule.target_block,
            contract_block=schedule.contract_block or contract,
            zero_block=schedule.zero_block,
        )
        return schedule.assemble(blocks)

    def tensor_adjoint_product(
            self,
            W: Union[DenseElem, DenseElemFirstOn],
            Y: Union[DenseElem, DenseElemFirstOn],
            trunc: Any = None,
            side: Literal["left", "right"] = "left",
            *,
            w_first_on: bool = False,
            y_first_on: bool = False,
            first_on_out: bool = False,
    ) -> Union[DenseElem, DenseElemFirstOn]:
        """
        Compute the graded adjoint product of two truncated tensor-algebra elements.

        This is the shared implementation behind the left and right adjoint actions.
        It contracts homogeneous levels of ``W`` against shifted homogeneous levels
        of ``Y`` and sums all contributions that land in the same output degree.

        Degree convention
        -----------------
        Let ``W_i`` denote the degree-``i`` level of ``W`` and ``Y_j`` the degree-``j``
        level of ``Y``. For each output degree ``n``, this routine computes

            Z_n = sum_i Adj(W_i, Y_{n+i}),

        where ``Adj`` is either the left or right homogeneous adjoint contraction,
        depending on ``side``.

        More precisely:
        - for ``side="left"``, use ``tensor_adjoint_left_homogeneous(W_i, Y_{n+i})``;
        - for ``side="right"``, use ``tensor_adjoint_right_homogeneous(W_i, Y_{n+i})``.

        Input format
        --------------------
        The inputs may be stored either as dense graded elements or as first-on
        graded elements:

        - dense:
            ``(A_0, A_1, ..., A_N)``
        - first-on:
            ``(A_1, A_2, ..., A_N)``

        The flags ``w_first_on`` and ``y_first_on`` specify which convention is used
        for ``W`` and ``Y`` respectively.

        Output format
        ---------------------
        The returned tuple is intended to be either

        - dense, if ``first_on_out=False``:
            ``(Z_0, Z_1, ..., Z_M)``
        - first-on, if ``first_on_out=True``:
            ``(Z_1, Z_2, ..., Z_M)``

        where ``M`` is the largest output degree allowed by the input ranges and by
        ``trunc``.

        Important
        ---------
        This routine should return output in a canonical graded format:
        - dense output must begin at degree 0,
        - first-on output must begin at degree 1.

        If the lowest nonzero computable degree is higher than that starting degree,
        the missing lower degrees should be represented by zero levels rather than
        being silently skipped. Otherwise the returned tuple no longer has a well-defined
        dense/first-on interpretation.

        Parameters
        ----------
        W :
            Multiplier graded element. Its stored levels are interpreted as starting
            at degree 0 or degree 1 according to ``w_first_on``.
        Y :
            Target graded element. Its stored levels are interpreted as starting
            at degree 0 or degree 1 according to ``y_first_on``.
        trunc :
            Optional maximum output degree. If ``None``, all output degrees permitted
            by the available levels of ``W`` and ``Y`` are produced.
        side :
            Which adjoint action to use:
            - ``"left"``  -> left homogeneous adjoint contraction,
            - ``"right"`` -> right homogeneous adjoint contraction.
        w_first_on :
            Whether ``W`` is stored in first-on format.
        y_first_on :
            Whether ``Y`` is stored in first-on format.
        first_on_out :
            Whether to return the result in first-on format.

        Returns
        -------
        DenseElem or DenseElemFirstOn
            The graded adjoint product, truncated at degree ``trunc`` if requested,
            in the storage convention specified by ``first_on_out``.

        Notes
        -----
        If ``W`` has degrees ``i`` in some range and ``Y`` has degrees ``j`` in some
        range, then the admissible output degrees are those for which at least one
        pair ``(i, j)`` satisfies ``j = n + i``. This determines the natural output
        degree window before truncation is applied.
        """
        return self._standard_tensor_adjoint_product(
            W,
            Y,
            trunc,
            side=side,
            w_first_on=w_first_on,
            y_first_on=y_first_on,
            first_on_out=first_on_out,
        )

    # ----------------------------------------------------------------------
    # Polynomial / Series Operations
    # ----------------------------------------------------------------------

    def _series_argument(self, X: DenseElemFirstOn) -> tuple[str, Any]:
        """Classify a positive-level total-degree series argument statically."""
        X = tuple(X)
        if len(X) == 0:
            return "empty", X
        if len(X) == 1:
            return "first", X[0]
        return "general", X

    def _series_validate_left_factor(self, g: DenseElem) -> DenseElem:
        """Validate/normalize the dense left factor of ``g * exp(X)``."""
        return g

    def _series_validate_exponential_argument(self, X):
        """Validate the native-coordinate argument before empty fast paths."""
        return X

    def _series_order(self, trunc: int) -> int:
        """Nilpotence order used by formal-series orchestration."""
        return trunc

    def _series_validate_log_argument(self, X: DenseElemFirstOn) -> DenseElemFirstOn:
        return X

    def _series_truncate_left_factor(self, g: DenseElem, trunc: int) -> DenseElem:
        return g[:trunc + 1]

    def _fmexp_first_level(
            self,
            g: DenseElem,
            z: Array,
            *,
            trunc: int,
    ) -> DenseElem:
        """Pruned total-degree Horner specialization through shared grades."""
        g = tuple(g)
        if not g:
            raise ValueError("g must include the scalar level.")
        z = self.xp.asarray(z)
        layout = self.resolve_layout(trunc, include_scalar=True)

        def base_block(grade: int, like):
            if grade < len(g):
                return g[grade]
            if like is None:
                raise ValueError("g must include the scalar level.")
            return self.xp.zeros_like(like)

        return graded_horner_first_level(
            layout,
            max_order=trunc,
            base_block=base_block,
            generator_blocks=self._generator_blocks(z, layout=layout),
            right_generator_output_block=(
                self._right_multiply_generator_output_block
            ),
            assemble=tuple,
        )

    def _series_exponential_data(
            self,
            X: DenseElemFirstOn,
            *,
            trunc: int,
            left_factor: DenseElem,
    ) -> tuple[DenseElem, DenseElem, int]:
        """Construct the dense total-degree argument and identity."""
        del left_factor
        X = tuple(X)
        zero0 = self.xp.zeros_like(X[0][..., :1])
        one0 = self.xp.ones_like(zero0)
        return (zero0,) + tuple(X[:trunc]), (one0,), trunc

    def _series_identity_for_argument(
            self,
            X: DenseElemFirstOn,
            *,
            trunc: int,
    ) -> DenseElem:
        del trunc
        return (self.xp.ones_like(X[0][..., :1]),)

    def _series_exponential_zero(
            self,
            *,
            trunc: int,
            output_zero_level: bool,
    ) -> DenseElem:
        del trunc
        return (self.xp.asarray([1.0]),) if output_zero_level else tuple()

    def _series_logarithm_data(
            self,
            X: DenseElemFirstOn,
            *,
            trunc: int,
    ) -> tuple[DenseElem, DenseElem, int]:
        X = tuple(X)
        zero0 = self.xp.zeros_like(X[0][..., :1])
        H = (zero0,) + X
        zero = (zero0,) + tuple(self.xp.zeros_like(level) for level in X)
        return H, zero, trunc

    def _series_logarithm_zero(
            self,
            *,
            trunc: int,
            output_zero_level: bool,
    ) -> DenseElem:
        del trunc
        return (self.xp.asarray([0.0]),) if output_zero_level else tuple()

    def _series_output(
            self,
            result: Any,
            *,
            trunc: Any,
            output_zero_level: bool,
    ) -> Any:
        del trunc
        return result if output_zero_level else result[1:]

    # The following aliases are the coordinate-neutral series boundary used
    # after a shear core has transported values to standard coordinates.
    # Layout specializations with coordinate-tagged containers may
    # override them without weakening their public native validation.
    def _standard_series_validate_left_factor(self, g):
        return self._series_validate_left_factor(g)

    def _standard_series_validate_exponential_argument(self, X):
        return self._series_validate_exponential_argument(X)

    def _standard_series_truncate_left_factor(self, g, trunc):
        return self._series_truncate_left_factor(g, trunc)

    def _standard_series_exponential_data(self, X, *, trunc, left_factor):
        return self._series_exponential_data(
            X, trunc=trunc, left_factor=left_factor
        )

    def _standard_series_validate_log_argument(self, X):
        return self._series_validate_log_argument(X)

    def _standard_series_logarithm_data(self, X, *, trunc):
        return self._series_logarithm_data(X, trunc=trunc)

    def _standard_series_logarithm_zero(
            self,
            *,
            trunc,
            output_zero_level: bool,
    ):
        return self._series_logarithm_zero(
            trunc=trunc,
            output_zero_level=output_zero_level,
        )

    def _standard_series_output(self, result, *, trunc, output_zero_level):
        return self._series_output(
            result,
            trunc=trunc,
            output_zero_level=output_zero_level,
        )

    def _standard_tensor_fmexp(
            self,
            g,
            X,
            *,
            trunc,
            output_zero_level: bool,
    ):
        """Formal multiplicative exponential for the ordinary product."""
        active = self.normalize_truncation(trunc)
        g = self._standard_series_validate_left_factor(g)
        X = self._standard_series_validate_exponential_argument(X)
        kind, argument = self._series_argument(X)
        if self._series_order(active) == 0 or kind == "empty":
            result = self._standard_series_truncate_left_factor(g, active)
        elif kind == "first":
            result = self._fmexp_first_level(g, argument, trunc=active)
        else:
            H, identity, max_order = self._standard_series_exponential_data(
                argument, trunc=active, left_factor=g
            )
            E = formal_exponential_series(
                H,
                identity=identity,
                max_order=max_order,
                product=lambda A, B: self._standard_tensor_product(
                    A, B, trunc=active
                ),
                summation=lambda A, B: self._standard_tensor_summation(
                    A, B, trunc=active
                ),
                scalar_multiply=self._standard_tensor_scalar_multiply,
            )
            result = self._standard_tensor_product(g, E, trunc=active)
        return self._standard_series_output(
            result,
            trunc=active,
            output_zero_level=output_zero_level,
        )

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batch=("g", "X"))
    def tensor_fmexp(
            self,
            g: DenseElem,  # graded element with explicit level-0
            X: DenseElemFirstOn,  # levels start at 1; treat X₀ ≡ 0
            *,
            trunc: Any = None,
            output_zero_level: bool = True,
    ) -> Tuple[Array, ...]:
        """
        Efficient formal multiplicative exponential: compute Y = g ⊗ exp(X).

        Conventions
        -----------
        - `g` is a dense graded element (g₀, g₁, …).
        - `X` has levels starting at 1: (X₁, X₂, …), with X₀ ≡ 0.

        Algorithm
        ---------
        - For the pure degree-1 case (`len(X) == 1`), use the fused Horner scheme
          for `g ⊗ exp(X₁)`.
        - In the general case, build the truncated exponential
              exp(X) = I + X + X^{⊗2}/2! + ... + X^{⊗trunc}/trunc!
          exactly via the truncated tensor power series, and then compute
              Y = g ⊗ exp(X).

        Parameters
        ----------
        g : DenseElem
            Left factor with explicit level-0.
        X : DenseElemFirstOn
            Exponent argument with levels starting at degree 1.
        trunc : int
            Maximum degree (inclusive) of the output.
        output_zero_level : bool, default True
            If False, drop degree-0 from the returned tuple.

        Returns
        -------
        Tuple[Array, ...]
            (Y₀, …, Y_trunc) for `output_zero_level=True`; otherwise (Y₁, …, Y_trunc).
        """
        return self._standard_tensor_fmexp(
            g,
            X,
            trunc=trunc,
            output_zero_level=output_zero_level,
        )

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batch=("X",))
    def tensor_exponential(
            self, X: DenseElemFirstOn, *, trunc: Any = None, output_zero_level: bool = True
    ) -> Tuple[Array, ...]:
        """
        Algebra exponential for inputs starting at degree 1 (treat X₀ ≡ 0).

        Implemented via the efficient primitive:
            exp(X) = I ⊗ exp(X)  with  I = (1,)  (level-0 only)
        """
        active = self.normalize_truncation(trunc)
        X = self._series_validate_exponential_argument(X)
        kind, argument = self._series_argument(X)
        if kind == "empty":
            return self._series_exponential_zero(
                trunc=active,
                output_zero_level=output_zero_level,
            )
        identity = self._series_identity_for_argument(X, trunc=active)
        return self.tensor_fmexp(
            identity,
            X,
            trunc=active,
            output_zero_level=output_zero_level,
        )

    def _standard_tensor_logarithm(
            self,
            X,
            *,
            trunc,
            output_zero_level: bool,
    ):
        """Formal logarithm for the ordinary product."""
        active = self.normalize_truncation(trunc)
        X = self._standard_series_validate_log_argument(X)
        kind, argument = self._series_argument(X)
        if kind == "empty":
            return self._standard_series_logarithm_zero(
                trunc=active,
                output_zero_level=output_zero_level,
            )
        H, zero, max_order = self._standard_series_logarithm_data(
            X, trunc=active
        )
        result = formal_logarithm_series(
            H,
            zero=zero,
            max_order=max_order,
            product=lambda A, B: self._standard_tensor_product(
                A, B, trunc=active
            ),
            summation=lambda A, B: self._standard_tensor_summation(
                A, B, trunc=active
            ),
            scalar_multiply=self._standard_tensor_scalar_multiply,
        )
        return self._standard_series_output(
            result,
            trunc=active,
            output_zero_level=output_zero_level,
        )

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batch=("X",))
    def tensor_logarithm(self, X: DenseElemFirstOn, *, trunc: Any = None, output_zero_level: bool = True) -> Tuple[Array, ...]:
        """
        Algebra logarithm with input levels starting at 1.

        Convention
        ----------
        Input X = (X₁, X₂, …) is interpreted with X₀ ≡ 1.
        Let H := (0, X₁, X₂, …) (zero scalar part). Then
            log(1 + H) = (0, ∑_{n≥1} (-1)^{n+1} H^{⊗ n} / n),
        truncated to degree `trunc`. No padding of higher degrees is performed; the
        convolution grows degrees naturally via `tensor_product(..., trunc=trunc)`.

        Parameters
        ----------
        X : DenseElemFirstOn
            Levels start at degree 1, last dim d**k, shared batch shape.
        trunc : int
            Maximum degree (inclusive) of the output.
        output_zero_level : bool, default True
            Whether to include degree-0 in the returned tuple (it's identically zero).

        Returns
        -------
        DenseElem (or degrees 1..trunc when output_zero_level=False)
        """
        return self._standard_tensor_logarithm(
            X,
            trunc=trunc,
            output_zero_level=output_zero_level,
        )

    def tensor_slice(self, A: DenseElem) -> _TensorSliceProxy:
        """
        Return a proxy that applies an index or slice to every level of ``A``.

        Usage::

            td.tensor_slice(A)[i:j]       # → tuple(lvl[i:j] for lvl in A)
            td.tensor_slice(A)[..., 0]    # → tuple(lvl[..., 0] for lvl in A)
        """
        A = self._validate_graded_element(A, name="A")
        return _TensorSliceProxy(A)

    # ----------------------------------------------------------------------
    # API Transformers (from_flat / to_flat / densify)
    # ----------------------------------------------------------------------

    def tensor_densify(self, levels: Elem) -> DenseElem:
        """
        Turn a possibly sparse graded element (levels may be None) into a **dense** one.

        Rules
        -----
        • If a positive degree (k ≥ 1) level is present, infer the base dimension `d`
          from the first such level:  width(level_k) must equal d**k.
          - All other present levels are validated against this `d` and shared batch shape.
        • Missing levels are filled with exact zeros of matching batch shape and width d**k
          (for k=0 the width is 1).
        • If `d` **cannot** be inferred (because the input is empty, all levels are None,
          or only degree-0 is provided), return a canonical zero scalar:
             `(np.array([0.0]),)`.

        Parameters
        ----------
        levels : Tuple[Optional[Array], ...]
            Packed levels `(X₀, X₁, ..., X_N)`, where any `X_k` may be `None`.

        Returns
        -------
        Tuple[Array, ...]
            Dense graded element `(Y₀, Y₁, ..., Y_N)` with no `None`s.

        Raises
        ------
        ValueError
            If present levels disagree on batch shape, or a present width at degree k
            does not equal `d**k`.
        """
        levels = tuple(levels)

        # prune trailing Nones
        last_present = -1
        for i in range(len(levels) - 1, -1, -1):
            if levels[i] is not None:
                last_present = i
                break
        if last_present == -1:
            return (self.xp.asarray([0.0]),)

        levels = levels[: last_present + 1]  # drop trailing Nones
        present = [(k, L) for k, L in enumerate(levels) if L is not None]

        # infer d from the smallest present positive degree; if none, d=1
        pos = [(k, L) for k, L in present if k > 0]
        if pos:
            k0, L0 = min(pos, key=lambda kv: kv[0])
            width_k0 = L0.shape[-1]
            d = int(round(width_k0 ** (1.0 / k0)))
            if d < 1 or d ** k0 != width_k0:
                raise ValueError(
                    f"tensor_densify: cannot infer integer base d from degree {k0} width {width_k0}."
                )
        else:
            d = 1  # only degree-0 is present

        N = len(levels) - 1
        ref_max = levels[N] if levels[N] is not None else next(L for _, L in reversed(present))
        batch = ref_max.shape[:-1]
        max_width = ref_max.shape[-1]

        # validate present levels
        for k, L in present:
            exp_w = 1 if k == 0 else d ** k
            if L.shape[:-1] != batch:
                raise ValueError("tensor_densify: batch shapes differ across levels.")
            if L.shape[-1] != exp_w:
                raise ValueError(
                    f"tensor_densify: degree {k} has width {L.shape[-1]} but expected {exp_w}."
                )

        # fill missing levels using slice -> zeros_like (no shape kwarg)
        out = []
        for k in range(N + 1):
            L = levels[k]
            if L is not None:
                out.append(L)
            else:
                w = 1 if k == 0 else d ** k
                # guaranteed w <= max_width because k <= N
                out.append(self.xp.zeros_like(ref_max[..., :w]))
        return tuple(out)

    @dummy_jit(static_argnums=0, static_argnames=("dim", "insert_zero_level"), dynamic_batch=("X",))
    def tensor_from_flat(
            self,
            flat: Array,
            dim: int,
            insert_zero_level: Optional[Union[bool, float]] = None,
    ) -> DenseElem:
        """
        Split a flattened graded tensor into a **dense** (X₀, X₁, ..., X_N).

        Layouts supported on the last axis of `flat`:
          • [X₀ | X₁ | ... | X_N]  with widths 1, d, d**2, ...
          • [X₁ | ... | X_N]       with widths d, d**2, ...

        Degree-0 policy (always return a DenseElem):
          - insert_zero_level is None  → if X₀ missing, **insert zeros**; if present, keep.
          - insert_zero_level is True  → set/insert X₀ = **ones**.
          - insert_zero_level is False → set/insert X₀ = **zeros**.
          - insert_zero_level is float → set/insert X₀ = that **scalar**.

        Parameters
        ----------
        flat : Array
            Packed array with the concatenated degrees on the last axis.
        dim : int
            Base dimension `d` so that width(Xₖ) = d**k.
        insert_zero_level : None | bool | float, optional
            Degree-0 policy as described above.

        Returns
        -------
        DenseElem
            A tuple (X₀, X₁, ..., X_N), each with matching batch shape and last dim d**k.
        """
        # Coerce to array without asarray()
        flat = self.xp.asarray(flat)
        total = int(flat.shape[-1])

        def find_N(total_len: int, start: int) -> Optional[int]:
            acc, k = 0, start
            while acc < total_len:
                acc += dim ** k
                if acc == total_len:
                    return k
                k += 1
            return None

        N0 = find_N(total, 0)  # assumes X0 present
        N1 = find_N(total, 1)  # assumes X0 absent
        if N0 is None and N1 is None:
            raise ValueError("tensor_from_flat: last axis is not a valid sum of powers of `dim`.")

        # Prefer layout with X0 if both fit
        start = 0 if N0 is not None else 1
        N = N0 if start == 0 else N1  # type: ignore[assignment]

        parts = []
        off = 0
        X0 = None
        if start == 0:
            # X0 is present in `flat`: slice it directly
            X0 = flat[..., :1]
            off = 1

        # Slice remaining parts: widths dim^1, dim^2, ..., dim^N
        for k in range(1, N + 1):
            w = dim ** k
            parts.append(flat[..., off:off + w])
            off += w

        # Materialize/override X0 using only *_like without shape kwarg (slice to target shape first)
        if isinstance(insert_zero_level, (int, float)):
            X0 = self.xp.ones_like(flat[..., :1]) * float(insert_zero_level)
        elif insert_zero_level is True:
            X0 = self.xp.ones_like(flat[..., :1])
        elif insert_zero_level is False:
            X0 = self.xp.zeros_like(flat[..., :1])
        else:  # None
            if X0 is None:  # missing → insert zeros
                X0 = self.xp.zeros_like(flat[..., :1])

        return (X0,) + tuple(parts)

    @dummy_jit(static_argnums=0, static_argnames=("start_at_level_one",), dynamic_batch=("levels",))
    def tensor_to_flat(self, levels: DenseElem, *, start_at_level_one: bool = False) -> Array:
        """
        Concatenate per-degree levels into a single flattened array.

        Parameters
        ----------
        levels : sequence of ndarray
            ``(X₀, X₁, ..., X_N)`` with matching batch shapes.
        start_at_level_one : bool, optional
            If ``True``, pack only degrees ``1..N`` (drop ``X₀`` in the output).
            Otherwise pack ``0..N``.

        Returns
        -------
        ndarray
            The concatenation along the last axis.
        """
        levels = tuple(levels)
        if not levels:
            return self.xp.asarray([], dtype=float)
        scalar_reference = self.xp.asarray(levels[0])
        if start_at_level_one:
            levels = levels[1:]
        return (
            self.xp.concat(levels, axis=-1)
            if levels
            else scalar_reference[..., :0]
        )

    # ----------------------------------------------------------------------
    # Matrix Tensor Operations
    # ----------------------------------------------------------------------

    def _matrix_map_schedule(
            self,
            A: DenseElem,
            trunc: Optional[int],
    ) -> GradedMapSchedule:
        """Resolve total-degree levels for matrix left/right maps."""
        A = tuple(A)
        N = self._effective_truncation(trunc, len(A) - 1)
        return GradedMapSchedule(
            grades=tuple(range(N + 1)),
            block=lambda grade: A[grade],
            assemble=tuple,
        )

    def _standard_matrix_product_schedule(
            self,
            A: DenseElem,
            B: DenseElem,
            trunc: Optional[int],
            *,
            row_axis: int,
            col_axis: int,
    ) -> GradedConvolutionSchedule:
        """Resolve total-degree levels for a matrix-valued graded product."""
        del row_axis, col_axis
        A, B = tuple(A), tuple(B)
        if len(A) == 0 or len(B) == 0:
            if trunc is not None:
                self.normalize_truncation(trunc)
            return GradedConvolutionSchedule(
                grades=tuple(),
                splits=lambda _grade: tuple(),
                left_contains=lambda _grade: False,
                right_contains=lambda _grade: False,
                left_block=lambda grade: A[grade],
                right_block=lambda grade: B[grade],
                assemble=tuple,
            )
        N = self._effective_truncation(trunc, len(A) + len(B) - 2)
        return GradedConvolutionSchedule(
            grades=tuple(range(N + 1)),
            splits=lambda grade: (
                (i, grade - i)
                for i in range(
                    max(0, grade - (len(B) - 1)), min(len(A) - 1, grade) + 1
                )
            ),
            left_contains=lambda grade: 0 <= grade < len(A),
            right_contains=lambda grade: 0 <= grade < len(B),
            left_block=lambda grade: A[grade],
            right_block=lambda grade: B[grade],
            assemble=tuple,
        )

    def _matrix_product_schedule(
            self,
            A,
            B,
            trunc,
            *,
            row_axis: int,
            col_axis: int,
    ) -> GradedConvolutionSchedule:
        return self._standard_matrix_product_schedule(
            A,
            B,
            trunc,
            row_axis=row_axis,
            col_axis=col_axis,
        )

    def _canonicalize_matrix_axes(
            self,
            X: Array,
            row_axis: int,
            col_axis: int,
    ) -> Tuple[Array, int, int]:
        """
        Move the matrix axes of ``X`` to the canonical positions ``(-3, -2)``.

        Parameters
        ----------
        X : ndarray
            Array with a final tensor-coordinate axis.
        row_axis, col_axis : int
            Positions of the row and column axes.

        Returns
        -------
        Xc : ndarray
            Array with row and column axes moved to ``(-3, -2)``.
        row_axis, col_axis : int
            Normalized original axis positions.
        """
        ndim = X.ndim
        row_axis = row_axis % ndim
        col_axis = col_axis % ndim

        if row_axis == col_axis:
            raise ValueError("row_axis and col_axis must be distinct.")
        if row_axis == ndim - 1 or col_axis == ndim - 1:
            raise ValueError("row_axis and col_axis may not coincide with the tensor axis.")

        Xc = self.xp.moveaxis(X, (row_axis, col_axis), (-3, -2))
        return Xc, row_axis, col_axis

    def _restore_matrix_axes(
            self,
            X: Array,
            row_axis: int,
            col_axis: int,
    ) -> Array:
        """
        Restore canonical matrix axes ``(-3, -2)`` to ``(row_axis, col_axis)``.
        """
        return self.xp.moveaxis(X, (-3, -2), (row_axis, col_axis))

    @dummy_jit(static_argnums=0, static_argnames=("row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_right_homogeneous(
            self,
            A: Array,
            M: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        """
        Right-multiply a homogeneous matrix-valued tensor-algebra coefficient by
        a numeric matrix.

        The default axis convention is

            ``batch + (T?, n, k, d_r)``,

        where the final axis stores tensor coordinates, an optional time axis
        precedes the matrix axes, and the matrix axes are ``(row_axis, col_axis)``.

        Parameters
        ----------
        A : ndarray
            Homogeneous matrix-valued coefficient with shape
            ``batch + (T?, n, k, d_r)`` up to axis placement.
        M : ndarray
            Numeric matrix with shape ``(..., k, l)``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in ``A``.

        Returns
        -------
        ndarray
            Homogeneous coefficient with shape
            ``batch + (T?, n, l, d_r)`` up to axis placement.
        """
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)

        a = self.xp.expand_dims(A, axis=-2)  # ... n k 1 a
        m = self.xp.expand_dims(M, axis=-1)  # ... k l 1
        m = self.xp.expand_dims(m, axis=-4)  # ... 1 k l 1
        out = (a * m).sum(axis=-3)  # ... n l a

        return self._restore_matrix_axes(out, row_axis, col_axis)

    @dummy_jit(static_argnums=0, static_argnames=("row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_left_homogeneous(
            self,
            M: Array,
            A: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        """
        Left-multiply a homogeneous matrix-valued tensor-algebra coefficient by
        a numeric matrix.

        The default axis convention is

            ``batch + (T?, k, l, d_r)``,

        where the final axis stores tensor coordinates, an optional time axis
        precedes the matrix axes, and the matrix axes are ``(row_axis, col_axis)``.

        Parameters
        ----------
        M : ndarray
            Numeric matrix with shape ``(..., n, k)``.
        A : ndarray
            Homogeneous matrix-valued coefficient with shape
            ``batch + (T?, k, l, d_r)`` up to axis placement.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in ``A``.

        Returns
        -------
        ndarray
            Homogeneous coefficient with shape
            ``batch + (T?, n, l, d_r)`` up to axis placement.
        """
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)

        m = self.xp.expand_dims(M, axis=-1)  # ... n k 1
        m = self.xp.expand_dims(m, axis=-1)  # ... n k 1 1
        a = self.xp.expand_dims(A, axis=-4)  # ... 1 k l a
        out = (m * a).sum(axis=-3)  # ... n l a

        return self._restore_matrix_axes(out, row_axis, col_axis)

    def _standard_matrix_product_homogeneous(
            self,
            A: Array,
            B: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        """Raw ordinary matrix-valued homogeneous concatenation."""
        requested_row_axis, requested_col_axis = row_axis, col_axis
        A, row_axis, col_axis = self._canonicalize_matrix_axes(
            A, requested_row_axis, requested_col_axis
        )
        B, _, _ = self._canonicalize_matrix_axes(
            B, requested_row_axis, requested_col_axis
        )

        x = self.xp.expand_dims(A, axis=-2)
        x = self.xp.expand_dims(x, axis=-1)
        y = self.xp.expand_dims(B, axis=-4)
        y = self.xp.expand_dims(y, axis=-2)
        out = (x * y).sum(axis=-4)
        out = self.xp.reshape(
            out, out.shape[:-2] + (out.shape[-2] * out.shape[-1],)
        )
        return self._restore_matrix_axes(out, row_axis, col_axis)

    @dummy_jit(static_argnums=0, static_argnames=("row_axis", "col_axis"), dynamic_batch=("A", "B"))
    def tensor_matrix_product_homogeneous(
            self,
            A: Array,
            B: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        """
        Multiply two homogeneous matrix-valued tensor-algebra coefficients.

        If ``A`` has shape ``batch + (T?, n, k, d_i)`` and ``B`` has shape
        ``batch + (T?, k, l, d_j)`` up to axis placement, then the result has
        shape ``batch + (T?, n, l, d_i d_j)`` up to axis placement.

        Parameters
        ----------
        A, B : ndarray
            Homogeneous matrix-valued coefficients.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in ``A`` and ``B``.
            Both operands are assumed to follow the same convention.

        Returns
        -------
        ndarray
            Homogeneous matrix-valued coefficient.
        """
        return self._standard_matrix_product_homogeneous(
            A, B, row_axis=row_axis, col_axis=col_axis
        )

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_right(
            self,
            A: DenseElem,
            M: Array,
            trunc: Optional[int] = None,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> DenseElem:
        """
        Right-multiply a matrix-valued tensor-algebra element by a numeric matrix.

        Each homogeneous level ``A[r]`` is interpreted using the matrix-axis
        convention specified by ``row_axis`` and ``col_axis``. Under the default
        convention, levels have shape

            ``batch + (T?, n, k, d_r)``.

        Parameters
        ----------
        A : tuple of ndarray
            Matrix-valued tensor-algebra element.
        M : ndarray
            Numeric matrix with shape ``(..., k, l)``.
        trunc : int, optional
            If given, keep only degrees ``0, ..., trunc``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in each level of ``A``.

        Returns
        -------
        tuple of ndarray
            Matrix-valued tensor-algebra element with column dimension updated
            from ``k`` to ``l``.
        """
        A = self._validate_graded_element(A, name="A")
        schedule = self._matrix_map_schedule(A, trunc)
        blocks = tuple(
            self.tensor_matrix_product_right_homogeneous(
                schedule.block(grade), M, row_axis=row_axis, col_axis=col_axis
            )
            for grade in schedule.grades
        )
        return schedule.assemble(blocks)

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_left(
            self,
            M: Array,
            A: DenseElem,
            trunc: Optional[int] = None,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> DenseElem:
        """
        Left-multiply a matrix-valued tensor-algebra element by a numeric matrix.

        Each homogeneous level ``A[r]`` is interpreted using the matrix-axis
        convention specified by ``row_axis`` and ``col_axis``. Under the default
        convention, levels have shape

            ``batch + (T?, k, l, d_r)``.

        Parameters
        ----------
        M : ndarray
            Numeric matrix with shape ``(..., n, k)``.
        A : tuple of ndarray
            Matrix-valued tensor-algebra element.
        trunc : int, optional
            If given, keep only degrees ``0, ..., trunc``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in each level of ``A``.

        Returns
        -------
        tuple of ndarray
            Matrix-valued tensor-algebra element with row dimension updated from
            ``k`` to ``n``.
        """
        A = self._validate_graded_element(A, name="A")
        schedule = self._matrix_map_schedule(A, trunc)
        blocks = tuple(
            self.tensor_matrix_product_left_homogeneous(
                M, schedule.block(grade), row_axis=row_axis, col_axis=col_axis
            )
            for grade in schedule.grades
        )
        return schedule.assemble(blocks)

    def _standard_matrix_product_block(
            self,
            left: Array,
            right: Array,
            left_grade: Any,
            right_grade: Any,
            output_grade: Any,
            *,
            row_axis: int,
            col_axis: int,
    ) -> Array:
        """Ordinary matrix-product block used below coordinate transport."""
        del left_grade, right_grade, output_grade
        return self._standard_matrix_product_homogeneous(
            left, right, row_axis=row_axis, col_axis=col_axis
        )

    def _matrix_product_block(
            self,
            left,
            right,
            left_grade,
            right_grade,
            output_grade,
            *,
            row_axis: int,
            col_axis: int,
    ):
        return self._standard_matrix_product_block(
            left,
            right,
            left_grade,
            right_grade,
            output_grade,
            row_axis=row_axis,
            col_axis=col_axis,
        )

    def _standard_tensor_matrix_product(
            self,
            A,
            B,
            trunc=None,
            *,
            row_axis: int = -3,
            col_axis: int = -2,
    ):
        """Execute an ordinary matrix tensor product without redispatch."""
        schedule = self._standard_matrix_product_schedule(
            A,
            B,
            trunc,
            row_axis=row_axis,
            col_axis=col_axis,
        )
        blocks = graded_convolution(
            schedule.grades,
            splits=schedule.splits,
            left_contains=schedule.left_contains,
            right_contains=schedule.right_contains,
            left_block=schedule.left_block,
            right_block=schedule.right_block,
            product_block=lambda left, right, lg, rg, og: (
                self._standard_matrix_product_block(
                    left,
                    right,
                    lg,
                    rg,
                    og,
                    row_axis=row_axis,
                    col_axis=col_axis,
                )
            ),
            zero_block=schedule.zero_block,
            product_grade=schedule.product_grade,
        )
        return schedule.assemble(blocks)

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "row_axis", "col_axis"), dynamic_batch=("A", "B"))
    def tensor_matrix_product(
            self,
            A: DenseElem,
            B: DenseElem,
            trunc: Optional[int] = None,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> DenseElem:
        """
        Multiply two matrix-valued tensor-algebra elements.

        If ``A`` and ``B`` are represented levelwise as
        ``batch + (T?, n, k, d_i)`` and ``batch + (T?, k, l, d_j)``,
        respectively, then this computes the graded Cauchy product

            ``(A * B)[r] = sum_{i+j=r} A[i] @ B[j]``,

        where the homogeneous products contract the inner matrix index and use
        the tensor product on the final tensor-coordinate axis.

        Parameters
        ----------
        A, B : tuple of ndarray
            Matrix-valued tensor-algebra elements.
        trunc : int, optional
            If given, truncate the product to degrees ``0, ..., trunc``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in each homogeneous
            level of ``A`` and ``B``. Both operands are assumed to use the
            same convention.

        Returns
        -------
        tuple of ndarray
            Matrix-valued tensor-algebra product.
        """
        A = self._validate_graded_element(A, name="A")
        B = self._validate_graded_element(B, name="B")
        return self._standard_tensor_matrix_product(
            A,
            B,
            trunc,
            row_axis=row_axis,
            col_axis=col_axis,
        )
