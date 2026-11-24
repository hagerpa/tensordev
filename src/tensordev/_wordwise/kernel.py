r"""Pallas kernel of the wordwise signature and scalar-FSSK recursions.

A prefix-graph node :math:`v` carries a state :math:`P_v \in \mathbb{R}^R`, and an increment
:math:`\Delta` acts by :math:`(A_\Delta X)_v = \sum_{(u, \ell) \to v} X_u \Delta^\ell`.  One time
step is

.. math::
    P \leftarrow P E + \sum_{n=1}^{N} X_n A_\Delta^n, \qquad
    X_n = P \varphi_{n-1}, \quad X_n = \psi_{n-1} \text{ at the root.}

The signature is :math:`R = 1`, :math:`E = 1`, :math:`\psi_{n-1} = \varphi_{n-1} = 1/n!`; an FSSK
state loads :math:`E`, :math:`\psi`, :math:`\varphi`.  Every edge adds one letter, so the nodes
are graded by their level, and with :math:`H_{N+1} = 0`

.. math::
    H_s = A_\Delta (X_s + H_{s+1}), \qquad \sum_{n=1}^{N} X_n A_\Delta^n = H_1,

where :math:`H_s` at a node of level :math:`k` enters a state only if :math:`s + k - 1 \le N`.

One launch computes all levels.  Each lane owns a set of nodes that contains the prefixes of its
nodes, laid out in slots per level: its states are tiles indexed by level and slot, and
:math:`A_\Delta` from one level to the next is a matrix of increments :math:`\Delta^\ell`, one
entry per pair of slots joined by an edge.  A node owned by several lanes is computed by each of
them and stored by one.  The plans differ in the tables of a launch only:

* total degree: a lane is a word of length :math:`N - L` with the chain of its prefixes and the
  tree of the words that extend it by at most :math:`L` letters;
* ordered bidegree: a lane is a word of length :math:`N - 1` with the chain of its prefixes and
  its extensions by one prime letter, or by one double-prime letter;
* partially symmetrized bidegree: a lane is a coordinate of the top bidegree with the lattice of
  its prefixes, the nodes :math:`(j, \beta)` of :class:`~tensordev._wordwise.plans.PrefixGraphPlan`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, reduce
from itertools import accumulate
from numbers import Integral
from operator import add
from threading import RLock
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import div, rem

from tensordev._wordwise.dispatch import _colocate_array, _concrete_single_device
from tensordev._wordwise.layout import WordwiseLayoutPlan
from tensordev.core.utils.precompute import _readonly

Array = jax.Array

_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_DEGREE = 32
_MAX_TILE = 1024
_MAX_LETTER_GROUP = 32
# State values of the tree of a total-degree lane, and the lanes that a deeper tree must leave.
_TREE_VALUES = 128
_TREE_LANES = 1 << 11
# Tile operations of one time step, and entries of the tables of one launch.
_MAX_STEP_OPERATIONS = 1 << 16
_MAX_TABLE_ELEMENTS = 1 << 24
_LAUNCH_CACHE_SIZE = 32


class WordwiseError(RuntimeError):
    """Base error raised before a wordwise kernel is lowered."""


class WordwiseUnavailableError(WordwiseError):
    """The installed JAX build does not provide the Pallas Triton API."""


class WordwiseUnsupportedError(WordwiseError):
    """An input or dtype outside the kernel contract."""


class WordwiseResourceError(WordwiseError):
    """Static kernel resources exceed a deterministic guard."""


class WordwisePlanError(WordwiseError):
    """A layout plan that the kernel cannot execute."""


def _load_pallas():
    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as pltriton
    except (ImportError, AttributeError) as exc:
        raise WordwiseUnavailableError(
            "the installed JAX build does not provide the required Pallas/Triton API"
        ) from exc
    return pl, pltriton


def _require_integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return int(value)


def _normalize_block_size(block_size: int | None, *, steps: int) -> tuple[int, int]:
    """Steps per emitted block and the number of blocks."""
    if block_size is None or block_size == -1:
        return steps, 1
    block_size = _require_integer(block_size, name="block_size", minimum=1)
    if steps % block_size:
        raise ValueError(f"block_size={block_size} must divide the step count {steps}.")
    return block_size, steps // block_size


def _tile(lanes: int, requested: int) -> int:
    """The requested power-of-two tile, shrunk to the smallest one covering ``lanes``."""
    requested = _require_integer(requested, name="tile", minimum=1)
    if requested > _MAX_TILE or requested & (requested - 1):
        raise WordwiseResourceError(
            f"the tile must be a power of two no greater than {_MAX_TILE}, got {requested}."
        )
    return min(requested, 1 << (lanes - 1).bit_length())


# ---------------------------------------------------------------------------
# Launches
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Launch:
    r"""The lanes of one kernel launch and the nodes they own.

    A lane is a row :math:`r` of the tables with a prime word :math:`c'` and a double-prime word
    :math:`c''`; with ``inner = (w', w'')`` the numbers of these words it is lane
    :math:`(r w' + c') w'' + c''`.  Level :math:`k` has ``levels[k - 1]`` slots, numbered through
    the levels.  The record ``nodes[r, slot] = (b, s', m', s'', m'', w)`` places the node of a slot
    at the coordinate

    .. math::
        b + \lfloor c' / s' \rfloor m' + \lfloor c'' / s'' \rfloor m'',

    :math:`b = -1` marking a slot without node, and the lane stores it if :math:`w = 1`,
    :math:`s' \mid c'` and :math:`s'' \mid c''`.  A pair ``(level, slot, predecessor, letter,
    partial)`` is an edge into a slot from the slot ``predecessor`` of the level below.  Its letter
    is ``letter``, or for ``letter = None`` the code in the next column of ``edges``: a letter,
    :math:`-1` for the last letter of :math:`\lfloor c' / s' \rfloor`, :math:`-2` for :math:`d'`
    plus the last letter of :math:`\lfloor c'' / s'' \rfloor`, or :math:`0` for no edge, which
    ``partial`` pairs alone contain.

    Level :math:`N` has ``top`` slots, of which program :math:`g` owns ``levels[-1]`` from slot
    :math:`g` ``levels[-1]`` on; its letters are those of the first program plus :math:`g`
    ``levels[-1]``.
    """

    levels: tuple[int, ...]
    pairs: tuple[tuple[int, int, int, int | None, bool], ...]
    rows: int
    inner: tuple[int, int]
    top: int
    nodes: np.ndarray
    edges: np.ndarray

    @property
    def lanes(self) -> int:
        return self.rows * self.inner[0] * self.inner[1]

    def operations(self, state_dim: int) -> int:
        """Tile operations of one time step: increments and matrix products of all :math:`H_s`."""
        N, R = len(self.levels), max(state_dim, 1)
        return R * sum(N - level + 1 for level, *_ in self.pairs) + R * R * sum(
            (N - level + 1) * count for level, count in enumerate(self.levels, start=1)
        )


def _table(values: Any) -> np.ndarray:
    """A read-only int32 table; one without columns gets one."""
    values = np.asarray(values, dtype=np.int32)
    return _readonly(values if values.shape[-1] else np.zeros(values.shape[:-1] + (1,), np.int32))


def _lookup(codes: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Positions of ``wanted`` among the distinct ``codes``."""
    order = np.argsort(codes)
    return order[np.searchsorted(codes[order], wanted)]


def _tree(plan: WordwiseLayoutPlan, tree: int) -> Launch:
    """Total degree: the words of length ``N - tree`` with their prefixes and extensions."""
    d, N = plan.alphabet_dim, len(plan.blocks) - 1
    depth = N - tree
    offsets = [block.flat_offset - 1 for block in plan.blocks[1:]]
    levels = (1,) * depth + tuple(d ** j for j in range(1, tree)) + (
        d ** tree if tree > 1 else min(d, _MAX_LETTER_GROUP),
    )
    nodes = [(offsets[k - 1], d ** (depth - k), 1, 1, 1, 1) for k in range(1, depth + 1)] + [
        (offsets[depth + j - 1] + i, 1, d ** j, 1, 1, 1) for j in range(1, tree + 1) for i in range(d ** j)
    ]
    pairs = tuple((k, 0, 0, None, False) for k in range(1, depth + 1)) + tuple(
        (depth + j, i, i // d, i % d, False) for j in range(1, tree + 1) for i in range(levels[depth + j - 1])
    )
    return Launch(
        levels, pairs, 1, (d ** depth, 1), d ** tree, _table([nodes]), _table(np.full((1, depth), -1))
    )


def _patterns(plan: WordwiseLayoutPlan) -> tuple[Launch, ...]:
    """Ordered bidegree: per type of the last letter, the words of length ``N - 1`` that it extends.

    A row is a set of positions of the prime letters: a block is ordered by these sets, then by
    the prime word, then by the double-prime word.
    """
    d_prime, d_doubleprime = plan.dims
    d, N = plan.alphabet_dim, sum(plan.truncation)

    def rows(grade):
        """Block of a bidegree, its inner widths and the codes of its words with zero inner words."""
        block, inner = plan.block(grade), (d_prime ** grade[0], d_doubleprime ** grade[1])
        return block, inner, np.asarray(block.word_codes, dtype=np.int64)[:: inner[0] * inner[1]]

    def bases(grade, codes):
        """Coordinates of the words ``codes`` of a bidegree, which have zero inner words."""
        block, inner, block_codes = rows(grade)
        return block.flat_offset - 1 + _lookup(block_codes, codes) * inner[0] * inner[1]

    launches, stored = [], np.empty(0, dtype=np.int64)
    for grade, first, count in (
            ((plan.truncation[0] - 1, plan.truncation[1]), 0, d_prime),
            ((plan.truncation[0], plan.truncation[1] - 1), d_prime, d_doubleprime),
    ):
        if min(grade) < 0:
            continue
        _, inner, codes = rows(grade)
        prime = (codes[:, None] // d ** np.arange(N - 2, -1, -1)) % d < d_prime
        primes = np.cumsum(prime, axis=1)
        nodes = np.ones((codes.size, N - 1 + count, 6), dtype=np.int64)
        for k in range(1, N):
            for a in np.unique(primes[:, k - 1]):
                members = primes[:, k - 1] == a
                nodes[members, k - 1, 0] = bases((a, k - a), codes[members] // d ** (N - 1 - k))
                nodes[members, k - 1, 1:4] = (
                    d_prime ** (grade[0] - a), d_doubleprime ** (k - a), d_doubleprime ** (grade[1] - k + a)
                )
        chain = nodes[:, : N - 1, 0]
        values, firsts = np.unique(chain, return_index=True)
        new = ~np.isin(values, stored)
        stored = np.concatenate([stored, values[new]])
        nodes[:, : N - 1, 5] = np.isin(np.arange(chain.size), firsts[new]).reshape(chain.shape)
        nodes[:, N - 1:, 0] = bases(plan.truncation, codes * d + first)[:, None] + np.arange(count) * (
            1 if first else inner[1]
        )
        nodes[:, N - 1:, 2] = inner[1] * (d_doubleprime if first else d_prime)
        nodes[:, N - 1:, 4] = d_doubleprime if first else 1
        group = min(count, _MAX_LETTER_GROUP)
        launches.append(Launch(
            (1,) * (N - 1) + (group,),
            tuple((k, 0, 0, None, False) for k in range(1, N))
            + tuple((N, i, 0, first + i, False) for i in range(group)),
            codes.size, inner, count, _table(nodes),
            _table(np.where(prime, -1, -2 if d_doubleprime > 1 else d_prime)),
        ))
    return tuple(launches)


def _lattices(plan: WordwiseLayoutPlan) -> tuple[Launch, ...]:
    """Partially symmetrized bidegree: the coordinates of the top bidegree with their lattices.

    A row is a rank of the top bidegree, whose graph has its nodes sorted by level; the slot of a
    node is its position among those of its level.
    """
    graph = plan.blocks[-1].prefix_plan
    n, m = plan.truncation
    N, d_prime, rows = n + m, plan.dims[0], graph.graph_count
    prime_degree = graph.node_prime_degrees.astype(np.int64)
    level = prime_degree + graph.node_doubleprime_degrees
    owner = np.repeat(np.arange(rows), graph.node_counts)
    runs = np.flatnonzero(np.diff(owner * (N + 1) + level, prepend=-1))
    slot = np.arange(level.size) - np.repeat(runs, np.diff(np.append(runs, level.size)))
    levels = tuple(int(slot[level == k].max()) + 1 for k in range(1, N + 1))
    if rows * sum(levels) * 6 > _MAX_TABLE_ELEMENTS:
        raise WordwiseResourceError(
            f"the node table has {rows * sum(levels) * 6} entries, exceeding {_MAX_TABLE_ELEMENTS}."
        )
    columns = np.cumsum((0, 0) + levels[:-1])

    offsets = np.zeros((n + 1, m + 1), dtype=np.int64)
    for block in plan.blocks:
        offsets[block.grade] = block.flat_offset - 1
    base = offsets[prime_degree, graph.node_doubleprime_degrees] + graph.node_ranks * d_prime ** prime_degree
    positive = np.flatnonzero(level)
    writes = np.zeros(level.size, dtype=np.int64)
    writes[positive[np.unique(base[positive], return_index=True)[1]]] = 1
    one = np.ones(level.size, dtype=np.int64)
    nodes = np.tile(np.asarray([-1, 1, 1, 1, 1, 0]), (rows, sum(levels), 1))
    nodes[owner[positive], (columns[level] + slot)[positive]] = np.stack(
        [base, d_prime ** (n - prime_degree), one, one, one, writes], axis=1
    )[positive]

    edge_owner = np.repeat(np.arange(rows), graph.edge_counts)
    start = graph.graph_node_offsets[edge_owner].astype(np.int64)
    destination, predecessor = start + graph.destination_indices, start + graph.predecessor_indices
    triples, column = np.unique(
        np.stack([level[destination], slot[destination], slot[predecessor]], axis=1),
        axis=0, return_inverse=True,
    )
    edges = np.zeros((rows, len(triples)), dtype=np.int64)
    edges[edge_owner, column.reshape(-1)] = np.maximum(graph.letter_codes, -1)
    pairs = tuple(
        (*map(int, triple), None, bool(np.any(edges[:, index] == 0))) for index, triple in enumerate(triples)
    )
    return (Launch(levels, pairs, rows, (d_prime ** n, 1), 1, _table(nodes), _table(edges)),)


_LAUNCHES: OrderedDict[tuple[object, ...], tuple[Launch, ...]] = OrderedDict()
_LAUNCHES_LOCK = RLock()


def launches(plan: WordwiseLayoutPlan, tree: int = 1) -> tuple[Launch, ...]:
    """The launches of a plan; ``tree`` is the number of tree levels of a total-degree launch."""
    if len(plan.blocks) < 2 or plan.blocks[0].total_degree != 0 or plan.blocks[0].width != 1:
        raise WordwisePlanError(
            "the layout plan must begin with its scalar block and have a positive truncation."
        )
    if plan.blocks[-1].total_degree > _MAX_DEGREE:
        raise WordwiseResourceError(
            f"degree {plan.blocks[-1].total_degree} exceeds the {_MAX_DEGREE}-degree kernel guard."
        )
    key = plan.fingerprint + (tree,)
    with _LAUNCHES_LOCK:
        if key in _LAUNCHES:
            _LAUNCHES.move_to_end(key)
            return _LAUNCHES[key]
    result = (
        (_tree(plan, tree),) if plan.grading == "total_degree"
        else _lattices(plan) if plan.partially_symmetrized else _patterns(plan)
    )
    with _LAUNCHES_LOCK:
        _LAUNCHES[key] = result
        while len(_LAUNCHES) > _LAUNCH_CACHE_SIZE:
            _LAUNCHES.popitem(last=False)
    return result


def _tree_levels(d: int, N: int, state_dim: int, batch: int) -> int:
    """Tree levels of a total-degree launch: as many as leave the tree and the lanes within their bounds."""
    tree = 1
    while (
            tree < N
            and sum(d ** j for j in range(1, tree + 2)) * max(state_dim, 1) <= _TREE_VALUES
            and batch * d ** (N - tree - 1) >= _TREE_LANES
    ):
        tree += 1
    return tree


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _call(
        shape: tuple[int, int, int],
        dtype: np.dtype,
        state_dim: int,
        coefficient_axes: tuple[tuple[int, int], ...],
        graph: tuple[Any, ...],
        alphabet: tuple[int, int],
        width: int,
        seed_batch: int,
        readout_batch: int,
        emission: tuple[int, bool] | None,
        tile: int,
        aliased: bool,
        interpret: bool,
):
    """``state_dim == 0`` is the signature; ``graph`` holds the fields of a launch but its tables."""
    pl, pltriton = _load_pallas()
    batch, steps, _ = shape
    levels, pairs, rows, (prime_width, doubleprime_width), top = graph
    d_prime, d_doubleprime = alphabet
    N, R = len(levels), state_dim
    lanes = rows * prime_width * doubleprime_width
    columns = tuple(accumulate((0,) + levels[:-1]))
    edge_columns = tuple(accumulate((0,) + tuple(letter is None for *_, letter, _ in pairs)))
    incoming = [[[] for _ in range(count)] for count in levels]
    for pair, (level, slot, predecessor, _, _) in enumerate(pairs):
        incoming[level - 1][slot].append((predecessor, pair))
    constants = tuple(1.0 / math.factorial(k) for k in range(1, N + 1))
    blocks = 1 if emission is None else steps // emission[0]
    components = bool(R) and not readout_batch

    def kernel(*refs):
        refs = list(refs)
        y_ref, output_ref = refs.pop(0), refs.pop()
        E_ref, psi_ref, phi_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if R else (None,) * 3
        nodes_ref, edges_ref = refs.pop(0), refs.pop(0)
        initial_ref = refs.pop(0) if seed_batch else None
        weights_ref = refs.pop(0) if readout_batch else None

        b = pl.program_id(0)
        q = pl.program_id(1) * tile + jnp.arange(tile, dtype=jnp.int32)
        valid = q < lanes
        row = jnp.minimum(q // (prime_width * doubleprime_width), rows - 1) if rows > 1 else 0
        prime_word, doubleprime_word = (q // doubleprime_width) % prime_width, q % doubleprime_width
        shift = pl.program_id(2) * levels[-1]

        def record(level, slot):
            """Whether a slot is one of the launch, and the loader of its table fields."""
            present = True
            if level == N:
                present, slot = shift + slot < top, jnp.minimum(shift + slot, top - 1)
            return present, lambda field: pltriton.load(nodes_ref.at[row, columns[level - 1] + slot, field])

        def node(level, slot):
            """Existence, coordinate and store mask of a slot."""
            present, field = record(level, slot)
            base, stride = field(0), field(1)
            exists = valid & present & (base >= 0)
            index = base + div(prime_word, stride) * field(2)
            stores = exists & (field(5) > 0) & (rem(prime_word, stride) == 0)
            if doubleprime_width > 1:
                index = index + div(doubleprime_word, field(3)) * field(4)
                stores = stores & (rem(doubleprime_word, field(3)) == 0)
            return exists, index, stores if level == N else stores & (shift == 0)

        def letter(pair):
            """Letter of an edge and the mask of its load."""
            level, slot, _, static, partial = pairs[pair]
            if static is not None:
                return static + (jnp.minimum(shift, top - 1 - slot) if level == N else 0), {}
            code = pltriton.load(edges_ref.at[row, edge_columns[pair]])
            _, field = record(level, slot)
            inner = div(prime_word, field(1)) % d_prime
            if doubleprime_width > 1:
                doubleprime_letter = d_prime + div(doubleprime_word, field(3)) % d_doubleprime
                inner = jnp.where(code == -2, doubleprime_letter, inner)
            mask = {"mask": valid & (code != 0), "other": 0} if partial else {}
            return jnp.where(code < 0, inner, code), mask

        letters = tuple(map(letter, range(len(pairs))))

        def coefficient(ref, axes, t, *index):
            return pltriton.load(ref.at[(0 if axes[0] == 1 else b, 0 if axes[1] == 1 else t) + index])

        def coefficients(t):
            if not R:
                return None, constants, constants
            axis = range(R)
            E_axes, psi_axes, phi_axes = coefficient_axes
            return (
                tuple(tuple(coefficient(E_ref, E_axes, t, r, c) for c in axis) for r in axis),
                tuple(tuple(coefficient(psi_ref, psi_axes, t, k, r) for r in axis) for k in range(N)),
                tuple(
                    tuple(tuple(coefficient(phi_ref, phi_axes, t, n, r, c) for c in axis) for r in axis)
                    for n in range(N - 1)
                ),
            )

        def times(state, matrix):
            """Row vector ``state`` times ``matrix``, which is a scalar for the signature."""
            if not R:
                return tuple(x * matrix for x in state)
            return tuple(reduce(add, (state[r] * matrix[r][c] for r in range(R))) for c in range(R))

        def step(t, P):
            E, psi, phi = coefficients(t)
            D = tuple(pltriton.load(y_ref.at[b, t, index], **mask) for index, mask in letters)
            sources = {}

            def source(s, level, slot):
                """X_s + H_{s+1} at a slot; level 0 is the root."""
                if (s, level, slot) not in sources:
                    sources[s, level, slot] = (
                        (psi[s - 1] if R else (jnp.asarray(psi[s - 1], dtype=dtype),)) if level == 0
                        else tuple(map(
                            add, times(P[level - 1][slot], phi[s - 1]), action(s + 1, level, slot)
                        ))
                    )
                return sources[s, level, slot]

            def action(s, level, slot):
                """H_s at a slot."""
                return tuple(
                    reduce(add, terms) for terms in zip(*(
                        tuple(x * D[pair] for x in source(s, level - 1, predecessor))
                        for predecessor, pair in incoming[level - 1][slot]
                    ))
                )

            return tuple(
                tuple(
                    tuple(map(add, times(state, E) if R else state, action(1, level, slot)))
                    for slot, state in enumerate(P[level - 1])
                )
                for level in range(1, N + 1)
            )

        def seed(level, slot):
            exists, index, _ = node(level, slot)
            return tuple(
                pltriton.load(initial_ref.at[0 if seed_batch == 1 else b, index, r], mask=exists, other=0)
                for r in range(R)
            )

        zero = tuple(jnp.zeros((tile,), dtype=dtype) for _ in range(max(R, 1)))
        initial = tuple(
            tuple(seed(level, slot) if seed_batch else zero for slot in range(count))
            for level, count in enumerate(levels, start=1)
        )
        if readout_batch:
            weights = tuple(
                pltriton.load(weights_ref.at[0 if readout_batch == 1 else b, r]) for r in range(R)
            )

        def emit(block, P):
            lead = (b,) if emission is None else (b, block)
            for level, count in enumerate(levels, start=1):
                for slot in range(count):
                    _, index, stores = node(level, slot)
                    # Masked elements point at the spare coordinate: the Pallas interpreter writes them back.
                    index = jnp.where(stores, index, width)
                    state = P[level - 1][slot]
                    if readout_batch:
                        state = (reduce(add, (x * w for x, w in zip(state, weights))),)
                    for r, value in enumerate(state):
                        coordinate = lead + ((r,) if components else ()) + (index,)
                        pltriton.store(output_ref.at[coordinate], value, mask=stores)

        if emission is None:
            emit(None, jax.lax.fori_loop(0, steps, step, initial))
            return
        block_size, accumulate_blocks = emission

        def block_step(block, P):
            start = block * block_size
            P = jax.lax.fori_loop(
                0, block_size, lambda i, state: step(start + i, state), P if accumulate_blocks else initial
            )
            emit(block, P)
            return P

        jax.lax.fori_loop(0, blocks, block_step, initial)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch,) + (() if emission is None else (blocks,)) + ((R,) if components else ()) + (width + 1,),
            dtype,
        ),
        grid=(batch, -(-lanes // tile), -(-top // levels[-1])),
        input_output_aliases=(
            {3 + (3 if R else 0) + bool(seed_batch) + bool(readout_batch): 0} if aliased else {}
        ),
        compiler_params=pltriton.CompilerParams(num_warps=min(max(tile // 32, 1), 8), num_stages=1),
        interpret=interpret,
        name="tensordev_wordwise",
    )


def wordwise_blocks(
        increments: Any,
        *,
        plan: WordwiseLayoutPlan,
        coefficients: tuple[Any, Any, Any] | None = None,
        initial: Any = None,
        weights: Any = None,
        block_size: int | None = None,
        accumulate: bool = True,
        tile: int = 32,
        tree: int | None = None,
        interpret: bool = False,
) -> tuple[Array, ...]:
    r"""The blocks of ``plan`` in standard coordinates: those of positive degree for a state.

    ``increments`` has shape ``(B, S, d)``.  Without ``coefficients`` the blocks are those of the
    signature, of shape ``(B, w)``, and ``(B, blocks, w)`` for several emitted blocks.
    ``coefficients = (E, psi, phi)`` with shapes ``(cb, cs, R, R)``, ``(cb, cs, >= N, R)`` and
    ``(cb, cs, >= N - 1, R, R)``, ``cb`` in ``{1, B}`` and ``cs`` in ``{1, S}``, gives the FSSK
    state ``(B, [blocks,] R, w)`` from ``initial``, of shape ``(1 or B, W, R)`` over the :math:`W`
    positive coordinates and zero by default; ``weights`` of shape ``(1 or B, R)`` replace the
    state by its readout ``(B, w)``, a signature.  With ``accumulate=False`` every block starts
    from ``initial``.  ``tree`` is the number :math:`L` of tree levels of a total-degree launch.
    """
    increments = jnp.asarray(increments)
    if increments.ndim != 3 or min(increments.shape) <= 0:
        raise WordwiseUnsupportedError(
            f"increments must have a nonempty shape (batch, steps, alphabet), got {increments.shape}."
        )
    batch, steps, d = map(int, increments.shape)
    dtype = np.dtype(increments.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise WordwiseUnsupportedError(f"increments must have dtype float32 or float64, got {dtype}.")
    if d != plan.alphabet_dim:
        raise ValueError(f"the alphabet width {d} disagrees with the plan dimension {plan.alphabet_dim}.")
    if not isinstance(accumulate, bool) or not isinstance(interpret, bool):
        raise TypeError("accumulate and interpret must be bools.")
    N, width = plan.blocks[-1].total_degree, plan.output_size - 1
    block_size, blocks = _normalize_block_size(block_size, steps=steps)
    emission = None if blocks == 1 else (block_size, accumulate)

    operands, R, axes = [increments], 0, ()
    if coefficients is not None:
        E, psi, phi = (jnp.asarray(_colocate_array(c, increments), dtype=dtype) for c in coefficients)
        R = int(E.shape[-1])
        if (E.ndim, psi.ndim, phi.ndim) != (4, 4, 5) or E.shape[-2:] != (R, R) or psi.shape[-1] != R \
                or phi.shape[-2:] != (R, R) or psi.shape[2] < N or phi.shape[2] < N - 1:
            raise ValueError(
                f"E, psi, phi must have shapes (cb, cs, R, R), (cb, cs, >= {N}, R), "
                f"(cb, cs, >= {N - 1}, R, R); got {E.shape}, {psi.shape}, {phi.shape}."
            )
        axes = tuple(tuple(map(int, c.shape[:2])) for c in (E, psi, phi))
        if any(cb not in (1, batch) or cs not in (1, steps) for cb, cs in axes):
            raise ValueError(f"coefficient axes must be (1 or {batch}, 1 or {steps}), got {axes}.")
        if phi.shape[2] == 0:  # zero-length Pallas operands are not portable across JAX versions
            phi = jnp.zeros(
                phi.shape[:2] + (1, R, R), dtype=dtype, device=_concrete_single_device(increments)
            )
        operands += [E, psi, phi]
    elif initial is not None or weights is not None:
        raise ValueError("initial and weights require coefficients.")
    tail = []
    if initial is not None:
        initial = jnp.asarray(_colocate_array(initial, increments), dtype=dtype)
        if initial.ndim != 3 or initial.shape[0] not in (1, batch) or initial.shape[1:] != (width, R):
            raise ValueError(f"initial must have shape (1 or {batch}, {width}, {R}), got {initial.shape}.")
        tail.append(initial)
    if weights is not None:
        if blocks != 1:
            raise WordwiseUnsupportedError("the readout is emitted for the terminal state only.")
        weights = jnp.asarray(_colocate_array(weights, increments), dtype=dtype)
        if weights.ndim != 2 or weights.shape[0] not in (1, batch) or weights.shape[1] != R:
            raise ValueError(f"weights must have shape (1 or {batch}, {R}), got {weights.shape}.")
        tail.append(weights)

    shape = (
        (batch,) + (() if emission is None else (blocks,)) + ((R,) if R and weights is None else ())
        + (width + 1,)
    )
    if max(math.prod(shape), steps) > _INT32_MAX:
        raise WordwiseResourceError(
            f"the output has {math.prod(shape)} elements, exceeding signed int32 indexing."
        )
    if plan.grading != "total_degree":
        tree = 1
    elif tree is None:
        tree = _tree_levels(d, N, R, batch)
    elif not 1 <= _require_integer(tree, name="tree", minimum=1) <= N:
        raise ValueError(f"tree must be at most the degree {N}, got {tree}.")
    plan_launches = launches(plan, tree)
    for launch in plan_launches:
        if launch.operations(R) > _MAX_STEP_OPERATIONS:
            raise WordwiseResourceError(
                f"one time step needs {launch.operations(R)} tile operations, "
                f"exceeding {_MAX_STEP_OPERATIONS}."
            )

    output = None
    for launch in plan_launches:
        call = _call(
            (batch, steps, d), dtype, R, axes,
            (launch.levels, launch.pairs, launch.rows, launch.inner, launch.top),
            plan.dims if plan.grading == "bidegree" else (d, 0),
            width,
            0 if initial is None else int(initial.shape[0]),
            0 if weights is None else int(weights.shape[0]),
            emission, _tile(launch.lanes, tile), len(plan_launches) > 1, interpret,
        )
        tables = [_colocate_array(table, increments) for table in (launch.nodes, launch.edges)]
        arguments = operands + tables + tail
        if len(plan_launches) > 1:
            arguments.append(
                jnp.zeros(shape, dtype=dtype, device=_concrete_single_device(increments))
                if output is None else output
            )
        output = call(*arguments)
    blocks = tuple(output[..., block.flat_offset - 1:block.flat_stop - 1] for block in plan.blocks[1:])
    return blocks if R and weights is None else (jnp.ones_like(blocks[0][..., :1]),) + blocks


def clear_kernel_caches() -> None:
    """Clear the launches and the Pallas-call factories."""
    _call.cache_clear()
    with _LAUNCHES_LOCK:
        _LAUNCHES.clear()


__all__ = [
    "Launch",
    "WordwiseError",
    "WordwisePlanError",
    "WordwiseResourceError",
    "WordwiseUnavailableError",
    "WordwiseUnsupportedError",
    "clear_kernel_caches",
    "launches",
    "wordwise_blocks",
]
