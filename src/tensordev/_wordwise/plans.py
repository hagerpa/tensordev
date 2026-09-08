"""Packed host plans for quotient-native wordwise recurrences."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb

import numpy as np

from tensordev.core.utils.precompute import _readonly


@dataclass(frozen=True, slots=True, eq=False)
class PrefixGraphPlan:
    """CSR-style predecessor graphs for every rank of one bidegree block.

    All nodes and edges are stored in one set of arrays.  This avoids a Python
    object and several NumPy allocations per multiset rank.  Edge indices are
    local to their graph, so :meth:`graph` can expose zero-copy array slices.

    A non-negative letter code is an absolute double-prime alphabet index.  A
    negative code ``-(p + 1)`` selects position ``p`` of the output's dense
    prime word at execution time.
    """

    grade: tuple[int, int]
    graph_node_offsets: np.ndarray
    graph_edge_offsets: np.ndarray
    terminal_indices: np.ndarray
    node_edge_offsets: np.ndarray
    node_prime_degrees: np.ndarray
    node_doubleprime_degrees: np.ndarray
    node_ranks: np.ndarray
    predecessor_indices: np.ndarray
    destination_indices: np.ndarray
    letter_codes: np.ndarray
    coefficients: np.ndarray

    @property
    def graph_count(self) -> int:
        return int(self.terminal_indices.size)

    @property
    def node_count(self) -> int:
        return int(self.node_prime_degrees.size)

    @property
    def edge_count(self) -> int:
        return int(self.predecessor_indices.size)

    @property
    def node_counts(self) -> np.ndarray:
        return np.diff(self.graph_node_offsets)

    @property
    def edge_counts(self) -> np.ndarray:
        return np.diff(self.graph_edge_offsets)

    def graph(self, rank: int) -> "PrefixGraphView":
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
            raise TypeError(f"rank must be an integer, got {rank!r}.")
        rank = int(rank)
        if not 0 <= rank < self.graph_count:
            raise IndexError(
                f"rank {rank} is outside [0, {self.graph_count})."
            )
        return PrefixGraphView(self, rank)

    def memory_bytes(self) -> int:
        return sum(
            int(array.nbytes)
            for array in (
                self.graph_node_offsets,
                self.graph_edge_offsets,
                self.terminal_indices,
                self.node_edge_offsets,
                self.node_prime_degrees,
                self.node_doubleprime_degrees,
                self.node_ranks,
                self.predecessor_indices,
                self.destination_indices,
                self.letter_codes,
                self.coefficients,
            )
        )


@dataclass(frozen=True, slots=True, eq=False)
class PrefixGraphView:
    """Lazy, zero-copy view of one rank in a packed prefix graph plan."""

    plan: PrefixGraphPlan
    terminal_rank: int

    @property
    def _node_slice(self) -> slice:
        offsets = self.plan.graph_node_offsets
        return slice(
            int(offsets[self.terminal_rank]),
            int(offsets[self.terminal_rank + 1]),
        )

    @property
    def _edge_slice(self) -> slice:
        offsets = self.plan.graph_edge_offsets
        return slice(
            int(offsets[self.terminal_rank]),
            int(offsets[self.terminal_rank + 1]),
        )

    @property
    def node_prime_degrees(self) -> np.ndarray:
        return self.plan.node_prime_degrees[self._node_slice]

    @property
    def node_doubleprime_degrees(self) -> np.ndarray:
        return self.plan.node_doubleprime_degrees[self._node_slice]

    @property
    def node_ranks(self) -> np.ndarray:
        return self.plan.node_ranks[self._node_slice]

    @property
    def predecessor_indices(self) -> np.ndarray:
        return self.plan.predecessor_indices[self._edge_slice]

    @property
    def destination_indices(self) -> np.ndarray:
        return self.plan.destination_indices[self._edge_slice]

    @property
    def letter_codes(self) -> np.ndarray:
        return self.plan.letter_codes[self._edge_slice]

    @property
    def coefficients(self) -> np.ndarray:
        return self.plan.coefficients[self._edge_slice]

    @property
    def node_edge_offsets(self) -> np.ndarray:
        # Every graph contributes node_count + 1 entries.  Since preceding
        # graphs contribute one extra endpoint each, the start is the packed
        # node offset plus the graph rank.
        start = self._node_slice.start + self.terminal_rank
        return self.plan.node_edge_offsets[start : start + self.node_count + 1]

    @property
    def maximum_in_degree(self) -> int:
        offsets = self.node_edge_offsets
        return int(np.max(np.diff(offsets), initial=0))

    @property
    def scalar_index(self) -> int:
        return 0

    @property
    def terminal_index(self) -> int:
        return int(self.plan.terminal_indices[self.terminal_rank])

    @property
    def node_count(self) -> int:
        return self._node_slice.stop - self._node_slice.start

    @property
    def edge_count(self) -> int:
        return self._edge_slice.stop - self._edge_slice.start

    @property
    def total_degree(self) -> int:
        return int(
            self.node_prime_degrees[self.terminal_index]
            + self.node_doubleprime_degrees[self.terminal_index]
        )


def _rank_normal_form(blocks: tuple[tuple[int, ...], ...]) -> int:
    """Rank a validated normal form without allocating a flattened tuple."""
    prefix = 0
    rank = 0
    position = 0
    last_position = len(blocks) * len(blocks[0]) - 1
    for block in blocks:
        for multiplicity in block:
            if position == last_position:
                return rank
            position += 1
            prefix += multiplicity
            rank += comb(position - 1 + prefix, position)
    return rank


def _graph_counts(placements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized exact node and edge counts for every terminal rank."""
    values = np.asarray(placements, dtype=np.int64)
    choices = values + 1
    block_nodes = np.prod(choices, axis=2, dtype=np.int64)
    node_counts = np.sum(block_nodes, axis=1, dtype=np.int64)

    # For one species, alpha_i choices have a positive value and every other
    # species has alpha_j + 1 choices.  This counts predecessor edges, rather
    # than weighting them by the removed multiplicity.
    other_choices = block_nodes[..., None] // choices
    doubleprime_edges = np.sum(
        values * other_choices,
        axis=(1, 2),
        dtype=np.int64,
    )
    prime_degree = placements.shape[1] - 1
    edge_counts = doubleprime_edges + prime_degree
    return node_counts, edge_counts


def build_prefix_graph_plan(
    placements: np.ndarray,
    *,
    grade: tuple[int, int],
    d_prime: int,
) -> PrefixGraphPlan:
    """Pack all multiset-prefix graphs from an authoritative placement table.

    Resource validation belongs to the caller and must occur before this
    routine.  Counts are computed in one vectorized pass, then every output
    array is allocated exactly once.
    """

    placements = np.asarray(placements)
    n, _m = grade
    if placements.ndim != 3 or placements.shape[1] != n + 1:
        raise ValueError(
            "placements must have shape (rank_count, prime_degree + 1, "
            "d_doubleprime)."
        )
    if placements.shape[2] <= 0:
        raise ValueError("the double-prime alphabet must be nonempty.")
    if np.any(placements < 0):
        raise ValueError("placement multiplicities must be non-negative.")
    if d_prime <= 0:
        raise ValueError(f"d_prime must be positive, got {d_prime}.")

    node_counts, edge_counts = _graph_counts(placements)
    graph_count = placements.shape[0]
    node_offsets64 = np.empty(graph_count + 1, dtype=np.int64)
    edge_offsets64 = np.empty(graph_count + 1, dtype=np.int64)
    node_offsets64[0] = 0
    edge_offsets64[0] = 0
    np.cumsum(node_counts, out=node_offsets64[1:])
    np.cumsum(edge_counts, out=edge_offsets64[1:])
    int32_max = np.iinfo(np.int32).max
    if node_offsets64[-1] > int32_max or edge_offsets64[-1] > int32_max:
        raise OverflowError("packed prefix graph exceeds signed int32 range.")

    node_offsets = node_offsets64.astype(np.int32)
    edge_offsets = edge_offsets64.astype(np.int32)
    terminal_indices = np.empty(graph_count, dtype=np.int32)
    node_edge_offsets = np.empty(node_offsets[-1] + graph_count, dtype=np.int32)
    node_prime_degrees = np.empty(node_offsets[-1], dtype=np.int32)
    node_doubleprime_degrees = np.empty(node_offsets[-1], dtype=np.int32)
    node_ranks = np.empty(node_offsets[-1], dtype=np.int32)
    predecessor_indices = np.empty(edge_offsets[-1], dtype=np.int32)
    destination_indices = np.empty(edge_offsets[-1], dtype=np.int32)
    letter_codes = np.empty(edge_offsets[-1], dtype=np.int32)
    coefficients = np.ones(edge_offsets[-1], dtype=np.int32)

    d_doubleprime = placements.shape[2]
    for terminal_rank, placement_array in enumerate(placements):
        placement = tuple(
            tuple(map(int, block))
            for block in placement_array
        )
        node_keys = []
        for prime_degree, upper in enumerate(placement):
            completed = sum(sum(block) for block in placement[:prime_degree])
            for beta in product(*(range(value + 1) for value in upper)):
                node_keys.append(
                    (
                        prime_degree + completed + sum(beta),
                        prime_degree,
                        beta,
                    )
                )
        node_keys.sort()
        key_to_index = {
            (prime_degree, beta): index
            for index, (_degree, prime_degree, beta) in enumerate(node_keys)
        }
        if len(node_keys) != int(node_counts[terminal_rank]):
            raise AssertionError("prefix node count disagrees with its closed form.")

        node_start = int(node_offsets[terminal_rank])
        for local_index, (_degree, prime_degree, beta) in enumerate(node_keys):
            normal_form = placement[:prime_degree] + (beta,)
            packed_index = node_start + local_index
            node_prime_degrees[packed_index] = prime_degree
            node_doubleprime_degrees[packed_index] = sum(
                sum(block) for block in normal_form
            )
            node_ranks[packed_index] = _rank_normal_form(normal_form)

        edge_index = int(edge_offsets[terminal_rank])
        edge_start = edge_index
        node_edge_start = node_start + terminal_rank
        node_edge_offsets[node_edge_start] = 0
        for destination, (_degree, prime_degree, beta) in enumerate(node_keys):
            for species, multiplicity in enumerate(beta):
                if multiplicity == 0:
                    continue
                predecessor_beta = list(beta)
                predecessor_beta[species] -= 1
                predecessor_indices[edge_index] = key_to_index[
                    (prime_degree, tuple(predecessor_beta))
                ]
                destination_indices[edge_index] = destination
                letter_codes[edge_index] = d_prime + species
                edge_index += 1

            if prime_degree > 0 and not any(beta):
                predecessor_indices[edge_index] = key_to_index[
                    (prime_degree - 1, placement[prime_degree - 1])
                ]
                destination_indices[edge_index] = destination
                letter_codes[edge_index] = -prime_degree
                edge_index += 1
            node_edge_offsets[node_edge_start + destination + 1] = (
                edge_index - edge_start
            )
        if edge_index != int(edge_offsets[terminal_rank + 1]):
            raise AssertionError("prefix edge count disagrees with its closed form.")

        terminal_indices[terminal_rank] = key_to_index[(n, placement[n])]
        if key_to_index[(0, (0,) * d_doubleprime)] != 0:
            raise AssertionError("scalar prefix must be the first local node.")

    return PrefixGraphPlan(
        grade=grade,
        graph_node_offsets=_readonly(node_offsets),
        graph_edge_offsets=_readonly(edge_offsets),
        terminal_indices=_readonly(terminal_indices),
        node_edge_offsets=_readonly(node_edge_offsets),
        node_prime_degrees=_readonly(node_prime_degrees),
        node_doubleprime_degrees=_readonly(node_doubleprime_degrees),
        node_ranks=_readonly(node_ranks),
        predecessor_indices=_readonly(predecessor_indices),
        destination_indices=_readonly(destination_indices),
        letter_codes=_readonly(letter_codes),
        coefficients=_readonly(coefficients),
    )


__all__ = [
    "PrefixGraphPlan",
    "PrefixGraphView",
    "build_prefix_graph_plan",
]
