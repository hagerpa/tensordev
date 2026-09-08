"""Tests for bounded wordwise layout and prefix-graph plans."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

import tensordev as td
from tensordev._wordwise import (
    PlanLimits,
    PlanResourceError,
    build_layout_plan,
    check_plan_resources,
    clear_layout_plan_cache,
    estimate_layout_plan,
)
from tensordev.core.bigraded.precompute import colex_placements
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
)


def _ordered_word_codes(dims, grade):
    d_prime, d_doubleprime = dims
    n, m = grade
    dimension = sum(dims)
    codes = []
    for placement in colex_placements(n + m, n):
        prime_positions = set(placement)
        for prime_word in product(range(d_prime), repeat=n):
            for doubleprime_word in product(range(d_doubleprime), repeat=m):
                prime_index = 0
                doubleprime_index = 0
                code = 0
                for position in range(n + m):
                    if position in prime_positions:
                        letter = prime_word[prime_index]
                        prime_index += 1
                    else:
                        letter = d_prime + doubleprime_word[doubleprime_index]
                        doubleprime_index += 1
                    code = dimension * code + letter
                codes.append(code)
    return np.asarray(codes, dtype=np.int32)


def test_dimension_free_total_plan_has_arithmetic_decoding_and_validates_width():
    plan = build_layout_plan(td.Jax(), 4, alphabet_dim=3)

    assert plan.fingerprint == ("total_degree", (3,), 4, False)
    assert plan.grades == (0, 1, 2, 3, 4)
    assert tuple(block.width for block in plan.blocks) == (1, 3, 9, 27, 81)
    assert all(block.word_codes is None for block in plan.blocks)
    assert plan.memory_bytes() == plan.estimate.metadata_bytes == 0

    with pytest.raises(ValueError, match="alphabet_dim is required"):
        build_layout_plan(td.Jax(), 2)
    with pytest.raises(ValueError, match="disagrees"):
        build_layout_plan(td.Jax(d=2, max_trunc=3), 2, alphabet_dim=3)


def test_coordinate_choice_and_core_identity_do_not_enter_total_plan_key():
    clear_layout_plan_cache()
    standard = td.Jax(d=2, max_trunc=3)
    shear = td.make_core(
        dims=(1, 1),
        max_trunc=3,
        coordinates="shear",
    )

    standard_plan = build_layout_plan(standard, 3)
    shear_plan = build_layout_plan(shear, 3)

    assert shear_plan is standard_plan
    assert standard_plan.fingerprint == ("total_degree", (2,), 3, False)


def test_ordered_bidegree_decoder_matches_independent_word_enumeration():
    core = td.make_core(dims=(2, 2), max_trunc=(2, 2))
    plan = build_layout_plan(core, (2, 1))

    assert plan.grades == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1))
    for block_index, block in enumerate(plan.blocks):
        np.testing.assert_array_equal(
            block.word_codes,
            _ordered_word_codes(core.dims, block.grade),
        )
        assert not block.word_codes.flags.writeable
        group = plan.execution_groups[block.execution_group]
        start, stop = group.block_slices[
            group.block_indices.index(block_index)
        ]
        assert (start, stop) == (
            block.group_offset,
            block.group_offset + block.width,
        )
        assert np.shares_memory(block.word_codes, group.decoder_codes)
    assert tuple(group.total_degree for group in plan.execution_groups) == (0, 1, 2, 3)
    assert all(
        len({plan.blocks[index].total_degree for index in group.block_indices}) == 1
        for group in plan.execution_groups
    )
    assert plan.memory_bytes() == plan.estimate.metadata_bytes


def test_active_plan_contains_no_capacity_only_grades():
    core = td.make_core(dims=(1, 2), max_trunc=(4, 4))
    plan = build_layout_plan(core, (1, 2))

    assert plan.truncation == (1, 2)
    assert all(n <= 1 and m <= 2 for n, m in plan.grades)
    assert len(plan.blocks) == 6


@pytest.mark.parametrize("dims", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("truncation", ((0, 2), (1, 1), (2, 2)))
def test_partial_estimates_equal_packed_plan_payload(dims, truncation):
    clear_layout_plan_cache()
    core = td.make_core(
        dims=dims,
        max_trunc=truncation,
        partially_symmetrized=True,
    )
    plan = build_layout_plan(core, truncation)
    packed = tuple(block.prefix_plan for block in plan.blocks)

    assert all(prefix is not None for prefix in packed)
    assert plan.estimate.exact
    assert sum(prefix.graph_count for prefix in packed) == plan.estimate.prefix_graphs
    assert sum(prefix.node_count for prefix in packed) == plan.estimate.prefix_nodes
    assert sum(prefix.edge_count for prefix in packed) == plan.estimate.prefix_edges
    assert plan.memory_bytes() == plan.estimate.metadata_bytes
    assert plan.estimate.prefix_graphs == sum(
        multiset_placement_count(dims[1], grade)
        for grade in plan.grades
    )

    for prefix in packed:
        for array in (
            prefix.graph_node_offsets,
            prefix.graph_edge_offsets,
            prefix.terminal_indices,
            prefix.node_edge_offsets,
            prefix.node_prime_degrees,
            prefix.node_doubleprime_degrees,
            prefix.node_ranks,
            prefix.predecessor_indices,
            prefix.destination_indices,
            prefix.letter_codes,
            prefix.coefficients,
        ):
            assert not array.flags.writeable


def test_prefix_graphs_are_closed_and_destination_sorted():
    core = td.make_core(
        dims=(2, 2),
        max_trunc=(2, 2),
        partially_symmetrized=True,
    )
    plan = build_layout_plan(core, (2, 2))

    for block in plan.blocks:
        prefix = block.prefix_plan
        assert prefix.graph_count == core.plan_store.grade_plan(block.grade).rank_count
        for rank in range(prefix.graph_count):
            graph = prefix.graph(rank)
            degrees = graph.node_prime_degrees + graph.node_doubleprime_degrees
            np.testing.assert_array_equal(
                degrees[graph.destination_indices],
                degrees[graph.predecessor_indices] + 1,
            )
            assert graph.scalar_index == 0
            assert graph.node_ranks[graph.terminal_index] == rank
            assert graph.node_prime_degrees[graph.terminal_index] == block.grade[0]
            assert (
                graph.node_doubleprime_degrees[graph.terminal_index]
                == block.grade[1]
            )
            assert graph.node_edge_offsets[0] == 0
            assert graph.node_edge_offsets[-1] == graph.edge_count
            assert graph.maximum_in_degree <= core.dims[1] + 1
            for node in range(graph.node_count):
                edge_slice = slice(
                    graph.node_edge_offsets[node],
                    graph.node_edge_offsets[node + 1],
                )
                assert np.all(graph.destination_indices[edge_slice] == node)
                codes = graph.letter_codes[edge_slice]
                # Double-prime species are emitted in alphabet order and the
                # optional dense-prime predecessor is last.
                assert np.all(codes[:-1] <= codes[1:]) or np.any(codes < 0)
                if np.any(codes < 0):
                    assert codes[-1] < 0
                    assert np.count_nonzero(codes < 0) == 1
            assert np.all(graph.coefficients == 1)


def test_resource_guard_runs_before_prefix_construction(monkeypatch):
    import tensordev._wordwise.layout as layout_module

    clear_layout_plan_cache()
    core = td.make_core(
        dims=(1, 2),
        max_trunc=(2, 2),
        partially_symmetrized=True,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("prefix allocation must not start")

    monkeypatch.setattr(layout_module, "build_prefix_graph_plan", unexpected)
    with pytest.raises(PlanResourceError, match="output coordinates"):
        build_layout_plan(
            core,
            (2, 2),
            limits=PlanLimits(max_output_coordinates=1),
        )


def test_huge_structural_requests_terminate_with_conservative_bounds():
    estimate = estimate_layout_plan(
        grading="bidegree",
        dims=(1, 1),
        truncation=(10**9, 0),
    )

    assert not estimate.exact
    assert estimate.block_count == 10**9 + 1
    with pytest.raises(PlanResourceError):
        check_plan_resources(estimate)

    total = estimate_layout_plan(
        grading="total_degree",
        dims=2,
        truncation=10**9,
    )
    assert total.output_coordinates > np.iinfo(np.int32).max
    with pytest.raises(PlanResourceError):
        check_plan_resources(total)


def test_partial_word_code_guard_only_covers_dense_prime_decoder():
    ordered = estimate_layout_plan(
        grading="bidegree",
        dims=(1, 2),
        truncation=(20, 0),
    )
    partial = estimate_layout_plan(
        grading="bidegree",
        dims=(1, 2),
        truncation=(20, 0),
        partially_symmetrized=True,
    )

    assert ordered.maximum_word_code > np.iinfo(np.int32).max
    assert partial.maximum_word_code == 0


def test_layout_cache_enforces_aggregate_metadata_byte_cap(monkeypatch):
    import tensordev._wordwise.layout as layout_module

    first_core = td.make_core(dims=(1, 1), max_trunc=(2, 1))
    second_core = td.make_core(dims=(1, 2), max_trunc=(2, 1))
    clear_layout_plan_cache()
    first_size = build_layout_plan(first_core).memory_bytes()
    clear_layout_plan_cache()
    second_size = build_layout_plan(second_core).memory_bytes()
    byte_cap = max(first_size, second_size)
    monkeypatch.setattr(layout_module, "_PLAN_CACHE_MAX_BYTES", byte_cap)
    clear_layout_plan_cache()

    first = build_layout_plan(first_core)
    second = build_layout_plan(second_core)

    assert layout_module._PLAN_CACHE_BYTES <= byte_cap
    assert len(layout_module._PLAN_CACHE) == 1
    assert build_layout_plan(second_core) is second
    assert build_layout_plan(first_core) is not first


def test_clear_layout_plan_cache_resets_retained_byte_accounting():
    import tensordev._wordwise.layout as layout_module

    clear_layout_plan_cache()
    core = td.make_core(dims=(1, 2), max_trunc=(2, 1))
    assert build_layout_plan(core).memory_bytes() > 0
    assert layout_module._PLAN_CACHE_BYTES > 0

    clear_layout_plan_cache()

    assert not layout_module._PLAN_CACHE
    assert layout_module._PLAN_CACHE_BYTES == 0
