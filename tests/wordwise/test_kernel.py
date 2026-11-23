"""The wordwise kernel in interpret mode against a dense evaluation of its recursion."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev._wordwise import kernel
from tensordev._wordwise.kernel import (
    WordwiseResourceError,
    WordwiseUnsupportedError,
    clear_kernel_caches,
    launches,
    wordwise_blocks,
)
from tensordev._wordwise.layout import build_layout_plan

CORES = {
    "total": dict(dims=2, max_trunc=4),
    "total-one-level": dict(dims=3, max_trunc=1),
    "ordered": dict(dims=(2, 2), max_trunc=(2, 2)),
    "ordered-single-letters": dict(dims=(1, 1), max_trunc=(2, 2)),
    "ordered-one-doubleprime-letter": dict(dims=(3, 1), max_trunc=(1, 2)),
    "ordered-prime-only": dict(dims=(1, 2), max_trunc=(2, 0)),
    "partial": dict(dims=(2, 2), max_trunc=(2, 2), partially_symmetrized=True),
    "partial-wide": dict(dims=(1, 3), max_trunc=(1, 3), partially_symmetrized=True),
    "partial-no-prime": dict(dims=(2, 2), max_trunc=(0, 2), partially_symmetrized=True),
    "partial-single-letters": dict(dims=(1, 1), max_trunc=(2, 2), partially_symmetrized=True),
}
BATCH, STEPS, BLOCK, STATE = 2, 4, 2, 3


def _plan(name):
    core = td.make_core(**CORES[name])
    return core, build_layout_plan(core, core.max_truncation)


def _trees(plan):
    degree = plan.blocks[-1].total_degree
    return sorted({1, min(2, degree), degree}) if plan.grading == "total_degree" else [None]


def _increments(plan, seed=0):
    return np.random.default_rng(seed).normal(size=(BATCH, STEPS, plan.alphabet_dim)) / 2


def _cat(blocks):
    return np.concatenate([np.asarray(block) for block in blocks], axis=-1)


def _dense(y, E, psi, phi, P):
    """``P <- P E + sum_n X_n A^n`` over all words; level ``k`` has shape ``(R,) + (d,) * k``."""
    N = len(P)
    for t in range(len(y)):
        H = [0.0] * (N + 1)
        for s in range(N, 0, -1):
            X = [psi[t, s - 1]] + [
                np.einsum("r...,rc->c...", P[k - 1], phi[t, s - 1]) for k in range(1, N - s + 1)
            ]
            H = [0.0] + [(X[k] + H[k])[..., None] * y[t] for k in range(N - s + 1)] + [0.0] * (s - 1)
        P = [np.einsum("r...,rc->c...", P[k], E[t]) + H[k + 1] for k in range(N)]
    return P


def _dense_blocks(y, E, psi, phi, seed, block_size, accumulate):
    """Dense levels after every emitted block: level ``k`` has shape ``(blocks, R) + (d,) * k``."""
    out, P = [], seed
    for start in range(0, len(y), block_size):
        window = slice(start, start + block_size)
        P = _dense(y[window], E[window], psi[window], phi[window], P if accumulate else seed)
        out.append(P)
    return [np.stack(level) for level in zip(*out)]


def _coordinates(core, plan, levels):
    """Dense levels ``lead + (d,) * k`` of all words in the coordinates of a plan, ``lead + (W,)``."""
    lead = levels[0].shape[:-1]
    levels = [level.reshape(lead + (-1,)) for level in levels]
    if plan.grading == "total_degree":
        return np.concatenate(levels, axis=-1)
    ordered = build_layout_plan(td.make_core(dims=plan.dims, max_trunc=plan.truncation), plan.truncation)
    blocks = [levels[block.total_degree - 1][..., block.word_codes] for block in ordered.blocks[1:]]
    if plan.partially_symmetrized:
        blocks = core.tensor_partially_symmetrize(ordered.assemble_first_on(blocks), first_on=True).blocks
    return _cat(blocks)


@pytest.mark.parametrize("name", CORES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_signature_matches_path_signature(name, dtype):
    core, plan = _plan(name)
    increments = jnp.asarray(_increments(plan), dtype=dtype)
    tolerance = dict(float32=2e-5, float64=1e-12)[dtype]

    def reference(**options):
        signature = td.path_signature(increments, core=core, increment_input=True, execution="jax", **options)
        return _cat(getattr(signature, "blocks", signature))

    for tree in _trees(plan):
        for tile in (2, 32):
            got = wordwise_blocks(increments, plan=plan, tile=tile, tree=tree, interpret=True)
            np.testing.assert_allclose(_cat(got), reference(), rtol=tolerance, atol=tolerance)
    for accumulate in (True, False):
        got = wordwise_blocks(increments, plan=plan, block_size=BLOCK, accumulate=accumulate, interpret=True)
        np.testing.assert_allclose(
            _cat(got), reference(block_size=BLOCK, accumulate=accumulate), rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("name", CORES)
@pytest.mark.parametrize("seeded", [False, True], ids=["zero-shared", "seeded-batched"])
def test_state_matches_dense_recursion(name, seeded):
    """Zero seed and shared coefficients, or a batch of seeds and coefficients per batch and step."""
    core, plan = _plan(name)
    rng = np.random.default_rng(1)
    N, d = plan.blocks[-1].total_degree, plan.alphabet_dim
    axes = (BATCH, STEPS) if seeded else (1, 1)
    y = _increments(plan, seed=2)
    E, psi, phi = (
        rng.normal(size=axes + (STATE, STATE)) / 2,
        rng.normal(size=axes + (N, STATE)),
        rng.normal(size=axes + (N - 1, STATE, STATE)) / 2,
    )
    weights = rng.normal(size=(BATCH if seeded else 1, STATE))
    seed = [
        rng.normal(size=(BATCH, STATE) + (d,) * k) if seeded else np.zeros((BATCH, STATE) + (d,) * k)
        for k in range(1, N + 1)
    ]
    dense = [np.broadcast_to(c, (BATCH, STEPS) + c.shape[2:]) for c in (E, psi, phi)]

    def expected(block_size, accumulate):
        per_batch = [
            _dense_blocks(y[b], *(c[b] for c in dense), [level[b] for level in seed], block_size, accumulate)
            for b in range(BATCH)
        ]
        return _coordinates(core, plan, [np.stack(level) for level in zip(*per_batch)])

    options = dict(
        plan=plan, coefficients=(E, psi, phi), interpret=True,
        initial=np.moveaxis(_coordinates(core, plan, seed), -1, -2) if seeded else None,
    )
    terminal = expected(STEPS, True)[:, 0]
    for tree in _trees(plan):
        got = wordwise_blocks(y, tree=tree, tile=2, **options)
        np.testing.assert_allclose(_cat(got), terminal, atol=1e-11)
    for accumulate in (True, False):
        np.testing.assert_allclose(
            _cat(wordwise_blocks(y, block_size=BLOCK, accumulate=accumulate, **options)),
            expected(BLOCK, accumulate), atol=1e-11,
        )
    np.testing.assert_allclose(
        _cat(wordwise_blocks(y, weights=weights, **options)[1:]),
        np.einsum("brw,br->bw", terminal, np.broadcast_to(weights, (BATCH, STATE))), atol=1e-11,
    )


@pytest.mark.parametrize("name", CORES)
def test_launches_store_every_coordinate_once(name):
    _, plan = _plan(name)
    for tree in _trees(plan):
        stored = []
        for launch in launches(plan, tree or 1):
            prime, doubleprime = launch.inner
            lane = np.arange(launch.lanes)[:, None]
            prime_word, doubleprime_word = lane // doubleprime % prime, lane % doubleprime
            b, s1, m1, s2, m2, w = np.moveaxis(
                launch.nodes[lane[:, 0] // (prime * doubleprime)].astype(np.int64), -1, 0
            )
            keep = (b >= 0) & (w == 1) & (prime_word % s1 == 0) & (doubleprime_word % s2 == 0)
            stored.append((b + prime_word // s1 * m1 + doubleprime_word // s2 * m2)[keep])
        np.testing.assert_array_equal(np.sort(np.concatenate(stored)), np.arange(plan.output_size - 1))


def test_kernel_nests_under_jit_and_reuses_its_factory():
    _, plan = _plan("ordered")
    clear_kernel_caches()
    run = lambda increments, **options: wordwise_blocks(increments, plan=plan, interpret=True, **options)
    first, second = _increments(plan, seed=3), _increments(plan, seed=4)

    np.testing.assert_allclose(_cat(jax.jit(run)(first)), _cat(run(first)), atol=1e-13)
    run(second, accumulate=False)

    assert kernel._call.cache_info().misses == len(launches(plan))


def test_launch_cache_is_bounded(monkeypatch):
    clear_kernel_caches()
    monkeypatch.setattr(kernel, "_LAUNCH_CACHE_SIZE", 2)
    for name in ("total", "ordered", "partial"):
        launches(_plan(name)[1])
    assert len(kernel._LAUNCHES) == 2
    clear_kernel_caches()
    assert not kernel._LAUNCHES


@pytest.mark.parametrize(
    "error, increments, options",
    [
        (WordwiseUnsupportedError, np.zeros((2, 4, 2), np.int32), {}),
        (WordwiseUnsupportedError, np.zeros((4, 2)), {}),
        (WordwiseUnsupportedError, np.zeros((0, 4, 2)), {}),
        (ValueError, np.zeros((2, 4, 3)), {}),
        (ValueError, np.zeros((2, 4, 2)), dict(block_size=3)),
        (ValueError, np.zeros((2, 4, 2)), dict(tree=5)),
        (ValueError, np.zeros((2, 4, 2)), dict(initial=np.zeros((1, 30, 3)))),
        (ValueError, np.zeros((2, 4, 2)), dict(
            coefficients=(np.zeros((1, 1, 3, 3)), np.zeros((1, 1, 3, 3)), np.zeros((1, 1, 3, 3, 3))),
        )),
        (ValueError, np.zeros((2, 4, 2)), dict(
            coefficients=(np.zeros((3, 1, 3, 3)), np.zeros((1, 1, 4, 3)), np.zeros((1, 1, 3, 3, 3))),
        )),
        (WordwiseUnsupportedError, np.zeros((2, 4, 2)), dict(
            coefficients=(np.zeros((1, 1, 3, 3)), np.zeros((1, 1, 4, 3)), np.zeros((1, 1, 3, 3, 3))),
            weights=np.zeros((1, 3)), block_size=2,
        )),
        (WordwiseResourceError, np.zeros((2, 4, 2)), dict(tile=3)),
        (WordwiseResourceError, np.zeros((2, 4, 2)), dict(
            coefficients=(np.zeros((1, 1, 96, 96)), np.zeros((1, 1, 4, 96)), np.zeros((1, 1, 3, 96, 96))),
        )),
    ],
)
def test_inputs_are_checked_before_lowering(monkeypatch, error, increments, options):
    monkeypatch.setattr(kernel, "_load_pallas", lambda: pytest.fail("the kernel was lowered"))
    clear_kernel_caches()
    with pytest.raises(error):
        wordwise_blocks(increments, plan=_plan("total")[1], **options)
