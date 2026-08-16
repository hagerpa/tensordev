"""
Bucketing and padding of ragged path batches.

The default pads each chunk to its own longest path, which keeps the padding
minimal but makes chunk shapes depend on the contents. `target_length` pins
them instead, so a downstream jitted function sees one shape and compiles once.
"""
from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest

from tensordev.util.path_preprocessing import bucket_pad_ragged_paths

LENGTHS = (3, 5, 4, 7)


def _paths(dim: int = 2):
    return [
        np.arange(n * dim, dtype=float).reshape(1, n, dim) for n in LENGTHS
    ]


def test_chunks_pad_to_the_local_maximum_by_default():
    chunks = bucket_pad_ragged_paths(_paths(), chunk_size=2, sort=True)
    # sorted by length, so the chunks hold (3, 4) and (5, 7)
    assert [tuple(chunk.shape) for chunk in chunks] == [(2, 4, 2), (2, 7, 2)]


def test_target_length_makes_every_chunk_the_same_shape():
    chunks = bucket_pad_ragged_paths(_paths(), chunk_size=2, target_length=9)
    assert [tuple(chunk.shape) for chunk in chunks] == [(2, 9, 2), (2, 9, 2)]


def test_padding_repeats_the_terminal_value():
    """So that differencing the padded path yields a zero-increment tail."""
    path = np.arange(6, dtype=float).reshape(1, 3, 2)
    (chunk,) = bucket_pad_ragged_paths([path], chunk_size=1, target_length=5)

    np.testing.assert_allclose(np.asarray(chunk)[0, :3], path[0])
    np.testing.assert_allclose(
        np.asarray(chunk)[0, 3:], np.broadcast_to(path[0, -1], (2, 2)),
    )
    np.testing.assert_allclose(np.diff(np.asarray(chunk)[0], axis=0)[2:], 0.0)


def test_target_length_shorter_than_a_path_is_rejected():
    with pytest.raises(ValueError, match="Cannot pad from length"):
        bucket_pad_ragged_paths(_paths(), chunk_size=2, target_length=4)


def test_indices_recover_the_original_order():
    chunks, indices = bucket_pad_ragged_paths(
        _paths(), chunk_size=2, target_length=9, return_indices=True,
    )
    flat = [index for chunk_indices in indices for index in chunk_indices]
    assert sorted(flat) == list(range(len(LENGTHS)))

    restored = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=0)
    for position, original in enumerate(flat):
        length = LENGTHS[original]
        np.testing.assert_allclose(
            restored[position, :length], _paths()[original][0],
        )
