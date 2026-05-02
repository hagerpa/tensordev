from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from math import prod
from typing import Callable, Literal, Sequence, Union

import jax.numpy as jnp

Array = jnp.ndarray

DyadicOrder = Union[int, tuple[int, int]]


def normalize_dyadic_order(dyadic_order: DyadicOrder) -> tuple[int, int]:
    """Normalize ``dyadic_order`` to a ``(dx_order, dy_order)`` pair.

    Accepts either a single non-negative integer (applied to both axes) or a
    tuple of two non-negative integers.
    """
    if isinstance(dyadic_order, (list, tuple)):
        if len(dyadic_order) != 2:
            raise ValueError("dyadic_order tuple must have exactly two elements.")
        dx, dy = int(dyadic_order[0]), int(dyadic_order[1])
    else:
        dx = dy = int(dyadic_order)
    if dx < 0 or dy < 0:
        raise ValueError(f"dyadic_order must be non-negative, got ({dx}, {dy}).")
    return dx, dy


def velocity_to_increments(
        velocity: Callable[[Array], tuple[Array, ...]],
        grid: Array,
        *,
        quadrature: Literal["left", "midpoint", "trapezoid"] = "trapezoid",
        dyadic_order: int = 0,
) -> tuple[Array, ...]:
    """Convert a characteristic velocity to tensor-valued interval increments.

    For each interval ``[t_{i-1}, t_i]`` of the (optionally refined) *grid*,
    the velocity is numerically integrated by the chosen quadrature rule to
    produce one increment per tensor level.

    Parameters
    ----------
    velocity :
        Callable mapping a 1-D array of time-points to a tuple of tensor
        level arrays.
    grid :
        1-D node grid with at least two nodes.
    quadrature : {"left", "midpoint", "trapezoid"}, default="trapezoid"
        Quadrature rule.
    dyadic_order : int, default=0
        If positive, the node grid is dyadically refined (midpoints inserted
        ``dyadic_order`` times) before integration.

    Returns
    -------
    tuple of Array
        One array per tensor level, each with shape ``(S_fine, width)``.
    """
    grid = jnp.asarray(grid)
    if grid.ndim != 1:
        raise ValueError("The node grid must be one-dimensional.")
    if grid.shape[0] < 2:
        raise ValueError("The node grid must contain at least two nodes.")
    if dyadic_order < 0:
        raise ValueError("dyadic_order must be non-negative.")

    if dyadic_order > 0:
        for _ in range(dyadic_order):
            mids = 0.5 * (grid[:-1] + grid[1:])
            grid = jnp.sort(jnp.concatenate([grid, mids], axis=0))

    t0 = grid[:-1]
    t1 = grid[1:]
    dt = t1 - t0

    if quadrature == "left":
        vals = velocity(t0)
        return tuple(v * jnp.expand_dims(dt, axis=-1) for v in vals)
    if quadrature == "midpoint":
        tm = 0.5 * (t0 + t1)
        vals = velocity(tm)
        return tuple(v * jnp.expand_dims(dt, axis=-1) for v in vals)
    if quadrature == "trapezoid":
        v0 = velocity(t0)
        v1 = velocity(t1)
        return tuple(
            0.5 * (a + b) * jnp.expand_dims(dt, axis=-1) for a, b in zip(v0, v1)
        )
    raise ValueError(f"Unknown quadrature={quadrature!r}.")


def _flatten_path_sample_axes(x) -> tuple[Array, int]:
    """
    Flatten all leading sample axes of Euclidean path values into one axis.

    Input shape:
        batch + (length, dim)
    Output shape:
        (n, length, dim)
    """
    x = jnp.asarray(x)
    if x.ndim < 2:
        raise ValueError(
            "Expected Euclidean path values with shape batch + (length, dim)."
        )
    if x.shape[-2] < 1:
        raise ValueError("Path length must be positive.")
    if x.shape[-1] < 1:
        raise ValueError("Path dimension must be positive.")

    batch_shape = tuple(x.shape[:-2])
    n = int(prod(batch_shape)) if len(batch_shape) > 0 else 1
    return x.reshape((n,) + x.shape[-2:]), n


def _pad_path_values_to_length(x: Array, target_length: int) -> Array:
    """
    Pad Euclidean path values to a common node length by repeating the terminal value.

    This ensures that differencing along the time axis later produces a trailing
    zero-increment tail.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected shape (batch, length, dim), got {x.shape}.")

    cur_len = int(x.shape[-2])
    if cur_len < 1:
        raise ValueError("Cannot pad an empty path.")
    if cur_len == target_length:
        return x
    if cur_len > target_length:
        raise ValueError(
            f"Cannot pad from length {cur_len} down to {target_length}."
        )

    pad_len = target_length - cur_len
    tail = jnp.repeat(x[:, -1:, :], repeats=pad_len, axis=-2)
    return jnp.concatenate([x, tail], axis=-2)


def bucket_pad_ragged_paths(
        x: Sequence[Array],
        *,
        chunk_size: int | None = None,
        sort: bool = True,
        return_indices: bool = False,
):
    """
    Bucket and pad a ragged batch of Euclidean path values.

    Each outer element of `x` is interpreted as either
    - one path of shape (length, dim), or
    - a dense mini-batch of paths with shape batch + (length, dim).

    All leading sample axes are flattened, then the logical samples are optionally
    sorted by path length, grouped into chunks of size at most `chunk_size`, and
    padded inside each chunk to the local maximum path length by repeating the
    terminal path value.

    Parameters
    ----------
    x :
        Ragged batch of Euclidean path values.
    chunk_size :
        Maximum number of logical samples per chunk. If ``None`` or non-positive,
        all samples are placed into a single chunk.
    sort : bool, default=True
        Whether to sort logical samples by path length before chunking.
    return_indices : bool, default=False
        If ``True``, also return one index tuple per chunk that records the
        original logical sample order. This can be used downstream to reassemble
        outputs in the original order.

    Returns
    -------
    list[Array]
        Dense padded chunks, each of shape ``(n_chunk, max_len, dim)``.

    or

    tuple[list[Array], list[tuple[int, ...]]]
        If ``return_indices=True``, also return the original logical indices for
        each chunk.
    """
    if not isinstance(x, SequenceABC) or isinstance(x, (str, bytes)) or len(x) == 0:
        raise ValueError("Expected a non-empty ragged batch of Euclidean paths.")

    samples = []
    next_idx = 0
    path_dim: int | None = None

    for sample in x:
        flat, n_block = _flatten_path_sample_axes(sample)

        dim = int(flat.shape[-1])
        if path_dim is None:
            path_dim = dim
        elif dim != path_dim:
            raise ValueError(
                "All paths must share the same terminal feature dimension; "
                f"got {path_dim} and {dim}."
            )

        for i in range(n_block):
            one = flat[i:i + 1]
            samples.append((one, next_idx, int(one.shape[-2])))
            next_idx += 1

    n_total = len(samples)
    if n_total == 0:
        empty_chunks: list[Array] = []
        empty_indices: list[tuple[int, ...]] = []
        return (empty_chunks, empty_indices) if return_indices else empty_chunks

    if sort:
        samples.sort(key=lambda item: item[2])

    if chunk_size is None or chunk_size <= 0:
        chunk_size = n_total

    chunks: list[Array] = []
    indices: list[tuple[int, ...]] = []

    for start in range(0, n_total, chunk_size):
        block = samples[start:start + chunk_size]
        target_length = max(item[2] for item in block)

        padded = [_pad_path_values_to_length(item[0], target_length) for item in block]
        chunk = jnp.concatenate(padded, axis=0)

        chunks.append(chunk)
        indices.append(tuple(item[1] for item in block))

    return (chunks, indices) if return_indices else chunks