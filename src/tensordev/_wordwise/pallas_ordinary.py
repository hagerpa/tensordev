"""Tiled Pallas executor for homogeneous ordered signature coordinates.

The executor is intentionally layout-neutral.  It consumes canonical
increments and optional total-word codes, and returns arrays for a caller to
assemble into the selected tensor container.  Device selection and fallback
belong to the dispatch layer.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tensordev._wordwise.dispatch import _colocate_array, _concrete_single_device


Array = jax.Array

_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_DEGREE = 32
_MAX_TILE_WORDS = 1024
_MAX_LOCAL_BYTES = 64 * 1024
_MAX_OUTPUT_ELEMENTS = _INT32_MAX


class PallasOrdinaryError(RuntimeError):
    """Base error raised before an ordered Pallas kernel is lowered."""


class PallasOrdinaryUnavailableError(PallasOrdinaryError):
    """Raised when the supported Pallas/Triton API cannot be imported."""


class PallasOrdinaryUnsupportedError(PallasOrdinaryError):
    """Raised for an input or dtype outside the executor contract."""


class PallasOrdinaryResourceError(PallasOrdinaryError):
    """Raised when static kernel resources exceed a deterministic guard."""


def _load_pallas():
    """Import the experimental execution backend only when it is requested."""
    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as pltriton
    except (ImportError, AttributeError) as exc:
        raise PallasOrdinaryUnavailableError(
            "the installed JAX build does not provide the required "
            "Pallas/Triton API"
        ) from exc
    return pl, pltriton


def _require_integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _normalize_block_size(block_size: int | None, *, steps: int) -> tuple[int, int]:
    if block_size is None or block_size == -1:
        return steps, 1
    block_size = _require_integer(block_size, name="block_size", minimum=1)
    block_count, remainder = divmod(steps, block_size)
    if remainder:
        raise ValueError(
            f"block_size={block_size} must divide the step count {steps}."
        )
    return block_size, block_count


def _normalize_word_codes(
        word_codes: Any,
        *,
        word_space: int,
        degree: int,
) -> tuple[Array | None, int]:
    if word_codes is None:
        return None, word_space

    host_codes = None
    if isinstance(word_codes, (list, tuple, np.ndarray)):
        host_codes = np.asarray(word_codes)
        if host_codes.dtype != np.dtype(np.int32):
            raise PallasOrdinaryUnsupportedError(
                "word_codes must have dtype int32, got "
                f"{host_codes.dtype}."
            )

    # Keep host decoder tables on the host through validation and padding.
    # The executor can then transfer them directly to the input device instead
    # of first materializing a JAX array on the process-default device.
    codes = host_codes if host_codes is not None else jnp.asarray(word_codes)
    if codes.dtype != jnp.int32:
        raise PallasOrdinaryUnsupportedError(
            f"word_codes must have dtype int32, got {codes.dtype}."
        )
    if codes.ndim != 1:
        raise ValueError(
            f"word_codes must be one-dimensional, got shape {codes.shape}."
        )
    word_count = int(codes.shape[0])
    if word_count == 0:
        raise ValueError("word_codes must contain at least one code.")
    if word_count > _INT32_MAX:
        raise PallasOrdinaryResourceError(
            f"word_codes contains {word_count} entries, exceeding int32 indexing."
        )
    if degree == 0 and word_count != 1:
        raise ValueError("degree zero has exactly one coordinate.")

    if host_codes is not None:
        if np.any(host_codes < 0) or np.any(host_codes >= word_space):
            raise ValueError(
                "word_codes entries must lie in "
                f"[0, {word_space}), got range "
                f"[{int(host_codes.min())}, {int(host_codes.max())}]."
            )
    return codes, word_count


def _padded_word_count(word_count: int, tile_words: int) -> int:
    return ((word_count + tile_words - 1) // tile_words) * tile_words


def _effective_tile_words(word_count: int, requested: int) -> int:
    smallest_covering_tile = 1 << (word_count - 1).bit_length()
    return min(requested, smallest_covering_tile)


def _check_resources(
        *,
        flat_batch: int,
        steps: int,
        block_count: int,
        alphabet_dim: int,
        degree: int,
        tile_words: int,
        padded_word_count: int,
        itemsize: int,
) -> None:
    for name, value in (
        ("flat batch size", flat_batch),
        ("step count", steps),
        ("block count", block_count),
        ("alphabet dimension", alphabet_dim),
        ("padded word count", padded_word_count),
    ):
        if value > _INT32_MAX:
            raise PallasOrdinaryResourceError(
                f"{name} {value} exceeds signed int32 indexing."
            )

    # Prefixes, decoded letters, the current word increments, and the Horner
    # accumulator are simultaneously live in the Triton kernel.
    local_bytes = tile_words * (
        (degree + 1) * itemsize
        + degree * np.dtype(np.int32).itemsize
        + degree * itemsize
        + itemsize
    )
    if local_bytes > _MAX_LOCAL_BYTES:
        raise PallasOrdinaryResourceError(
            "ordered word tile requires an estimated "
            f"{local_bytes} local bytes, exceeding the "
            f"{_MAX_LOCAL_BYTES}-byte guard; reduce tile_words."
        )

    output_elements = flat_batch * block_count * padded_word_count
    if output_elements > _MAX_OUTPUT_ELEMENTS:
        raise PallasOrdinaryResourceError(
            "padded output contains "
            f"{output_elements} elements, exceeding the int32 execution guard."
        )


def _compiler_warps(tile_words: int) -> int:
    return min(max(tile_words // 32, 1), 8)


def _initialize_program_state(
        pl,
        pltriton,
        codes_ref,
        *,
        explicit_codes: bool,
        divisors: tuple[int, ...],
        alphabet_dim: int,
        degree: int,
        tile_words: int,
        dtype: np.dtype,
):
    batch_index = pl.program_id(0)
    word_positions = (
        pl.program_id(1) * tile_words
        + jnp.arange(tile_words, dtype=jnp.int32)
    )
    codes = (
        pltriton.load(codes_ref.at[word_positions])
        if explicit_codes
        else word_positions
    )
    letters = tuple(
        (codes // divisor) % alphabet_dim for divisor in divisors
    )
    prefixes = jnp.zeros((degree + 1, tile_words), dtype=dtype)
    prefixes = prefixes.at[0].set(1)
    return batch_index, word_positions, letters, prefixes


def _advance_prefixes(
        increments_ref,
        pltriton,
        prefixes,
        *,
        batch_index,
        time_index,
        letters,
        degree: int,
        tile_words: int,
        dtype: np.dtype,
):
    word_increments = tuple(
        pltriton.load(
            increments_ref.at[batch_index, time_index, letter]
        )
        for letter in letters
    )
    for prefix_length in range(degree, 0, -1):
        horner = jnp.zeros((tile_words,), dtype=dtype)
        for prefix_index in range(prefix_length):
            horner = (
                word_increments[prefix_index]
                * (prefixes[prefix_index] + horner)
                / jnp.asarray(prefix_length - prefix_index, dtype=dtype)
            )
        prefixes = prefixes.at[prefix_length].add(horner)
    return prefixes


@lru_cache(maxsize=128)
def _ordered_call(
        shape: tuple[int, int, int],
        dtype: np.dtype,
        degree: int,
        padded_word_count: int,
        tile_words: int,
        emission_block_size: int | None,
        explicit_codes: bool,
        interpret: bool,
):
    pl, pltriton = _load_pallas()
    flat_batch, steps, alphabet_dim = shape
    block_count = (
        1 if emission_block_size is None else steps // emission_block_size
    )
    divisors = tuple(
        alphabet_dim ** power for power in range(degree - 1, -1, -1)
    )

    def run_kernel(increments_ref, codes_ref, output_ref):
        batch_index, word_positions, letters, prefixes = (
            _initialize_program_state(
                pl,
                pltriton,
                codes_ref,
                explicit_codes=explicit_codes,
                divisors=divisors,
                alphabet_dim=alphabet_dim,
                degree=degree,
                tile_words=tile_words,
                dtype=dtype,
            )
        )

        def time_step(time_index, current):
            return _advance_prefixes(
                increments_ref,
                pltriton,
                current,
                batch_index=batch_index,
                time_index=time_index,
                letters=letters,
                degree=degree,
                tile_words=tile_words,
                dtype=dtype,
            )

        if emission_block_size is None:
            prefixes = jax.lax.fori_loop(0, steps, time_step, prefixes)
            pltriton.store(
                output_ref.at[batch_index, word_positions],
                prefixes[degree],
            )
            return

        def block_step(block_index, current):
            start = block_index * emission_block_size

            def local_time_step(local_index, block_current):
                return time_step(start + local_index, block_current)

            current = jax.lax.fori_loop(
                0,
                emission_block_size,
                local_time_step,
                current,
            )
            pltriton.store(
                output_ref.at[batch_index, block_index, word_positions],
                current[degree],
            )
            return current

        jax.lax.fori_loop(0, block_count, block_step, prefixes)

    if explicit_codes:
        kernel = run_kernel
    else:
        def kernel(increments_ref, output_ref):
            return run_kernel(increments_ref, None, output_ref)

    output_shape = (
        (flat_batch, padded_word_count)
        if emission_block_size is None
        else (flat_batch, block_count, padded_word_count)
    )
    mode = "terminal" if emission_block_size is None else "accumulated"
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
        grid=(flat_batch, padded_word_count // tile_words),
        compiler_params=pltriton.CompilerParams(
            num_warps=_compiler_warps(tile_words),
            num_stages=1,
        ),
        interpret=interpret,
        name=f"tensordev_ordered_ordinary_{mode}",
    )


def ordered_ordinary_pallas(
        increments: Array,
        *,
        degree: int,
        word_codes: Any = None,
        block_size: int | None = None,
        accumulate: bool = True,
        tile_words: int = 128,
        interpret: bool = False,
) -> Array:
    """Compute one homogeneous ordered signature block with Pallas.

    Parameters
    ----------
    increments : array
        Canonical increments with shape ``(flat_batch, steps, alphabet)``.
    degree : int
        Homogeneous total degree of every requested coordinate.
    word_codes : one-dimensional int32 array, optional
        Total-word codes in output order.  ``None`` requests the dense order
        ``0, ..., alphabet**degree - 1`` without a decoder operand.
    block_size : int, optional
        Steps per emitted block.  ``None`` and ``-1`` select the complete path.
    accumulate : bool, default True
        If several blocks are requested, carry prefix state across blocks when
        true; otherwise compute every block independently from the unit.
    tile_words : int, default 128
        Power-of-two number of independent words owned by one Pallas program.
    interpret : bool, default False
        Use Pallas interpreter mode.  This is a CPU correctness aid, not a GPU
        capability or performance test.

    Returns
    -------
    array
        Shape ``(flat_batch, words)`` for one block, otherwise
        ``(flat_batch, block_count, words)``.  No tensor container or scalar
        level is assembled here, except that ``degree=0`` directly returns its
        scalar block.
    """
    increments = jnp.asarray(increments)
    if increments.ndim != 3:
        raise ValueError(
            "increments must have canonical shape "
            f"(flat_batch, steps, alphabet), got {increments.shape}."
        )
    flat_batch, steps, alphabet_dim = map(int, increments.shape)
    if flat_batch <= 0:
        raise PallasOrdinaryUnsupportedError(
            "the Pallas executor requires a positive flat batch size."
        )
    if steps <= 0:
        raise PallasOrdinaryUnsupportedError(
            "the Pallas executor requires at least one increment."
        )
    if alphabet_dim <= 0:
        raise PallasOrdinaryUnsupportedError(
            "the Pallas executor requires a positive alphabet dimension."
        )

    dtype = np.dtype(increments.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise PallasOrdinaryUnsupportedError(
            "increments must have dtype float32 or float64, got "
            f"{dtype}."
        )
    degree = _require_integer(degree, name="degree", minimum=0)
    if degree > _MAX_DEGREE:
        raise PallasOrdinaryResourceError(
            f"degree {degree} exceeds the {_MAX_DEGREE}-degree kernel guard."
        )
    if not isinstance(accumulate, bool):
        raise TypeError("accumulate must be a bool.")
    if not isinstance(interpret, bool):
        raise TypeError("interpret must be a bool.")
    tile_words = _require_integer(
        tile_words, name="tile_words", minimum=1
    )
    if tile_words > _MAX_TILE_WORDS or tile_words & (tile_words - 1):
        raise PallasOrdinaryResourceError(
            "tile_words must be a power of two no greater than "
            f"{_MAX_TILE_WORDS}, got {tile_words}."
        )

    word_space = alphabet_dim**degree
    if word_space > _INT32_MAX:
        raise PallasOrdinaryResourceError(
            f"alphabet**degree={word_space} exceeds signed int32 indexing."
        )
    codes, word_count = _normalize_word_codes(
        word_codes,
        word_space=word_space,
        degree=degree,
    )
    tile_words = _effective_tile_words(word_count, tile_words)
    padded_count = _padded_word_count(word_count, tile_words)
    block_size, block_count = _normalize_block_size(
        block_size, steps=steps
    )
    _check_resources(
        flat_batch=flat_batch,
        steps=steps,
        block_count=block_count,
        alphabet_dim=alphabet_dim,
        degree=degree,
        tile_words=tile_words,
        padded_word_count=padded_count,
        itemsize=dtype.itemsize,
    )

    if degree == 0:
        shape = (
            (flat_batch, 1)
            if block_count == 1
            else (flat_batch, block_count, 1)
        )
        return jnp.ones(
            shape,
            dtype=increments.dtype,
            device=_concrete_single_device(increments),
        )

    explicit_codes = codes is not None
    if explicit_codes and padded_count != word_count:
        padding = (0, padded_count - word_count)
        codes = (
            np.pad(codes, padding)
            if isinstance(codes, np.ndarray)
            else jnp.pad(codes, padding)
        )
    if explicit_codes:
        codes = _colocate_array(codes, increments)

    shape = (flat_batch, steps, alphabet_dim)
    if block_count == 1:
        call = _ordered_call(
            shape,
            dtype,
            degree,
            padded_count,
            tile_words,
            None,
            explicit_codes,
            interpret,
        )
        result = call(increments, codes) if explicit_codes else call(increments)
        return result[:, :word_count]

    if not accumulate:
        blocks = increments.reshape(
            flat_batch * block_count,
            block_size,
            alphabet_dim,
        )
        block_shape = (flat_batch * block_count, block_size, alphabet_dim)
        call = _ordered_call(
            block_shape,
            dtype,
            degree,
            padded_count,
            tile_words,
            None,
            explicit_codes,
            interpret,
        )
        result = call(blocks, codes) if explicit_codes else call(blocks)
        return result[:, :word_count].reshape(
            flat_batch, block_count, word_count
        )

    call = _ordered_call(
        shape,
        dtype,
        degree,
        padded_count,
        tile_words,
        block_size,
        explicit_codes,
        interpret,
    )
    result = call(increments, codes) if explicit_codes else call(increments)
    return result[:, :, :word_count]


def clear_pallas_ordinary_cache() -> None:
    """Clear the bounded Pallas-call factory cache."""
    _ordered_call.cache_clear()


__all__ = [
    "PallasOrdinaryError",
    "PallasOrdinaryResourceError",
    "PallasOrdinaryUnavailableError",
    "PallasOrdinaryUnsupportedError",
    "clear_pallas_ordinary_cache",
    "ordered_ordinary_pallas",
]
