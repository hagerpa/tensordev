"""Shared causal FFT primitives used by Volterra iteration schemes."""

from __future__ import annotations

import jax
import jax.numpy as jnp


Array = jax.Array


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def apply_transformed_causal_fft(
    sources: Array,
    transformed_weights: Array,
    *,
    nfft: int,
    out_len: int,
) -> Array:
    """Apply channel-wise causal convolutions from frequency-domain weights.

    ``sources`` has shape ``(channels, time, *batch, coordinates)`` and
    ``transformed_weights`` has shape ``(channels, nfft // 2 + 1)``.  The
    returned array preserves every non-time axis and has ``out_len`` time
    entries.  Flattening only the trailing payload lets both Adams and the
    basis-expansion FFT scheme share this numerical primitive.
    """
    channels, steps = sources.shape[:2]
    trailing = sources.shape[2:]
    flattened = sources.reshape((channels, steps, -1))
    transformed_sources = jnp.fft.rfft(flattened, n=nfft, axis=1)
    result = jnp.fft.irfft(
        transformed_sources * transformed_weights[:, :, None],
        n=nfft,
        axis=1,
    )
    return result[:, :out_len].reshape((channels, out_len) + trailing).astype(
        sources.dtype
    )


def causal_fft_from_raw_weights(sources: Array, weights: Array) -> Array:
    """Apply one causal convolution to time-leading ``sources``.

    This Adams-facing wrapper interprets ``weights[j]`` as the lag ``j + 1``
    coefficient and returns exactly ``sources.shape[0]`` time entries.
    """
    steps = int(sources.shape[0])
    nfft = next_power_of_two(2 * steps - 1)
    transformed = jnp.fft.rfft(weights, n=nfft, axis=0)[None, :]
    return apply_transformed_causal_fft(
        sources[None, ...],
        transformed,
        nfft=nfft,
        out_len=steps,
    )[0]


__all__ = [
    "apply_transformed_causal_fft",
    "causal_fft_from_raw_weights",
    "next_power_of_two",
]
