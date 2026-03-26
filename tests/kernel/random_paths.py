from __future__ import annotations

import math

import jax.numpy as jnp
import jax.random as jr
from jax import lax


def path_to_increments(X):
    """
    Convert a DenseElemFirstOn path into interval increments.
    """
    return tuple(jnp.diff(level, axis=-2) for level in X)


def integrated_ou_first_on_path(
        key,
        *,
        batch,
        steps,
        dim,
        trunc,
        horizon=1.0,
        mean_reversion=5.0,
        volatility=5.0,
        level_lambda=1.0,
        factorial=False,
):
    """
    Dense first-on path whose every flattened coordinate is an independent
    integrated Ornstein-Uhlenbeck process.

    The level-k component is scaled by

        level_lambda**k

    and, if ``factorial=True``, additionally by ``1 / k!``.
    """
    dt = horizon / steps
    theta = mean_reversion
    alpha = math.exp(-theta * dt)

    level_keys = jr.split(key, trunc)
    levels = []

    for k in range(1, trunc + 1):
        width = dim ** k

        level_scale = level_lambda ** k
        if factorial:
            level_scale = level_scale / math.factorial(k)

        sigma_k = volatility * level_scale

        if theta == 0.0:
            beta = sigma_k * math.sqrt(dt)
        else:
            beta = sigma_k * math.sqrt((1.0 - math.exp(-2.0 * theta * dt)) / (2.0 * theta))

        noises = jr.normal(level_keys[k - 1], (steps, batch, width))
        x0 = jnp.zeros((batch, width), dtype=noises.dtype)
        u0 = jnp.zeros((batch, width), dtype=noises.dtype)

        def step(carry, noise):
            u_prev, x_prev = carry
            u_next = alpha * u_prev + beta * noise
            x_next = x_prev + 0.5 * dt * (u_prev + u_next)
            return (u_next, x_next), x_next

        (_, _), xs = lax.scan(step, (u0, x0), noises)
        Xk = jnp.concatenate([x0[:, None, :], jnp.moveaxis(xs, 0, 1)], axis=1)
        levels.append(Xk)

    return tuple(levels)


def random_trigonometric_polynomial_paths(
        key,
        *,
        batch,
        steps,
        dim,
        n_modes=5,
        scale=0.25,
        decay=1.5,
        horizon=1.0,
):
    """
    Batched random trigonometric-polynomial paths.
    """
    key_sin, key_cos = jr.split(key, 2)

    t = jnp.linspace(0.0, horizon, steps + 1, dtype=jnp.float64)
    modes = jnp.arange(1, n_modes + 1, dtype=jnp.float64)
    weights = scale / (modes ** decay)

    sin_coeffs = jr.normal(
        key_sin,
        (batch, dim, n_modes),
        dtype=jnp.float64,
    ) * weights[None, None, :]
    cos_coeffs = jr.normal(
        key_cos,
        (batch, dim, n_modes),
        dtype=jnp.float64,
    ) * weights[None, None, :]

    angles = 2.0 * math.pi * modes[:, None] * t[None, :] / horizon
    sin_basis = jnp.sin(angles)
    cos_basis = jnp.cos(angles) - 1.0

    X = (
        jnp.einsum("bdm,mt->bdt", sin_coeffs, sin_basis)
        + jnp.einsum("bdm,mt->bdt", cos_coeffs, cos_basis)
    )

    return jnp.swapaxes(X, 1, 2)


def random_trigonometric_polynomial_paths_first_on(
        key,
        *,
        batch,
        steps,
        dim,
        trunc,
        level_lambda=1.0,
        factorial=False,
        **kwargs,
):
    """
    Dense first-on path built from independent random trigonometric polynomial
    paths at each positive tensor level.

    The level-k component has shape
        (batch, steps + 1, dim**k)

    and is generated independently using
    ``random_trigonometric_polynomial_paths`` with ambient dimension ``dim**k``.

    The level-k component is then scaled by

        level_lambda**k

    and, if ``factorial=True``, additionally by ``1 / k!``.

    Parameters
    ----------
    key :
        JAX PRNG key.

    batch : int
        Batch size.

    steps : int
        Number of intervals, so the returned paths have ``steps + 1`` time points.

    dim : int
        Base dimension.

    trunc : int
        Number of positive tensor levels to generate.

    level_lambda : float, default=1.0
        Per-level geometric scaling parameter.

    factorial : bool, default=False
        If True, apply additional factorial damping ``1 / k!``.

    **kwargs :
        Passed through to ``random_trigonometric_polynomial_paths``.

    Returns
    -------
    tuple
        ``(X_1, ..., X_trunc)`` with
        ``X_k.shape == (batch, steps + 1, dim**k)``.
    """
    level_keys = jr.split(key, trunc)

    levels = []
    for k in range(1, trunc + 1):
        level_scale = level_lambda ** k
        if factorial:
            level_scale = level_scale / math.factorial(k)

        Xk = random_trigonometric_polynomial_paths(
            level_keys[k - 1],
            batch=batch,
            steps=steps,
            dim=dim ** k,
            **kwargs,
        )
        levels.append(level_scale * Xk)

    return tuple(levels)