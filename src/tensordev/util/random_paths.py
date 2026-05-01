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


import numpy as np
from numba import njit


def unit_speed_paths(
    *,
    dt: float,
    dt_fine: float = 0.001,
    n_paths: int,
    dim: int = 2,
    seed: int = 0,
    dtype=np.float64,
):
    """
    Generate deterministic smooth unit-speed paths on [0, 1].

    Returns shape:
        (n_paths, int(1 / dt) + 1, dim)
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3.")
    if dt <= 0.0 or dt_fine <= 0.0:
        raise ValueError("dt and dt_fine must be positive.")
    if dt_fine > dt:
        raise ValueError("dt_fine must be <= dt.")

    n = round(1.0 / dt)
    n_fine = round(1.0 / dt_fine)
    stride = round(dt / dt_fine)

    if abs(n * dt - 1.0) > 1e-12:
        raise ValueError("dt must divide 1.0.")
    if abs(n_fine * dt_fine - 1.0) > 1e-12:
        raise ValueError("dt_fine must divide 1.0.")
    if abs(stride * dt_fine - dt) > 1e-12:
        raise ValueError("dt_fine must divide dt.")

    rng = np.random.default_rng(seed)

    if dim == 2:
        params = _sample_params_2d(rng, n_paths)
        X = _unit_speed_paths_2d_numba(n_fine, stride, params)
    else:
        params = _sample_params_3d(rng, n_paths)
        X = _unit_speed_paths_3d_numba(n_fine, stride, params)

    return X.astype(dtype, copy=False)


def _sample_params_2d(rng, n_paths: int):
    return np.column_stack(
        [
            rng.uniform(1.25, 3.75, size=n_paths),          # turns
            rng.uniform(0.2, 1.0, size=n_paths),            # a1
            rng.uniform(0.1, 0.7, size=n_paths),            # a2
            rng.uniform(0.0, 2.0 * np.pi, size=n_paths),    # phase1
            rng.uniform(0.0, 2.0 * np.pi, size=n_paths),    # phase2
        ]
    )


def _sample_params_3d(rng, n_paths: int):
    return np.column_stack(
        [
            rng.uniform(1.25, 3.75, size=n_paths),          # turns
            rng.uniform(0.2, 1.0, size=n_paths),            # a_theta
            rng.uniform(0.15, 0.65, size=n_paths),          # a_phi
            rng.uniform(0.0, 2.0 * np.pi, size=n_paths),    # phase_theta
            rng.uniform(0.0, 2.0 * np.pi, size=n_paths),    # phase_phi
            rng.uniform(0.0, 2.0 * np.pi, size=n_paths),    # phase_theta2
            rng.uniform(0.0, 2.0 * np.pi, size=n_paths),    # phase_phi2
        ]
    )


@njit(cache=True)
def _unit_speed_paths_2d_numba(n_fine: int, stride: int, params):
    n_paths = params.shape[0]
    n = n_fine // stride

    X = np.zeros((n_paths, n + 1, 2), dtype=np.float64)

    two_pi = 2.0 * np.pi
    six_pi = 6.0 * np.pi
    dt_fine = 1.0 / n_fine

    for b in range(n_paths):
        turns = params[b, 0]
        a1 = params[b, 1]
        a2 = params[b, 2]
        phase1 = params[b, 3]
        phase2 = params[b, 4]

        x0 = 0.0
        x1 = 0.0

        theta_prev = (
            two_pi * turns * 0.0
            + a1 * np.sin(two_pi * 0.0 + phase1)
            + a2 * np.sin(six_pi * 0.0 + phase2)
        )
        v0_prev = np.cos(theta_prev)
        v1_prev = np.sin(theta_prev)

        coarse_idx = 0

        for k in range(1, n_fine + 1):
            s = k * dt_fine

            theta = (
                two_pi * turns * s
                + a1 * np.sin(two_pi * s + phase1)
                + a2 * np.sin(six_pi * s + phase2)
            )

            v0 = np.cos(theta)
            v1 = np.sin(theta)

            x0 += 0.5 * (v0_prev + v0) * dt_fine
            x1 += 0.5 * (v1_prev + v1) * dt_fine

            v0_prev = v0
            v1_prev = v1

            if k % stride == 0:
                coarse_idx += 1
                X[b, coarse_idx, 0] = x0
                X[b, coarse_idx, 1] = x1

    return X


@njit(cache=True)
def _unit_speed_paths_3d_numba(n_fine: int, stride: int, params):
    n_paths = params.shape[0]
    n = n_fine // stride

    X = np.zeros((n_paths, n + 1, 3), dtype=np.float64)

    two_pi = 2.0 * np.pi
    four_pi = 4.0 * np.pi
    six_pi = 6.0 * np.pi
    eight_pi = 8.0 * np.pi
    half_pi = 0.5 * np.pi

    dt_fine = 1.0 / n_fine

    for b in range(n_paths):
        turns = params[b, 0]
        a_theta = params[b, 1]
        a_phi = params[b, 2]
        phase_theta = params[b, 3]
        phase_phi = params[b, 4]
        phase_theta2 = params[b, 5]
        phase_phi2 = params[b, 6]

        x0 = 0.0
        x1 = 0.0
        x2 = 0.0

        theta_prev = (
            two_pi * turns * 0.0
            + a_theta * np.sin(two_pi * 0.0 + phase_theta)
            + 0.35 * a_theta * np.sin(six_pi * 0.0 + phase_theta2)
        )
        phi_prev = (
            half_pi
            + a_phi * np.sin(four_pi * 0.0 + phase_phi)
            + 0.25 * a_phi * np.sin(eight_pi * 0.0 + phase_phi2)
        )

        sin_phi_prev = np.sin(phi_prev)
        v0_prev = np.cos(theta_prev) * sin_phi_prev
        v1_prev = np.sin(theta_prev) * sin_phi_prev
        v2_prev = np.cos(phi_prev)

        coarse_idx = 0

        for k in range(1, n_fine + 1):
            s = k * dt_fine

            theta = (
                two_pi * turns * s
                + a_theta * np.sin(two_pi * s + phase_theta)
                + 0.35 * a_theta * np.sin(six_pi * s + phase_theta2)
            )
            phi = (
                half_pi
                + a_phi * np.sin(four_pi * s + phase_phi)
                + 0.25 * a_phi * np.sin(eight_pi * s + phase_phi2)
            )

            sin_phi = np.sin(phi)

            v0 = np.cos(theta) * sin_phi
            v1 = np.sin(theta) * sin_phi
            v2 = np.cos(phi)

            x0 += 0.5 * (v0_prev + v0) * dt_fine
            x1 += 0.5 * (v1_prev + v1) * dt_fine
            x2 += 0.5 * (v2_prev + v2) * dt_fine

            v0_prev = v0
            v1_prev = v1
            v2_prev = v2

            if k % stride == 0:
                coarse_idx += 1
                X[b, coarse_idx, 0] = x0
                X[b, coarse_idx, 1] = x1
                X[b, coarse_idx, 2] = x2

    return X