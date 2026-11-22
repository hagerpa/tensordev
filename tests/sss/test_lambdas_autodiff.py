"""Derivatives of exp(-h Lambda), phi_1(-h Lambda) and phi_2(-h Lambda) in h and in Lambda."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from phi_reference import KINDS, phi
from tensordev.sss import DenseLambda, JordanLambda

SOLVES = {False: "solve_shifted", True: "solve_shifted_transpose"}
TIMES = jnp.array([0.123456789012345, 0.987654321098765], dtype=jnp.float64)


def _rhs(size):
    return jnp.arange(1.0, size * 2 + 1).reshape(size, 2) / 7


def _scalar_operator(rate, kind):
    if kind == "dense":
        return DenseLambda(jnp.reshape(rate, (1, 1)))
    return JordanLambda(real_rates=jnp.reshape(rate, (1,)), real_sizes=(1,))


@pytest.mark.parametrize("kind", ["dense", "jordan"])
@pytest.mark.parametrize("order", [1, 2])
def test_scalar_phi_values_and_derivatives_at_zero_tiny_and_large_rates(kind, order):
    rates = jnp.array([0.0, 1e-12, 0.7, 100.0, 1e4])
    dt = 0.2

    def scalar(rate):
        return getattr(_scalar_operator(rate, kind), f"phi{order}")(dt)[0, 0]

    value, reverse = jax.jit(jax.vmap(jax.value_and_grad(scalar)))(rates)
    _, forward = jax.jit(
        lambda x: jax.jvp(jax.vmap(scalar), (x,), (jnp.ones_like(x),))
    )(rates)
    expected = [
        phi([[float(rate)]], dt, order, matrix_direction=[[1.0]])
        for rate in rates
    ]
    np.testing.assert_allclose(
        value, [v.item() for v, _ in expected], rtol=2e-10, atol=1e-13
    )
    np.testing.assert_allclose(
        reverse, [g.item() for _, g in expected], rtol=2e-9, atol=1e-13
    )
    np.testing.assert_allclose(forward, reverse, rtol=2e-10, atol=1e-13)
    second = jax.jit(jax.grad(jax.grad(scalar)))(jnp.array(0.0, dtype=jnp.float64))
    expected_second = dt**2 / (3 if order == 1 else 12)
    np.testing.assert_allclose(second, expected_second, rtol=2e-10, atol=1e-13)


@pytest.mark.parametrize("method,order", KINDS)
def test_dense_constructor_and_matrix_time_derivatives_match_frechet(method, order):
    matrices = jnp.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.4, -1.7], [0.0, 0.4]],
            [[0.3, 1.2], [-0.2, 0.8]],
        ]
    )
    weight = jnp.array([[0.7, -0.2], [1.3, 0.4]])
    direction = jnp.array([[0.2, -0.4], [0.1, 0.3]])
    dt, dt_direction = 0.3, 0.2

    def loss(matrix, duration):
        return jnp.sum(weight * getattr(DenseLambda(matrix), method)(duration))

    evaluate = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    directional = jax.jit(
        lambda matrix, duration: jax.jvp(
            loss, (matrix, duration), (direction, jnp.asarray(dt_direction))
        )[1]
    )
    for matrix in matrices:
        value, (gradient, time_gradient) = evaluate(matrix, jnp.asarray(dt))
        reference, derivative = phi(matrix, dt, order, direction)
        _, time_derivative = phi(matrix, dt, order, dt_direction=1.0)
        np.testing.assert_allclose(value, np.sum(weight * reference), atol=2e-11)
        np.testing.assert_allclose(
            jnp.sum(gradient * direction), np.sum(weight * derivative), atol=2e-11
        )
        np.testing.assert_allclose(
            time_gradient, np.sum(weight * time_derivative), atol=2e-11
        )
        np.testing.assert_allclose(
            directional(matrix, jnp.asarray(dt)),
            np.sum(weight * (derivative + dt_direction * time_derivative)),
            atol=2e-11,
        )


def _mixed_jordan(parameters):
    return JordanLambda(
        real_rates=parameters[:1],
        real_sizes=(2,),
        osc_decays=parameters[1:2],
        osc_freqs=parameters[2:3],
        osc_sizes=(2,),
    )


def _mixed_matrix(parameters):
    rate, decay, frequency = np.asarray(parameters)
    matrix = np.zeros((6, 6))
    matrix[:2, :2] = [[rate, -1.0], [0.0, rate]]
    block = [[decay, -frequency], [frequency, decay]]
    matrix[2:4, 2:4] = block
    matrix[4:6, 4:6] = block
    matrix[2:4, 4:6] = -np.eye(2)
    return matrix


@pytest.mark.parametrize("method,order", KINDS)
def test_jordan_constructor_and_chain_derivatives_match_frechet(method, order):
    parameters = jnp.array([0.0, 0.0, 1.3])
    direction = jnp.array([0.2, -0.1, 0.3])
    weight = jnp.arange(36.0).reshape(6, 6) / 37
    dt = 0.3

    def loss(p, duration):
        return jnp.sum(weight * getattr(_mixed_jordan(p), method)(duration))

    value, (gradient, time_gradient) = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1))
    )(parameters, jnp.asarray(dt))
    matrix = _mixed_matrix(parameters)
    matrix_direction = _mixed_matrix(direction) - _mixed_matrix(np.zeros(3))
    expected, derivative = phi(matrix, dt, order, matrix_direction)
    _, time_derivative = phi(matrix, dt, order, dt_direction=1.0)
    np.testing.assert_allclose(value, np.sum(weight * expected), atol=3e-11)
    np.testing.assert_allclose(
        jnp.dot(gradient, direction), np.sum(weight * derivative), atol=3e-11
    )
    np.testing.assert_allclose(
        time_gradient, np.sum(weight * time_derivative), atol=3e-11
    )
    _, tangent = jax.jvp(
        loss, (parameters, jnp.asarray(dt)), (direction, jnp.asarray(0.0))
    )
    np.testing.assert_allclose(tangent, jnp.dot(gradient, direction), atol=3e-11)


def test_dense_raw_matrix_is_the_only_parameter_and_updates_all_operator_methods():
    matrix = jnp.array([[0.5, -0.4], [0.2, 0.7]])
    operator = DenseLambda(matrix)
    assert len(jax.tree.leaves(operator)) == 1
    changed = jax.tree.map(lambda x: 1.2 * x + jnp.eye(2) * 0.1, operator)
    expected_matrix = 1.2 * np.asarray(matrix) + np.eye(2) * 0.1
    rhs = jnp.array([[1.0, -0.3], [0.4, 0.8]])
    dt, zeta = 0.4, 0.8 + 0.6j
    for method, order in KINDS:
        expected = phi(expected_matrix, dt, order)
        np.testing.assert_allclose(getattr(changed, method)(dt), expected, atol=2e-11)
        np.testing.assert_allclose(
            getattr(changed, f"{method}_multiply_left")(dt, rhs),
            expected @ rhs,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            getattr(changed, f"{method}_multiply_right")(dt, rhs),
            rhs @ expected.T,
            atol=2e-11,
        )
    for transpose in (False, True):
        system = zeta * np.eye(2) + dt * (
            expected_matrix.T if transpose else expected_matrix
        )
        np.testing.assert_allclose(
            getattr(changed, SOLVES[transpose])(zeta, dt, rhs),
            np.linalg.solve(system, rhs),
            atol=2e-11,
        )
    gradient = jax.jit(jax.grad(lambda op: op.phi2(dt).sum()))(operator)
    _, expected_derivative = phi(matrix, dt, 2, np.ones((2, 2)))
    np.testing.assert_allclose(
        gradient.matrix.sum(), expected_derivative.sum(), atol=2e-11
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_dense_shifted_solve_matrix_and_time_gradients(transpose):
    matrix = jnp.array([[0.5, -0.4], [0.2, 0.7]])
    direction = jnp.array([[0.1, 0.3], [-0.2, 0.4]])
    rhs = jnp.array([[1.0 + 0.2j, -0.3j], [0.4, 0.8 - 0.1j]])
    weight = jnp.array([[0.7, -0.2], [0.3, 0.8]])
    dt, zeta = 0.4, 0.8 + 0.6j

    def loss(parameter, duration):
        solution = getattr(DenseLambda(parameter), SOLVES[transpose])(zeta, duration, rhs)
        return jnp.real(jnp.sum(weight * solution))

    value, (gradient, time_gradient) = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1))
    )(matrix, jnp.asarray(dt))
    mat = np.asarray(matrix.T if transpose else matrix)
    tangent = np.asarray(direction.T if transpose else direction)
    system = zeta * np.eye(2) + dt * mat
    solution = np.linalg.solve(system, rhs)
    derivative = -np.linalg.solve(system, dt * tangent @ solution)
    time_derivative = -np.linalg.solve(system, mat @ solution)
    np.testing.assert_allclose(value, np.real(np.sum(weight * solution)), atol=2e-11)
    np.testing.assert_allclose(
        jnp.sum(gradient * direction), np.real(np.sum(weight * derivative)), atol=2e-11
    )
    np.testing.assert_allclose(
        time_gradient, np.real(np.sum(weight * time_derivative)), atol=2e-11
    )


@pytest.mark.parametrize("kind", ["dense", "jordan"])
def test_precompute_inside_loss_preserves_parameter_and_time_gradients(kind):
    durations = jnp.array([0.0, 0.15, 0.4])
    if kind == "dense":
        operator = DenseLambda(jnp.array([[0.4, -0.8], [0.0, 0.4]]))
    else:
        operator = JordanLambda(real_rates=jnp.array([0.0]), real_sizes=(2,))
    rhs = jnp.array([[0.7, -0.3], [0.2, 0.8]])
    weight = jnp.array([[0.4, 1.2], [-0.3, 0.6]])

    def loss(op, times, prepared):
        result = jnp.asarray(0.0)
        for method, _ in KINDS:
            if prepared:
                coefficients = getattr(op, f"precompute_{method}")(times)
                for k in range(times.shape[0]):
                    action = getattr(coefficients[k], f"{method}_multiply_left")
                    result += jnp.sum(weight * action(times[k], rhs))
            else:
                values = getattr(op, method)(times)
                result += jnp.sum(weight * (values @ rhs))
        return result

    evaluate = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)), static_argnums=(2,))
    prepared_value, prepared_grad = evaluate(operator, durations, True)
    direct_value, direct_grad = evaluate(operator, durations, False)
    np.testing.assert_allclose(prepared_value, direct_value, atol=3e-11)
    for got, expected in zip(
        jax.tree.leaves(prepared_grad), jax.tree.leaves(direct_grad)
    ):
        np.testing.assert_allclose(got, expected, rtol=2e-10, atol=3e-11)
        assert np.all(np.isfinite(got))
    check_grads(lambda times: loss(operator, times, True), (durations,), order=1, modes=("rev",),
                eps=1e-5, atol=2e-9, rtol=0)


@pytest.mark.parametrize("kind", ["dense", "jordan"])
def test_optimizer_reprepares_operators_from_current_parameters(kind):
    operator = _scalar_operator(jnp.array(0.4, dtype=jnp.float64), kind)
    dt = 0.7
    target = phi([[1.3]], dt, 1)[0, 0]

    def loss(op):
        prepared = op.precompute_phi1(dt)
        value = prepared.phi1_multiply_left(dt, jnp.ones((1, 1)))[0, 0]
        return (value - target) ** 2

    evaluate = jax.jit(jax.value_and_grad(loss))
    previous = float(loss(operator))
    for _ in range(5):
        value, gradient = evaluate(operator)
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))
        operator = jax.tree.map(lambda p, g: p - 0.5 * g, operator, gradient)
        rate = operator.matrix[0, 0] if kind == "dense" else operator.real_rates[0]
        assert float(rate) > 0
        for method, order in KINDS:
            prepared = getattr(operator, f"precompute_{method}")(dt)
            actual = getattr(prepared, f"{method}_multiply_left")(dt, jnp.ones((1, 1)))
            np.testing.assert_allclose(
                actual, phi([[rate]], dt, order), atol=2e-11
            )
        current = float(loss(operator))
        assert current < previous
        previous = current


def _float32_operator(kind):
    """Use exactly representable parameters so only time precision is tested."""
    if kind == "dense":
        matrix = np.array([[0.5, -0.25], [0.125, 0.75]])
        operator = DenseLambda(jnp.asarray(matrix, dtype=jnp.float32))
    else:
        matrix = np.array([[0.5, 0.0, 0.0], [0.0, 0.25, -1.25], [0.0, 1.25, 0.25]])
        operator = JordanLambda(
            real_rates=jnp.array([0.5], dtype=jnp.float32),
            real_sizes=(1,),
            osc_decays=jnp.array([0.25], dtype=jnp.float32),
            osc_freqs=jnp.array([1.25], dtype=jnp.float32),
            osc_sizes=(1,),
        )
    return operator, matrix


@pytest.mark.parametrize("kind", ["dense", "jordan"])
@pytest.mark.parametrize("method,order", KINDS)
def test_requested_float64_preserves_time_precision_with_float32_parameters(
    kind, method, order
):
    operator, matrix = _float32_operator(kind)
    rhs = _rhs(matrix.shape[0])
    lhs = rhs.T
    direct = getattr(operator, method)
    left_action = getattr(operator, f"{method}_multiply_left")
    right_action = getattr(operator, f"{method}_multiply_right")
    references = phi(matrix, TIMES, order)

    for durations, expected in ((TIMES[0], references[0]), (TIMES, references)):
        result = direct(durations, dtype=jnp.float64)
        assert result.dtype == jnp.float64
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-13)
        left = left_action(durations, rhs, dtype=jnp.float64)
        right = right_action(durations, lhs, dtype=jnp.float64)
        assert left.dtype == right.dtype == jnp.float64
        np.testing.assert_allclose(left, expected @ rhs, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(
            right, lhs @ np.swapaxes(expected, -1, -2), rtol=2e-12, atol=2e-13
        )

    prepared = getattr(operator, f"precompute_{method}")(TIMES, dtype=jnp.float64)
    for k, expected in enumerate(references):
        left = getattr(prepared[k], f"{method}_multiply_left")(
            TIMES[k], rhs, dtype=jnp.float64
        )
        right = getattr(prepared[k], f"{method}_multiply_right")(
            TIMES[k], lhs, dtype=jnp.float64
        )
        np.testing.assert_allclose(left, expected @ rhs, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(right, lhs @ expected.T, rtol=2e-12, atol=2e-13)

    def outputs(duration):
        prepared = getattr(operator, f"precompute_{method}")(duration, dtype=jnp.float64)
        return jnp.stack(
            (
                direct(duration, dtype=jnp.float64).sum(),
                left_action(duration, rhs, dtype=jnp.float64).sum(),
                getattr(prepared, f"{method}_multiply_left")(duration, rhs, dtype=jnp.float64).sum(),
            )
        )

    time_gradients = jax.jit(jax.jacrev(outputs))(TIMES[0])
    _, derivative = phi(matrix, float(TIMES[0]), order, dt_direction=1.0)
    np.testing.assert_allclose(
        time_gradients, [derivative.sum()] + 2 * [(derivative @ rhs).sum()], rtol=2e-11, atol=2e-12
    )


@pytest.mark.parametrize("kind", ["dense", "jordan"])
@pytest.mark.parametrize("transpose", [False, True])
def test_requested_float64_preserves_shifted_solve_time_precision(kind, transpose):
    operator, matrix = _float32_operator(kind)
    if transpose:
        matrix = matrix.T
    size = matrix.shape[0]
    rhs = _rhs(size) + 0.25j
    zeta = 0.876543210987654 + 0.345678901234567j
    solve = getattr(operator, SOLVES[transpose])
    references = np.stack(
        [np.linalg.solve(zeta * np.eye(size) + float(h) * matrix, rhs) for h in TIMES]
    )
    for durations, expected in ((TIMES[0], references[0]), (TIMES, references)):
        result = solve(zeta, durations, rhs, dtype=jnp.float64)
        assert result.dtype == jnp.complex128
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-13)

    def loss(duration):
        return jnp.real(solve(zeta, duration, rhs, dtype=jnp.float64).sum())

    gradient = jax.jit(jax.grad(loss))(TIMES[0])
    system = zeta * np.eye(size) + float(TIMES[0]) * matrix
    derivative = -np.linalg.solve(system, matrix @ references[0])
    np.testing.assert_allclose(gradient, derivative.real.sum(), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("kind", ["dense", "jordan"])
def test_mixed_scaling_phi2_gradient_ignores_inactive_overflow(kind):
    r"""Check :math:`\varphi_2(a)` and its derivative when ``exp(a)`` overflows."""
    rate = jnp.array(710.0, dtype=jnp.float64)
    times = jnp.array([-1.0, 1000.0], dtype=jnp.float64)

    def values(parameter):
        return _scalar_operator(parameter, kind).phi2(times)[:, 0, 0]

    actual = jax.jit(values)(rate)
    gradient = jax.jit(jax.grad(lambda parameter: values(parameter).sum()))(rate)
    a = -np.asarray(times) * float(rate)
    # exp(a) / a**2 remains finite although exp(a) itself overflows at a=710.
    normalized_exponential = np.exp(a - 2 * np.log(np.abs(a)))
    expected = normalized_exponential - (1 + a) / a**2
    derivative = -np.asarray(times) * (
        (1 - 2 / a) * normalized_exponential + (1 + 2 / a) / a**2
    )
    assert np.all(np.isfinite(actual))
    assert np.isfinite(gradient)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=0)
    np.testing.assert_allclose(gradient, derivative.sum(), rtol=2e-10, atol=0)
