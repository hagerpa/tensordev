r"""Regularised incomplete beta function, differentiable in all three arguments.

:func:`betainc` returns the values of :func:`jax.scipy.special.betainc` and
defines their derivatives in ``a``, ``b`` and ``x`` through a
:func:`jax.custom_jvp`.

Method
------
For :math:`y < (p+1)/(p+q+2)`,

.. math::

    I_y(p, q) = q\,w\,F, \qquad
    w = \frac{y^p (1-y)^q\,\Gamma(p+q)}{\Gamma(p+1)\,\Gamma(q+1)}, \qquad
    F = {}_2F_1(p+q, 1; p+1; y),

and :math:`I_x(a, b) = 1 - I_{1-x}(b, a)` covers :math:`x > (a+1)/(a+b+2)`.
:math:`F` is the continued fraction DLMF 8.17.22-24

.. math::

    F = \cfrac{1}{1 + \cfrac{d_1}{1 + \cfrac{d_2}{1 + \cdots}}}, \qquad
    d_{2m} = \frac{m(q-m)\,y}{(p+2m-1)(p+2m)}, \qquad
    d_{2m+1} = -\frac{(p+m)(p+q+m)\,y}{(p+2m)(p+2m+1)},

whose convergents :func:`_hyp2f1_lentz` evaluates together with their
``p``- and ``q``-tangents, as in Boik & Robison-Cox (1998), *Derivatives of
the incomplete beta function*, J. Stat. Softw. 3(1).  Near
:math:`y = (p+1)/(p+q+2)` this takes :math:`O(\sqrt{\max(p, q)})` steps.
With :math:`\Delta\psi(z, e) = \psi(z+e) - \psi(z+1)`,

.. math::

    \partial_p I_y(p, q) = q\,w\,\bigl(F\,(\log y + \Delta\psi(p, q))
        + \partial_p F\bigr), \qquad
    \partial_q I_y(p, q) = w\,\bigl(F\,(1 + q\,(\log(1-y) + \Delta\psi(q, p)))
        + q\,\partial_q F\bigr),

which stay finite at :math:`p = 0` or :math:`q = 0`.  For
:math:`\min(z+e, z+1) \ge 10`, :math:`\Delta\psi` is evaluated without
cancellation as

.. math::

    \Delta\psi(z, e) = \log\Bigl(1 + \frac{e-1}{z+1}\Bigr)
        + \frac{e-1}{2(z+e)(z+1)}
        - \sum_{k=1}^{7} \frac{B_{2k}}{2k}\bigl((z+e)^{-2k} - (z+1)^{-2k}\bigr).

The derivative with respect to ``x`` is the beta density
:math:`x^{a-1}(1-x)^{b-1}/B(a, b)`, evaluated in log space.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jax.scipy.special import betainc as _jax_betainc
from jax.scipy.special import digamma, gammaln
from jax.typing import ArrayLike

__all__ = ["betainc"]

Array = jax.Array

# Lentz steps (two continued-fraction levels each); matches the level budget
# of jax.scipy.special.betainc.
_MAX_STEPS = {jnp.dtype(jnp.float64): 300, jnp.dtype(jnp.float32): 100}

# psi(z) ~ log z - 1/(2z) - sum_k c_k z^(-2k), c_k = B_{2k} / (2k).
_PSI_ASYMPTOTIC = (1 / 12, -1 / 120, 1 / 252, -1 / 240, 1 / 132, -691 / 32760, 1 / 12)
_PSI_ASYMPTOTIC_MIN = 10.0


def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array:
    r"""Regularised incomplete beta function :math:`I_x(a, b)`.

    .. math::

        I_x(a, b) = \frac{1}{B(a, b)} \int_0^x t^{a-1} (1-t)^{b-1}\,\mathrm{d}t.

    Values are those of :func:`jax.scipy.special.betainc`.  The function is
    differentiable in ``a``, ``b`` and ``x`` to any order and in any
    composition of forward and reverse mode (``jax.grad``, ``jax.jacfwd``,
    ``jax.jacrev``, ``jax.hessian``, ``jax.grad(jax.grad(...))``), under
    ``jit`` and ``vmap``.  Where the arguments vary along a ``lax.scan``,
    derivatives of a reverse-mode derivative (``jax.hessian``,
    ``jax.grad(jax.grad(...))``) raise; ``jax.jacrev(jax.jacfwd(...))``
    works there.

    Derivatives with respect to ``a`` and ``b`` come from the continued
    fraction for :math:`I` differentiated in forward mode (see the module
    docstring); each further order nests one more forward-mode pass.  In
    float64 the relative error of the first derivatives is about
    :math:`10^{-12}`, at most :math:`10^{-11}`, for
    :math:`a \in [10^{-3}, 120]`, :math:`b \in [10^{-3}, 30]` and
    :math:`x \in [0, 1]`, and that of the second derivatives about
    :math:`10^{-10}`.  In float32 it is typically :math:`10^{-6}`, and up to
    about :math:`10^{-3}` where a tiny shape parameter makes :math:`I` nearly
    0 or 1.

    The domain is :math:`a, b \ge 0` with :math:`a + b > 0` and
    :math:`0 \le x \le 1`.  At :math:`x \in \{0, 1\}` all derivatives with
    respect to ``a`` and ``b`` are zero, and the ``x``-derivative is the
    one-sided limit of the density (``inf`` when :math:`a < 1` at
    :math:`x = 0` or :math:`b < 1` at :math:`x = 1`); mixed or higher
    ``x``-derivatives at these two points are returned as zero.  An argument
    that is not being differentiated never contributes to a derivative, so an
    infinite density at an end point does not leak into ``a`` or ``b``
    gradients.  Outside the domain values and derivatives are NaN, except for
    the limits :func:`jax.scipy.special.betainc` itself defines.

    Parameters
    ----------
    a, b : ArrayLike
        Real shape parameters, :math:`a, b \ge 0`.
    x : ArrayLike
        Real argument, :math:`0 \le x \le 1`.

    Returns
    -------
    Array
        :math:`I_x(a, b)` with the broadcast shape of the arguments and their
        promoted floating dtype.
    """
    a, b, x = (jnp.asarray(v) for v in (a, b, x))
    dtype = jnp.result_type(a, b, x)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError("betainc does not support complex-valued inputs.")
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
    a, b, x = jnp.broadcast_arrays(*(v.astype(dtype) for v in (a, b, x)))
    return _ab_partials(0)(a, b, x)[0]


@functools.lru_cache(maxsize=None)
def _ab_partials(order: int):
    r"""Custom-JVP function of ``(a, b, x)`` with all ``order``-th ``(a, b)``-partials.

    Entry ``i`` of the leading axis, of length ``k + 1`` for ``k = order``, is
    :math:`\partial_a^{k-i}\partial_b^i I_x(a, b)`; its tangent is
    :math:`\partial_a^{k+1-i}\partial_b^i I\,\dot a
    + \partial_a^{k-i}\partial_b^{i+1} I\,\dot b
    + \partial_a^{k-i}\partial_b^i\partial_x I\,\dot x`.
    """
    impl = _incbeta_partials_impl(order)
    density = _density_partials_impl(order)

    @jax.custom_jvp
    def partials(a, b, x):
        return impl(a, b, x)

    def partials_jvp(primals, tangents):
        a, b, x = primals
        da, db, dx = tangents
        out = partials(a, b, x)
        terms = []
        if not (_is_zero(da) and _is_zero(db)):
            higher = _ab_partials(order + 1)(a, b, x)
            if not _is_zero(da):
                terms.append(higher[:-1] * da)
            if not _is_zero(db):
                terms.append(higher[1:] * db)
        if not _is_zero(dx):
            terms.append(density(a, b, x) * dx)
        return out, sum(terms[1:], terms[0]) if terms else jnp.zeros_like(out)

    partials.defjvp(partials_jvp, symbolic_zeros=True)
    return partials


def _is_zero(tangent) -> bool:
    return isinstance(tangent, SymbolicZero)


def _raise_order(fun):
    """Map a function returning all order-k (a, b)-partials to order k+1."""

    def raised(a, b, x):
        one = jnp.ones_like(a)
        _, d_a = jax.jvp(lambda a_: fun(a_, b, x), (a,), (one,))
        _, d_b = jax.jvp(lambda b_: fun(a, b_, x), (b,), (one,))
        return jnp.concatenate([d_a, d_b[-1:]])

    return raised


def _incbeta_partials_impl(order: int):
    if order == 0:
        return lambda a, b, x: _jax_betainc(a, b, x)[None]
    if order == 1:
        return lambda a, b, x: jnp.stack(_incbeta_grad_ab(a, b, x))
    return _raise_order(_incbeta_partials_impl(order - 1))


def _density_partials_impl(order: int):
    if order == 0:
        return lambda a, b, x: _beta_density(a, b, x)[None]
    return _raise_order(_density_partials_impl(order - 1))


def _beta_density(a: Array, b: Array, x: Array) -> Array:
    r""":math:`\partial_x I_x(a, b) = x^{a-1}(1-x)^{b-1} / B(a, b)`."""
    interior = (x > 0) & (x < 1)
    x_safe = jnp.where(interior, x, 0.5)
    log_beta = gammaln(a) + gammaln(b) - gammaln(a + b)
    inner = jnp.exp(
        (a - 1) * jnp.log(x_safe) + (b - 1) * jnp.log1p(-x_safe) - log_beta
    )
    # end points: the power of the vanishing factor decides 0, finite or inf
    power = jnp.where(x == 0, a - 1, b - 1)
    end = jnp.where(
        power > 0, 0.0, jnp.where(power == 0, jnp.exp(-log_beta), jnp.inf)
    ).astype(x.dtype)
    return jnp.where(_in_domain(a, b, x), jnp.where(interior, inner, end), jnp.nan)


def _in_domain(a: Array, b: Array, x: Array) -> Array:
    finite = (a < jnp.inf) & (b < jnp.inf)
    return (a >= 0) & (b >= 0) & (a + b > 0) & finite & (x >= 0) & (x <= 1)


def _incbeta_grad_ab(a: Array, b: Array, x: Array) -> tuple[Array, Array]:
    r"""(:math:`\partial_a I_x(a, b)`, :math:`\partial_b I_x(a, b)`)."""
    valid = _in_domain(a, b, x)
    interior = valid & (x > 0) & (x < 1)
    # a = b = 1, x = 1/2 where not interior: finite, and the continued
    # fraction converges there in about ten steps
    a = jnp.where(interior, a, 1.0)
    b = jnp.where(interior, b, 1.0)
    x = jnp.where(interior, x, 0.5)

    swap = x > (a + 1) / (a + b + 2)
    p = jnp.where(swap, b, a)
    q = jnp.where(swap, a, b)
    y = jnp.where(swap, 1 - x, x)

    f, f_p, f_q = _hyp2f1_lentz(p, q, y)
    log_y = jnp.log(y)
    log1m_y = jnp.log1p(-y)
    # J = I_y(p, q) = q * w * f
    w = jnp.exp(
        p * log_y + q * log1m_y + gammaln(p + q) - gammaln(p + 1) - gammaln(q + 1)
    )
    j_p = q * w * (f * (log_y + _digamma_shift_diff(p, q)) + f_p)
    j_q = w * (f * (1 + q * (log1m_y + _digamma_shift_diff(q, p))) + q * f_q)

    i_a = jnp.where(swap, -j_q, j_p)
    i_b = jnp.where(swap, -j_p, j_q)
    # all (a, b)-derivatives vanish where I is identically 0 or 1
    end = jnp.where(valid, 0.0, jnp.nan).astype(x.dtype)
    return jnp.where(interior, i_a, end), jnp.where(interior, i_b, end)


def _hyp2f1_lentz(p: Array, q: Array, y: Array) -> tuple[Array, Array, Array]:
    r""":math:`F = {}_2F_1(p+q, 1; p+1; y)` and its ``p``- and ``q``-partials.

    With :math:`d_j` of the module docstring, the convergents are
    :math:`F_1 = D_1 = 1/(1 + d_1)`, :math:`C_1 = 1` and, for :math:`j \ge 2`,

    .. math::
        D_j = \frac{1}{1 + d_j D_{j-1}}, \qquad
        C_j = 1 + \frac{d_j}{C_{j-1}}, \qquad
        F_j = F_{j-1}\, C_j D_j,

    where :math:`1/D_j` and :math:`C_j` below the smallest normal number
    :math:`\varepsilon_0` in absolute value are set to :math:`\varepsilon_0`.
    The ``p``- and ``q``-tangents follow by :func:`jax.jvp`.  Step ``m`` adds
    levels :math:`2m` and :math:`2m+1`.  An entry has converged once
    :math:`|F_{2m+1}/F_{2m-1} - 1| \le \epsilon` and each tangent
    :math:`\partial F` changes by at most :math:`\epsilon(|\partial F| + |F|)`,
    with :math:`\epsilon` the machine epsilon; the loop ends when all entries
    have converged or after ``_MAX_STEPS`` steps.  Accurate for
    :math:`y < (p+1)/(p+q+2)`.
    """
    dtype = y.dtype
    eps = jnp.finfo(dtype).eps
    tiny = jnp.finfo(dtype).tiny
    max_steps = _MAX_STEPS.get(jnp.dtype(dtype), 300)
    one = jnp.ones_like(y)
    zero = jnp.zeros_like(y)

    def first(p, q):
        d = 1 / (1 - (p + q) * y / (p + 1))
        return one, d, d

    def level(c, d, numerator):
        d = 1 + numerator * d
        d = jnp.where(jnp.abs(d) < tiny, tiny, d)
        c = 1 + numerator / c
        c = jnp.where(jnp.abs(c) < tiny, tiny, c)
        d = 1 / d
        return c, d, c * d

    def step(state, p, q, m):
        c, d, h = state
        c, d, r_even = level(c, d, m * (q - m) * y / ((p + 2 * m - 1) * (p + 2 * m)))
        c, d, r_odd = level(
            c, d, -(p + m) * (p + q + m) * y / ((p + 2 * m) * (p + 2 * m + 1))
        )
        ratio = r_even * r_odd
        return (c, d, h * ratio), ratio

    def body(carry):
        m, state, t_p, t_q, done = carry
        advance = lambda s, p_, q_: step(s, p_, q_, m)
        (new, ratio), (new_t_p, _) = jax.jvp(advance, (state, p, q), (t_p, one, zero))
        _, (new_t_q, _) = jax.jvp(advance, (state, p, q), (t_q, zero, one))
        h, h_p, h_q = new[2], new_t_p[2], new_t_q[2]
        converged = (
            (jnp.abs(ratio - 1) <= eps)
            & (jnp.abs(h_p - t_p[2]) <= eps * (jnp.abs(h_p) + jnp.abs(h)))
            & (jnp.abs(h_q - t_q[2]) <= eps * (jnp.abs(h_q) + jnp.abs(h)))
        )
        return m + 1, new, new_t_p, new_t_q, done | converged

    def cond(carry):
        return (carry[0] <= max_steps) & ~jnp.all(carry[4])

    state, t_p = jax.jvp(first, (p, q), (one, zero))
    _, t_q = jax.jvp(first, (p, q), (zero, one))
    carry = (jnp.ones((), dtype), state, t_p, t_q, jnp.zeros(y.shape, bool))
    _, state, t_p, t_q, _ = jax.lax.while_loop(cond, body, carry)
    return state[2], t_p[2], t_q[2]


def _digamma_shift_diff(z: Array, e: Array) -> Array:
    r""":math:`\psi(z+e) - \psi(z+1)`, without cancellation for large ``z``."""
    u = z + e
    v = z + 1
    large = jnp.minimum(u, v) >= _PSI_ASYMPTOTIC_MIN
    u_l = jnp.where(large, u, _PSI_ASYMPTOTIC_MIN)
    v_l = jnp.where(large, v, _PSI_ASYMPTOTIC_MIN)
    shift = jnp.where(large, e - 1, 0.0)  # u - v without the rounding of u and v

    def tail(t):
        t2 = 1 / (t * t)
        acc = jnp.zeros_like(t)
        for c in reversed(_PSI_ASYMPTOTIC):
            acc = (acc + c) * t2
        return acc

    asymptotic = (
        jnp.log1p(shift / v_l) + shift / (2 * u_l * v_l) - (tail(u_l) - tail(v_l))
    )
    return jnp.where(large, asymptotic, digamma(u) - digamma(v))
