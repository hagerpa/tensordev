"""A repeated public wordwise call makes no implicit host-device transfers."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.dispatch as dispatch
import tensordev._wordwise.fssk as fssk_assembly
from tensordev._wordwise import ordinary
from tensordev.sss import state_update
from tensordev.sss.kernel import FSSK


_X = jnp.asarray(np.random.default_rng(0).normal(scale=0.1, size=(2, 6, 2)))
_KERNEL = FSSK.from_matrix(
    Lambda=jnp.diag(jnp.asarray([0.3, 0.6])), A=jnp.eye(2)[None], b=jnp.asarray([[0.8, -0.2]])
)
_COEF = _KERNEL.coef(0.07, trunc=3)
_FSSK = dict(kernel=_KERNEL, dt=0.07, trunc=3, increment_input=True)
CALLS = {
    "path_signature": lambda execution: td.path_signature(_X, trunc=3, increment_input=True, execution=execution),
    "fssk_state": lambda execution: td.fssk_state(_X, **_FSSK, execution=execution),
    "fssk_state_from_coef": lambda execution: state_update.fssk_state_from_coef(
        _X, coef=_COEF, trunc=3, execution=execution
    ),
    "fssk_vsig": lambda execution: td.fssk_vsig(_X, **_FSSK, execution=execution),
    "fssk_vsig_blocks": lambda execution: td.fssk_vsig(_X, **_FSSK, block_size=3, execution=execution),
}


@pytest.fixture
def interpreted(monkeypatch, fresh_jit_caches):
    """Wordwise calls on the CPU with the kernel in interpret mode."""
    monkeypatch.setattr(dispatch, "_supported_cuda_device", lambda device: True)
    for module, name in (
            (ordinary, "run_ordinary_wordwise"),
            (fssk_assembly, "run_fssk_q1_wordwise"),
            (fssk_assembly, "run_fssk_q1_wordwise_readout"),
    ):
        monkeypatch.setattr(module, name, partial(getattr(module, name), interpret=True))


@pytest.mark.parametrize("name", CALLS)
def test_repeated_wordwise_call_transfers_nothing_implicitly(interpreted, name):
    call = CALLS[name]
    expected = call("jax")
    call("wordwise")
    with jax.transfer_guard("disallow"):
        actual = call("wordwise")
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12)


def test_wordwise_program_runs_on_the_checked_device():
    script = textwrap.dedent(
        """
        from functools import partial

        import jax
        import numpy as np
        import tensordev as td
        import tensordev._wordwise.dispatch as dispatch
        from tensordev._wordwise import ordinary

        checked, other = jax.devices("cpu")
        dispatch._supported_cuda_device = lambda device: device == checked
        ordinary.run_ordinary_wordwise = partial(ordinary.run_ordinary_wordwise, interpret=True)
        x = jax.numpy.asarray(np.random.default_rng(0).normal(scale=0.1, size=(2, 5, 2)), "float32")
        with jax.default_device(other):
            actual = td.path_signature(x, trunc=3, increment_input=True, execution="wordwise")
        expected = td.path_signature(x, trunc=3, increment_input=True, execution="jax")
        assert all(leaf.devices() == {checked} for leaf in jax.tree.leaves(actual))
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-7)
        """
    )
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
