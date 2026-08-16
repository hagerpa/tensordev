"""
Sharding must agree with direct evaluation for every batch size.

The interesting cases are the small ones. `pmap_batch` pads the batch up to a
multiple of the device count, and the padding used to be taken as `x[:pad]`,
which supplies only `min(pad, batch)` rows — so any batch smaller than its own
pad (a single item on four devices, say) produced a short array and a reshape
error. Batches of one arise naturally: a law-law kernel block has exactly one
pair.

Device count is a process-wide property fixed before JAX initialises, so the
multi-device cases run in a subprocess with `XLA_FLAGS` set. The in-process
test covers whatever devices this interpreter already has.
"""
from jax import config

config.update("jax_enable_x64", True)

import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest

from tensordev.kernel.parallel import pmap_batch

BATCHES = (1, 2, 3, 4, 5, 7, 8)

_SUBPROCESS = textwrap.dedent(
    """
    import jax, jax.numpy as jnp
    from tensordev.kernel.parallel import pmap_batch

    assert jax.device_count() == {devices}, jax.device_count()

    for batch in {batches}:
        x = jnp.arange(batch * 3, dtype=jnp.float32).reshape(batch, 3)
        y = jnp.arange(batch * 3, dtype=jnp.float32).reshape(batch, 3) * 2.0

        sharded = pmap_batch(lambda a, b: (a + b).sum(axis=-1), x, y,
                             num_devices={devices})
        direct = (x + y).sum(axis=-1)

        assert sharded.shape == direct.shape, (batch, sharded.shape, direct.shape)
        assert jnp.allclose(sharded, direct), batch

    # a pytree of several leaves, as free_kernel passes multi-level inputs
    for batch in {batches}:
        left = (jnp.ones((batch, 2)), jnp.ones((batch, 4)))
        right = (jnp.full((batch, 2), 3.0), jnp.full((batch, 4), 5.0))
        sharded = pmap_batch(
            lambda a, b: a[0].sum(axis=-1) + b[1].sum(axis=-1),
            left, right, num_devices={devices},
        )
        assert sharded.shape == (batch,), (batch, sharded.shape)
        assert jnp.allclose(sharded, 2.0 + 20.0), batch

    # leaves disagreeing on axis 0 are reported as such
    try:
        pmap_batch(lambda a, b: a.sum(axis=-1),
                   jnp.zeros((4, 3)), jnp.zeros((3, 3)), num_devices={devices})
    except ValueError as exc:
        assert "agree on axis 0" in str(exc), exc
    else:
        raise AssertionError("mismatched leaves must raise")

    print("ok")
    """
)


def _run_with_devices(devices: int) -> None:
    script = _SUBPROCESS.format(devices=devices, batches=BATCHES)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "JAX_LOGGING_LEVEL": "ERROR",
        },
    )
    assert completed.returncode == 0, (
        f"num_devices={devices} failed:\n{completed.stdout}\n{completed.stderr}"
    )
    assert "ok" in completed.stdout


@pytest.mark.parametrize("devices", [2, 4, 8])
def test_sharding_matches_direct_evaluation(devices: int):
    """Every batch size, including batches smaller than the device count."""
    _run_with_devices(devices)


@pytest.mark.parametrize("batch", BATCHES)
def test_sharding_on_the_available_devices(batch: int):
    """Same contract on however many devices this interpreter exposes."""
    devices = jax.device_count()
    x = np.arange(batch * 3, dtype=np.float64).reshape(batch, 3)

    sharded = pmap_batch(lambda a: a.sum(axis=-1), x, num_devices=devices)
    np.testing.assert_allclose(np.asarray(sharded), x.sum(axis=-1))


@pytest.mark.skipif(jax.device_count() < 2, reason="needs at least two devices")
def test_mismatched_leaves_are_reported():
    """Below two devices `pmap_batch` returns before sharding, so nothing to check."""
    with pytest.raises(ValueError, match="agree on axis 0"):
        pmap_batch(lambda a, b: a.sum(axis=-1), np.zeros((4, 3)), np.zeros((3, 3)),
                   num_devices=jax.device_count())
