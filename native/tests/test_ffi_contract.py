from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _register(extension, test_name: str, registration_name: str) -> str:
    state_types = extension.type_registrations()
    jax.devices("cpu")
    try:
        jax.ffi.register_ffi_type(
            "tensordev.ragged_horner_state.v1",
            state_types["tensordev.ragged_horner_state.v1"],
            platform="cpu",
        )
    except ValueError as error:
        if "already registered" not in str(error):
            raise
    capsule = extension.registrations()[registration_name]
    jax.ffi.register_ffi_target(
        test_name,
        capsule,
        platform="cpu",
        api_version=1,
    )
    return test_name


def _minimal_horner_metadata() -> np.ndarray:
    metadata = np.zeros((3, 16), dtype=np.int32)
    metadata[:, 4:6] = -1
    metadata[:, 13] = 1
    metadata[:, 14] = 1
    metadata[:, 15] = 2

    metadata[0, :4] = (0, 1, 1, 1)
    metadata[1, :4] = (1, 1, 1, 1)
    metadata[1, 4] = 0
    metadata[2, :4] = (1, 2, 2, 1)
    metadata[2, 5] = 0
    metadata[2, 6:8] = (2, 1)
    metadata[2, 8:13] = (0, 1, 0, 0, 1)
    return metadata


def test_ragged_horner_dynamic_arity_and_nested_vmap(native_extension):
    target = _register(
        native_extension,
        "tensordev_native_test_horner_f64",
        "tensordev_cpu_sym_horner_f64_v2",
    )
    metadata = jnp.asarray(_minimal_horner_metadata())
    selected = jnp.asarray([0], dtype=jnp.int32)
    collisions = jnp.empty((0,), dtype=jnp.int32)

    def native(blocks, generator):
        specs = (
            jax.ShapeDtypeStruct((1,), generator.dtype),
            jax.ShapeDtypeStruct((1,), generator.dtype),
            jax.ShapeDtypeStruct((2,), generator.dtype),
        )
        return jax.ffi.ffi_call(
            target,
            specs,
            vmap_method="expand_dims",
        )(
            generator,
            *blocks,
            metadata=np.asarray(metadata).reshape(-1),
            selected_edges=np.asarray(selected),
            collision_targets=np.asarray(collisions),
        )

    def expected(blocks, generator):
        scalar, prime, double = blocks
        return (
            scalar,
            prime + scalar * generator[0],
            double + scalar * jnp.stack((generator[2], generator[1]), axis=-1),
        )

    scalar = jnp.arange(6, dtype=jnp.float64).reshape(2, 3, 1) / 11
    prime = jnp.arange(6, dtype=jnp.float64).reshape(2, 3, 1) / 13
    double = jnp.arange(12, dtype=jnp.float64).reshape(2, 3, 2) / 17
    generator = jnp.arange(9, dtype=jnp.float64).reshape(3, 3) / 19
    blocks = (scalar, prime, double)
    actual = jax.jit(
        jax.vmap(
            jax.vmap(native, in_axes=(0, 0)),
            in_axes=(0, None),
        )
    )(blocks, generator)
    reference = jax.vmap(
        jax.vmap(expected, in_axes=(0, 0)),
        in_axes=(0, None),
    )(blocks, generator)
    for left, right in zip(actual, reference):
        np.testing.assert_allclose(left, right, atol=2e-14, rtol=2e-14)
