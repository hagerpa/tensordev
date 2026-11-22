"""Lower every wordwise Pallas kernel family for CUDA on a CPU host.

Interpret mode accepts value-level primitives (scatter, slice, gather) that
the Pallas Triton backend cannot lower, so the interpreter tests alone do not
show that a kernel compiles for a GPU.  ``jax.export`` for
``platforms=["cuda"]`` runs the real Triton lowering without a device.
"""

from __future__ import annotations

import contextlib
import importlib.util

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from jax import export as jax_export

import tensordev as td
from families import FAMILIES
from tensordev._wordwise.fssk import (
    PreparedFSSKQ1Call,
    run_fssk_q1_wordwise,
    run_fssk_q1_wordwise_readout,
)
from tensordev._wordwise.layout import build_layout_plan
from tensordev._wordwise.ordinary import run_ordinary_wordwise
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryUnavailableError,
    _load_pallas,
)
from tensordev.development.free import _prepare_free_development_call
from tensordev.sss.state_update import _maximum_order


_TRITON_CUSTOM_CALL = "__gpu$xla.gpu.triton"
# On JAX >= 0.11, CPU-side Triton lowering needs an abstract GPU; any sm_86 device kind works.
_ABSTRACT_GPU_KIND = "NVIDIA A10"
_BATCH = 2
_STEPS = 4
_BLOCK_SIZE = 2
_STATE_DIM = 3

_DTYPES = ("float32", "float64")
_BLOCK_MODES = ("terminal", "accumulated", "independent")
_FSSK_OUTPUTS = ("terminal", "accumulated", "independent", "readout")
# Shared: one coefficient step, singleton coefficients and readout weights,
# and the implicit zero seed.  Batched: per-step and per-batch coefficients,
# per-batch readout weights, and an explicit per-batch seed.
_FSSK_VARIANTS = ("shared", "batched")


def _family(name):
    """Return a core, truncation, and alphabet width with degree-3 words."""
    if name == "total-standard":
        return td.make_core(dims=2, max_trunc=3), 3, 2
    if name == "total-shear":
        return td.make_core(dims=(1, 1), max_trunc=3, coordinates="shear"), 3, 2
    return td.make_core(
        dims=(2, 1),
        max_trunc=(2, 1),
        partially_symmetrized="partial" in name,
        coordinates="shear" if name.endswith("shear") else "standard",
    ), (2, 1), 3


def _kind(family):
    return "quotient" if "partial" in family else "ordered"


def _abstract_gpu():
    if importlib.util.find_spec("jax._src.pallas.triton.gpu_info") is None:
        return contextlib.nullcontext()
    device = jax.sharding.AbstractDevice(
        device_kind=_ABSTRACT_GPU_KIND,
        num_cores=None,
        platform="gpu",
    )
    return jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=device)
    )


def _export_cuda(function, *specs):
    jax.clear_caches()
    with _abstract_gpu():
        return jax_export.export(
            jax.jit(function),
            platforms=["cuda"],
            disabled_checks=[
                jax_export.DisabledSafetyCheck.custom_call(_TRITON_CUSTOM_CALL)
            ],
        )(*specs)


def _cuda_export_unavailable_reason() -> str | None:
    try:
        pl, pltriton = _load_pallas()
    except PallasOrdinaryUnavailableError as exc:
        return str(exc)

    def kernel(x_ref, output_ref):
        output_ref[...] = x_ref[...] + 1

    probe = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=1),
    )
    try:
        _export_cuda(probe, jax.ShapeDtypeStruct((8,), jnp.float32))
    except Exception as exc:  # noqa: BLE001 - any failure means no CUDA export
        return (
            "the installed jaxlib cannot export a Pallas Triton kernel for "
            f"CUDA: {type(exc).__name__}: {exc}"
        )
    return None


@pytest.fixture(scope="module")
def cuda_export():
    reason = _cuda_export_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)


def _walk(operation):
    for region in operation.regions:
        for block in region:
            for inner in block.operations:
                yield inner
                yield from _walk(inner)


def _module_problems(module) -> list[str]:
    """MLIR verifier errors and non-power-of-two tensors (Triton rejects them)."""
    from jax._src.lib.mlir import ir

    problems = []
    try:
        verified = module.operation.verify()
    except Exception as exc:  # noqa: BLE001 - MLIRError carries diagnostics
        problems.append(f"verifier: {exc}")
    else:
        if verified is False:
            problems.append("verifier rejected the module")
    for operation in _walk(module.operation):
        for result in operation.results:
            try:
                shape = tuple(ir.RankedTensorType(result.type).shape)
            except ValueError:
                continue
            if any(extent <= 0 or extent & (extent - 1) for extent in shape):
                problems.append(
                    f"{operation.operation.name} yields {result.type} at "
                    f"{operation.location}"
                )
    return problems


@pytest.fixture
def triton_modules(monkeypatch):
    """Verify each emitted Triton module through a private lowering hook.

    Yields ``None`` when the hook is unavailable; the export itself still
    exercises every lowering rule.
    """
    try:
        from jax._src.pallas.triton import lowering
    except ImportError:
        yield None
        return
    lower_module = getattr(lowering, "lower_jaxpr_to_triton_module", None)
    if lower_module is None:
        yield None
        return

    modules = {}

    def checked_lower_module(jaxpr, *args, **kwargs):
        result = lower_module(jaxpr, *args, **kwargs)
        name = str(jaxpr.debug_info.func_name)
        modules.setdefault(name, []).extend(_module_problems(result.module))
        return result

    monkeypatch.setattr(
        lowering, "lower_jaxpr_to_triton_module", checked_lower_module
    )
    yield modules


def _assert_lowered(modules, expected_kernel: str) -> None:
    if modules is None:
        return
    assert set(modules) == {expected_kernel}
    assert modules[expected_kernel] == []


def _kernel_mode(block_mode: str) -> str:
    return "terminal" if block_mode == "independent" else block_mode


@pytest.mark.parametrize("block_mode", _BLOCK_MODES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_ordinary_kernels_lower_for_cuda(
        cuda_export, triton_modules, family, dtype, block_mode
):
    core, truncation, alphabet_dim = _family(family)
    block_size = None if block_mode == "terminal" else _BLOCK_SIZE

    def signature(increments):
        call = _prepare_free_development_call(
            (increments,),
            trunc=truncation,
            increment_input=True,
            axis=-2,
            block_size=block_size,
            accumulate=block_mode == "accumulated",
            core=core,
        )
        return run_ordinary_wordwise(call)

    exported = _export_cuda(
        signature,
        jax.ShapeDtypeStruct((_BATCH, _STEPS, alphabet_dim), dtype),
    )

    assert exported.platforms == ("cuda",)
    _assert_lowered(triton_modules, f"tensordev_{_kind(family)}_ordinary_{_kernel_mode(block_mode)}")


@pytest.mark.parametrize("variant", _FSSK_VARIANTS)
@pytest.mark.parametrize("output", _FSSK_OUTPUTS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_fssk_kernels_lower_for_cuda(
        cuda_export, triton_modules, family, dtype, output, variant
):
    core, truncation, alphabet_dim = _family(family)
    plan = build_layout_plan(core, truncation)
    order = _maximum_order(truncation)
    batched = variant == "batched"
    coefficient_steps = _STEPS if batched else 1
    coefficient_batch = _BATCH if batched else 1
    block_size = None if output in ("terminal", "readout") else _BLOCK_SIZE
    coefficient_axes = (coefficient_steps, coefficient_batch)
    shapes = [
        (_STEPS, _BATCH, alphabet_dim),
        coefficient_axes + (_STATE_DIM, _STATE_DIM),
        coefficient_axes + (order, _STATE_DIM),
        coefficient_axes + (order - 1, _STATE_DIM, _STATE_DIM),
    ]
    if output == "readout":
        shapes.append((coefficient_batch, _STATE_DIM))
    if batched:
        shapes.extend((_BATCH, 1, 1, _STATE_DIM, block.width) for block in plan.blocks[1:])
    specs = [jax.ShapeDtypeStruct(shape, dtype) for shape in shapes]

    def state(y_time, E_time, psi_time, phi_time, *rest):
        weights = rest[0] if output == "readout" else None
        seed = rest[1:] if output == "readout" else rest
        call = PreparedFSSKQ1Call(
            y_time=y_time,
            E_time=E_time,
            psi_time=psi_time,
            phi_time=phi_time,
            initial_standard=plan.assemble_first_on(seed) if seed else None,
            core=core,
            trunc=truncation,
            batch_shape=(_BATCH,),
            axis=-2,
            block_size=block_size,
            accumulate=output == "accumulated",
            output_starting_state=False,
        )
        if weights is not None:
            return run_fssk_q1_wordwise_readout(call, weights)
        return run_fssk_q1_wordwise(call)

    exported = _export_cuda(state, *specs)

    assert exported.platforms == ("cuda",)
    _assert_lowered(triton_modules, f"tensordev_{_kind(family)}_fssk_q1_{output}")

