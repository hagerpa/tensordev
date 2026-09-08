# GPU masterplan: ordinary signatures and scalar FSSK signatures

## Status and scope

This document records the implementation and validation plan for a GPU-native
wordwise path through TensorDev. The 0.1.0 source tree contains the shared
planners, portable references, candidate Pallas executors, interpreter tests,
real-GPU tests, and benchmark driver described below. Core-neutral portable
execution is implemented for ordinary and exact scalar-FSSK signatures.

Native execution has not yet been validated or benchmarked on a supported
NVIDIA GPU. No architecture and workload allowlist is therefore installed,
native automatic dispatch is intentionally disabled, and public calls use the
portable JAX implementation on their selected device. The candidate executors
are reachable only through private tests and benchmarks. They must not be
described as a released speedup until the native-dispatch checklist passes.

The programme covers:

- ordinary path signatures;
- finite-state-space Volterra signatures for scalar FSSK kernels (`kernel.q ==
  1`);
- total-degree and bidegree truncation;
- ordered and partially symmetrized layouts wherever those layouts currently
  exist;
- standard and shear coordinates.

The following six core families are therefore in scope.

| Truncation | Layout | Coordinates |
| --- | --- | --- |
| total degree | ordered | standard |
| total degree | ordered | shear |
| bidegree | ordered | standard |
| bidegree | ordered | shear |
| bidegree | partially symmetrized | standard |
| bidegree | partially symmetrized | shear |

TensorDev deliberately has no total-degree partially symmetrized core.  This
programme does not introduce one.  “All layouts” means all combinations exposed
by `make_core` today, not a new seventh or eighth core family.

The dimension-free `Jax()` core is also in scope.  It is the package's initial
default even though it is not constructed through `make_core`.  For this core,
the ordinary-signature planner resolves the alphabet width from
`x.shape[-1]`; an explicitly configured `core.d`, when present, is validated
against that width.  The normalized active truncation is still required when
the core supplies no default.

The following are explicitly out of scope:

- FSSK kernels with `q > 1`;
- the generic discretized `td.vsig` algorithms (`quadratic`, `fft`, and
  `adams`);
- shuffle products, shear pairings, free developments, and arbitrary algebra
  homomorphisms;
- direct wordwise recurrences in shear coordinates;
- ROCm, TPU, and non-NVIDIA GPU kernels in the first release;
- a public `device`, `backend`, `method`, or `use_gpu` argument;
- a second production GPU implementation kept alongside the selected one.

The generic `td.vsig` path must remain unchanged.  It has discretization and
scheme semantics distinct from the exact state-space recurrence.  The `q == 1`
work below concerns `fssk_state`, `fssk_state_from_coef`, and `fssk_vsig` in
`tensordev.sss`.

## Terminology

This plan keeps three independent choices separate.

1. **Truncation:** total degree or bidegree.
2. **Layout:** ordered or partially symmetrized.
3. **Coordinates:** standard or shear.

“Ordinary signature” means the classical path signature.  It does not mean
“standard coordinates”.  Layout is the only term used for the ordered versus
partially symmetrized choice.

For an FSSK kernel, `kernel.path_dim` is the dimension of the input path,
whereas `kernel.m` is the latent alphabet on which the Volterra signature is
defined.  A bidegree split always applies to this latent alphabet, so
`sum(core.dims) == kernel.m`; it does not split `kernel.path_dim`.

## Non-negotiable design requirements

1. **One production wordwise recurrence per problem family.**  Ordered
   total-degree, ordered bidegree, partial symmetrization, and shear are
   layout/coordinate adaptations around shared ordinary-signature and scalar-
   FSSK recurrence actions, not copied kernels.  The existing portable executor
   and deliberately independent test oracles remain separate so that they can
   detect errors in the wordwise implementation.
2. **No CPU regression.**  Unsupported or unprofitable work must take the
   present JAX route.  CPU execution must retain its current numerical results,
   asymptotic memory use, warm execution speed, and compilation behaviour within
   the fixed benchmark gates below.
3. **No eager GPU precomputation.**  `make_core` and `set_default_core` must not
   build wordwise plans.  Plans are derived lazily from the active truncation,
   rather than the maximum capacity of a core.
4. **No public dispatch knob.**  JAX device placement determines the lowering.
   Moving only the path computation to a GPU remains possible with
   `jax.device_put`, `jax.default_device`, or an enclosing device-placed JIT.
5. **Shear is a boundary transform.**  Every numerical recurrence produces
   standard-coordinate values.  A shear core applies its existing forward
   coordinate transform once to the completed result.
6. **Correct differentiation is required before automatic dispatch.**  A fast
   forward pass with a silently slow or incorrect gradient is not release
   quality.
7. **Performance claims are empirical.**  Automatic dispatch is enabled only
   for workload regions that pass reproducible forward and differentiated
   benchmarks on supported hardware.
8. **Fallback is ordinary behaviour, not an error.**  Missing Pallas support,
   an unsupported dtype, a large local graph, or an unknown platform selects
   the existing implementation.

## Public API contract

### Ordinary signatures

`td.path_signature` and `td.Signature` keep their exact public signatures and
return types.  In particular:

- `trunc`, `axis`, `block_size`, `accumulate`, `starting_point`,
  `output_starting_point`, `parallel`, `accumulate_in_tree`, `core`, and
  `seq_core` retain their present meanings;
- integer and pair truncations continue to be normalized by the selected core;
- outputs remain tuples for total degree and `BigradedTensor` objects for
  bidegree;
- an explicitly supplied core and the background default core behave
  identically;
- `parallel=True` continues to mean “form step exponentials and combine them by
  an associative tree”.  It is not reinterpreted as a GPU switch.

The first wordwise release is eligible only when `parallel=False` and
`accumulate_in_tree=False`.  Calls using either tree option follow the current
portable implementation until a separate wordwise tree design is justified.

### Scalar FSSK signatures

The `q == 1` state-space entry points gain the same core-selection convention as
the rest of TensorDev:

```python
fssk_state(..., trunc=None, core=None, seq_core=None)
fssk_state_from_coef(..., trunc=None, core=None, seq_core=None)
fssk_vsig(..., trunc=None, core=None, seq_core=None)
fssk_readout(..., core=None)
```

The exact final spelling should follow the package's current annotation and
argument-order style.  Behavioural requirements are:

- `core=None` resolves the background default core on state construction,
  coefficient-level state construction, and `fssk_vsig`; direct readout follows
  the tuple rule below;
- `seq_core=None` resolves consistently with that core;
- the resolved sequential core owns the portable block and time scans, so
  `seq_core` is not a dormant compatibility argument;
- total-degree cores accept an integer active truncation;
- bidegree cores accept a pair bounded by their maximum truncation;
- the core alphabet dimension equals `kernel.m` (or `coef.m` for the
  coefficient-level entry point); a dimension-free total core resolves that
  width from the kernel or coefficients;
- `kernel.q > 1` retains the current total-degree standard-coordinate code and
  its current validation;
- `kernel.q > 1` with a non-standard or bidegree core remains unsupported;
- the existing `q > 1` route retains its current API and shapes, and existing
  `q == 1` total-degree calls remain API- and shape-compatible.

`fssk_state_from_coef` cannot infer a bidegree rectangle from
`coef.trunc`, because that field contains only a total depth.  When the selected
core is bidegree, `trunc` is normalized independently and the coefficients must
cover at least `sum(trunc)`; any unused higher-order coefficient rows are
ignored. For a total-degree call with `trunc=None`, the active truncation is
`coef.trunc`.

FSSK state/update/readout requires a positive maximum retained order.  The
present standalone functions already reject total truncation zero, while
`StateSpaceSignature` currently permits construction of an unusable zero-level
state that cannot be read out.  Generalization resolves this inconsistency by
rejecting total truncation zero and bidegree `(0, 0)` at construction, records
the validation correction in the changelog, and adds a regression test.  This
does not alter the ordinary signature's valid scalar-only truncation.

`StateSpaceSignature` is part of this API migration.  It binds the core,
sequential core, and normalized integer-or-pair truncation; marks those fields
static in its JAX PyTree registration; and propagates them through every
constructor and through `update_with_path`, `update_with_increment`, `states`,
`vsig`, `readout`, and `reset`.  Existing call forms remain valid.  For
`kernel.q > 1`, only the existing standard total-degree route is accepted and
the existing default-core guard behaviour is retained.

Direct `fssk_readout` needs an explicit compatibility rule.  With `core=None`
and a tuple state, it uses the standard total-degree interpretation without
consulting the background default core. `fssk_vsig` and
`StateSpaceSignature` paths always pass their resolved core to readout; a direct
caller must do likewise for a total-degree shear state, whose tuple structure
does not identify its coordinates.  Bidegree states are validated against the
explicit core rather than relying only on their structural metadata.

### Choosing a GPU

No TensorDev-specific device argument is introduced.  For example, a process
whose other work stays on the CPU can place only a signature input on a GPU:

```python
gpu = jax.devices("gpu")[0]
x_gpu = jax.device_put(x, gpu)
sig = td.path_signature(x_gpu, trunc=6)
```

Dispatch must be based on the platform on which JAX lowers the computation, not
on `jax.default_backend()` and not on a Python-time guess from the default
device.  This is essential for nested `jax.jit`, explicit sharding, and a CPU
default with GPU-placed inputs.

## Current execution seams

The ordinary path currently follows:

```text
path_signature
  -> free_development
     -> core.normalize_truncation
     -> core.prepare_development_input
     -> development_neutral
     -> seq_core.tensor_abra
        -> lax.scan(tensor_fmexp), or the existing associative path
```

The new dispatch seam belongs in `path_signature`, but the normalization work
shown above currently lives in `free_development`.  First extract private
preparation and finalization helpers:

```text
_prepare_free_development_call
    resolve core and sequential core
    normalize truncation and default axis
    call core.prepare_development_input
    construct and validate the neutral/seed
    derive one immutable seed/block policy

_run_portable_free_development
    execute the current tensor_abra route on prepared data

_finalize_signature_call
    restore block axes and native containers
    apply the policy's remaining seed/output-starting-point actions
```

`free_development` becomes a thin preparation-plus-portable wrapper;
`path_signature` prepares its one-level input once and dispatches the prepared
call to either the portable runner or the wordwise runner.  The helpers must be
used by both paths—copying the existing resolution, seed, block, or axis logic
into `path_signature` is prohibited.

The seed/block policy records whether the seed is passed into the portable scan
or applied after independent blocks, plus the canonical starting tensor and
prepend behaviour.  The portable runner consumes it in exactly the current
order; the wordwise runner maps the same policy onto its output.  This shares
semantics without forcing a new floating-point/product order onto the CPU path.

The whole time loop must be inside one compiled wordwise operation.  Dispatching
inside `tensor_fmexp` or `tensor_abra` would launch a kernel at every step and
would also contaminate every free development with a signature-specific
optimization.

The scalar FSSK path currently follows:

```text
fssk_vsig
  -> fssk_state
     -> project increments through kernel.A[0]
     -> kernel.coef
     -> fssk_state_from_coef
        -> nested lax.scan
        -> scalar tensor-algebra update
  -> fssk_readout
```

The wordwise operation should replace only the repeated state update/readout
work.  Projection, coefficient construction, input normalization, and output
axis handling remain shared JAX code.

## Proposed private architecture

Introduce one private package; names may be adjusted to match the final source
tree, but the responsibilities must remain separated.

```text
src/tensordev/_wordwise/
    __init__.py       # private; exports no public API
    layout.py         # immutable standard-coordinate output plans
    plans.py          # lazy decoding and prefix-graph construction
    reference.py      # small pure-JAX mathematical oracles
    pallas.py         # shared launch, tiling, and feature helpers
    ordinary.py       # ordinary-signature recurrence and wrapper
    fssk_q1.py        # scalar FSSK recurrence and wrapper
    dispatch.py       # eligibility and lowering-time platform selection
```

Public wrappers remain in `development/sig.py` and `sss/state_update.py`.
Core-specific source files expose only the minimum host-side information needed
to adapt their native layouts.  Numerical GPU code must not branch on concrete
core classes.

The package has three internal layers:

1. a **layout planner**, which describes independent output computations in
   standard coordinates;
2. a **recurrence action**, shared by all layout families;
3. an **executor**, initially Pallas on supported NVIDIA GPUs and the existing
   JAX implementation everywhere else.

This separation is the main guard against code duplication.

## The shared layout plan

Define an immutable `WordwiseLayoutPlan` (the precise class name is private)
with structural, host-side data only.  It records:

- grading kind: total degree or bidegree;
- alphabet dimensions;
- normalized active truncation;
- whether the layout is partially symmetrized;
- standard-coordinate grades and block widths;
- flat output offsets and reconstruction information;
- launch groups or buckets with homogeneous loop and storage bounds;
- optional ordered decoding metadata;
- optional partially symmetrized prefix graphs;
- a reconstruction method or static tree specification for the core's native
  tensor container.

Planning and execution use separate keys.

The host layout-plan key is dtype-independent and contains only grading family,
resolved alphabet dimensions, active truncation, and the partially symmetrized
flag.  For a total-degree kernel, a shear split `(d_prime, d_doubleprime)` is
canonicalized to the single width `d_prime + d_doubleprime`; the split belongs
only to the later shear-transform plan.  This is what permits a standard total
core and a total shear core of the same width to reuse the numerical recurrence.
For bidegree, the split remains part of the grading layout.

The executor/kernel-factory key additionally contains the plan fingerprint,
dtype, flat batch and step shape where compilation requires them, block/output
mode, uniform-versus-varying FSSK coefficients, FSSK state dimension, selected
Pallas lowering/target profile, and tile/resource parameters.  JAX's executable
cache remains authoritative; TensorDev does not introduce a second persistent
compilation cache or runtime autotuner.

Neither key contains:

- Python object identity;
- standard versus shear coordinates;
- shuffle-precomputation state;
- inactive grades up to a bidegree core's maximum capacity.

Consequently, standard and shear cores with the same underlying standard
layout reuse the same compiled recurrence.

Plan construction is lazy and bounded by an LRU cache.  Before allocating an
array, closed-form combinatorial counts are used wherever available.  Otherwise
the counter must terminate as soon as a byte, element, integer, or host-time
budget is exceeded; a “count-only” pass may not enumerate the same enormous
graph that it is intended to prevent.  Oversized or slow-to-plan cases select
the portable route without partially constructing large arrays.  Host planning
time is benchmarked and gated alongside plan bytes.

All device indices use signed 32-bit integers in the initial kernel.  A count or
offset exceeding `int32` is a clean fallback condition, not an overflow or an
attempt to allocate the impossible plan.

### Total-degree ordered layout

No decoder arrays are needed.  A degree-`r` coordinate is a base-`d` word code;
letters are recovered arithmetically.  The scalar block is created directly and
does not launch a program.

### Bidegree ordered layout

The retained word set is prefix-closed.  A block coordinate is determined by:

- the colexicographic placement of prime letters;
- the dense prime-word index;
- the dense double-prime-word index.

The baseline reuses the existing bidegree grade plan, particularly its
`placements` and `block_to_total_indices`, and decodes the resulting total word
code in base `sum(dims)`.  A benchmark must compare this map load with an
analytic placement/dense-index decoder.  Only the selected strategy is copied
to the device; the kernel must not carry both encodings.

### Partially symmetrized bidegree layout

An ordered-to-partial transform after computing every ordered word is a useful
correctness fallback, but it defeats the main memory benefit.  The production
path is quotient-native.

For each double-prime multiset rank, precompute the manuscript's finite
multiset-prefix set and close it under both predecessor operations.  When the
terminal multiset block is nonempty, every species present supplies one
predecessor edge that removes one occurrence of that species.  Only when the
terminal block is empty is the prime predecessor available, crossing to the
preceding complete block.  This is not an ordered terminal-letter chain.  Each
graph stores:

- its closed set of prefix nodes;
- predecessor/source and destination indices;
- the appended letter or generator class;
- exact multiplicity coefficients required by the partial symmetrization;
- the terminal node corresponding to the requested output coordinate.

`grade_plan.placements[rank]`, whose normal form has shape
`(prime_count + 1, d_doubleprime)`, is the starting point for this construction.
The graph depends on the multiset rank but not on the dense prime word.  It must
therefore be stored once and reused across all `d_prime ** prime_count` prime
words.  Duplicating it per scalar output is prohibited.

Existing grade, placement, and double-prime generator plans are the authoritative
source of ordering and predecessor data.  The wordwise planner references or
compacts those plans; it must not independently rebuild and retain an equivalent
full generator map.  Its cache owns only wordwise-specific prefix closure,
bucket, and terminal-node metadata that cannot be recovered cheaply inside the
kernel.

Ranks are bucketed by equal, or conservatively rounded, graph size so each
kernel has static local-storage bounds.  A natural grid is

```text
flat batch x multiset-rank bucket x prime-word tile
```

One program unit owns one output coordinate or one independent vector lane.
There are no output collisions and therefore no atomics.  This gives
deterministic accumulation order.

The same right-first-level predecessor action is used by the ordinary and FSSK
recurrences.  The ordered prefix chain is its implicit special case and carries
no graph arrays.

## Ordinary wordwise recurrence

For a target word `w = i_1 ... i_r`, retain its prefix values
`a_0, ..., a_r`, initially `a_0 = 1` and `a_p = 0` for `p > 0`.  For each path
increment, update the nonempty prefixes in descending order.  The inner Horner
recursion is

```text
h = 0
for q = 0, ..., p - 1:
    h = increment[i_(q + 1)] / (p - q) * (a_q + h)
a_p = a_p + h
```

The output is `a_r`.  Storage is `r + 1` scalars per ordered word and arithmetic
is quadratic in `r` per path step.  For a quotient graph, the same descending
right-generator action updates graph nodes rather than a single chain.

### Input normalization

Outside the wordwise primitive:

1. normalize and validate `axis`;
2. take `jnp.diff` unless `increment_input=True`;
3. move the step and coordinate axes to a canonical position;
4. broadcast and flatten all batch axes to `[flat_batch, steps, d]`;
5. retain a static reconstruction description for the output.

Keeping differencing outside the custom operation lets ordinary JAX rules map
cotangents back to path nodes.

### Launch structure

A Pallas program handles a tile of independent words; it is not launched once
per word.  Vector lanes own independent prefix chains or graph actions.  The
step loop lives inside the kernel and is expressed with JAX/Pallas control
flow, never Python-unrolled over path length.

Initial launch groups are homogeneous by total degree or bidegree so loop bounds
and local storage are static.  After the baseline is stable, benchmark combining
compatible bidegrees with the same total order to reduce launch count.  This
grouping is retained only if it improves end-to-end time without inflating
register pressure or compilation.

Programs should be grouped so a block or CTA processes multiple output words
for the same batch sample.  Benchmark direct global reads of increments against
staging a time/letter tile in shared memory.  Shared-memory staging is not
assumed to win for short paths.

### Blocks and accumulation

The existing blocking contract must be implemented without repeated host or
Python dispatch.

- `block_size=None`: run the complete path once and emit one result.
- `accumulate=True`: run the path once and emit the running state at block
  boundaries.
- `accumulate=False`: fold the block index into the flat batch dimension and
  process blocks independently from the same seed.
- block divisibility is validated before dispatch.
- `output_starting_point=True`: prepend the seed through the shared output
  assembly helper.

An arbitrary `starting_point` need not be folded into every wordwise recurrence.
The first implementation computes the path signature from the unit and applies
one existing native tensor product with the starting point to the assembled
result.  This preserves one implementation of Chen composition.  Initializing
the wordwise state directly from a seed is considered only if the final product
accounts for more than 10% of warmed end-to-end time in a release benchmark and
an exact layout-neutral design is available.

## Scalar FSSK recurrence

### Required core-neutral state prerequisite

The current scalar state code is hard-wired to `Jax()` and tuple-valued total
degree tensors.  GPU work must not copy that limitation into a second path.
Before writing the Pallas kernel:

1. make the `q == 1` state/update/readout code resolve a core and active
   truncation normally;
2. represent state in the selected core's native tensor container;
3. preserve leading state axes as batch-like axes, ending in the existing
   singleton and state dimensions before the coordinate axis;
4. keep the current `q > 1` state code exactly on its existing standard
   total-degree route;
5. prove portable equivalence for every in-scope core before enabling any GPU
   dispatch.

A total-degree hidden state remains a first-on tuple with no degree-zero level.
A bidegree hidden state is a `BigradedTensor` whose spec has
`include_scalar=False`.  Every coordinate conversion, partial symmetrization,
and ordered conversion applied to hidden state passes `first_on=True`; signature
outputs contain their scalar block and use `first_on=False`.  Readout contracts
the state axes, restores the scalar unit, and returns the selected core's native
signature container.

For a bidegree truncation `(N, M)`, coefficient construction needs scalar FSSK
coefficients through total depth `N + M`, while the layout planner retains only
coordinates inside the rectangle.  This must be explicit; passing a pair to
existing coefficient code that expects an integer is not valid.

### Ordered target recurrence

Let `y` be the projected latent increment, and let `E`, `psi`, and `phi` be the
existing step-local scalar FSSK coefficients.  For one target word of length
`r`, retain row-vector prefix states `z_1, ..., z_r` in state dimension `R`.
At a step, update `p` in descending order by the existing scalar recurrence,
expressed for the target chain as:

```text
h = psi[p - 1]
for a = 1, ..., p - 1:
    h = h * y[i_a] + z[a] @ phi[p - 1 - a]
z[p] = z[p] @ E + h * y[i_p]
```

The local storage is `r * R`.  With dense state matrices, work is quadratic in
word length and in `R` per step.  The implementation must be checked term by
term against `recursion_scalar.py`; the pseudocode above is not itself a licence
to change coefficient indexing, row/column conventions, or broadcast order.

Projection is shared and performed once:

```text
y = kernel.A[0] @ increment
```

`kernel.coef` remains existing JAX code and remains differentiable.  A uniform
scalar or length-one `dt` produces one coefficient set and is consumed in a
stride-zero/broadcast mode; it must not be materialized `steps` times.  A
time-varying `dt` supplies per-step coefficients.

FSSK input canonicalization preserves the current independent broadcasting
rules.  It moves the step axis to the front; broadcasts the projected-path
batch, coefficient batch (excluding its time axis), and initial-state batch to
one common shape; and flattens that shape only at the executor boundary.  A
uniform coefficient set is tagged rather than expanded, whereas varying
coefficients retain a leading step axis.  Output assembly unflattens every
batch axis and restores the requested block axis.  Tests cover cases in which
the path, coefficients, and seed each contribute different broadcast axes.

For terminal `fssk_vsig`, form the existing readout vector from
`exp(-Lambda * tau_dt)` and `kernel.b[0]` outside the recurrence.  The terminal
coordinate is the final state contracted with that vector.  A single kernel
factory parameter such as `emit_state` distinguishes:

- `fssk_state`, which emits the native hidden state at requested boundaries;
- `fssk_vsig`, which may fuse terminal readout and avoid materializing an
  externally unused state.

The recurrence source must not be duplicated between these two modes.

### Partially symmetrized scalar FSSK

The quotient-native target is to use the same predecessor graph as the ordinary
partially symmetrized recurrence.  Both the polynomial `f` and the state action
`ZG` can be evaluated through repeated right-generator actions; only node
payloads differ (scalars versus `R`-vectors or matrices).

This reuse is an implementation obligation, not an assumption.  Before the GPU
kernel is enabled, the reference layer must establish that:

- the graph is closed under every predecessor required by the scalar FSSK
  update;
- multiplicity factors match the existing partial symmetrization convention;
- direct quotient-native results equal ordered results followed by
  `tensor_partially_symmetrize(..., first_on=True)` at low and moderate
  truncations;
- state readout commutes with partial symmetrization.

If any obligation fails, the release fallback is ordered computation followed
by the existing partial symmetrization, or the wholly portable route.  No
heuristic quotient recurrence is shipped.

### Shear FSSK state and readout

The recurrence is performed in standard coordinates.  For a shear core:

- convert a supplied initial state to standard coordinates once;
- run the recurrence entirely in standard coordinates;
- transform each emitted state or signature to shear coordinates once at the
  output boundary;
- never transform at every time step.

Readout contracts only state-space axes and therefore commutes with the tensor
coordinate transform.  Tests must verify both orderings.  The implementation
should choose the ordering that avoids materializing a larger intermediate for
the requested output mode.

## Pallas execution backend

Use Pallas as the sole initial GPU backend.  It is already part of the supported
JAX dependency and avoids a CUDA toolchain, custom wheel, and a second language
implementation.  Imports remain private and lazy because Pallas is documented
as experimental and its API can move across supported JAX versions.

The implementation targets the Pallas lowering actually tested on supported
NVIDIA hardware.  It must not describe an untested GPU family as supported.
Pallas interpret mode on CPU is a development oracle only; it does not validate
GPU lowering, races, register pressure, shared-memory pressure, floating-point
behaviour, or performance.

Kernel rules:

- all plan and graph arrays are explicit operands; no JAX array is captured in
  a Pallas closure;
- words are tiled, with one vector lane/program unit owning one output;
- scalar blocks are assembled without launching a kernel;
- only `float32` and `float64` are candidates initially;
- lower-precision and complex dtypes use the portable route until separately
  specified and tested;
- local-storage and shared-memory estimates are checked before lowering;
- no atomics are used;
- no full prefix trajectory over time is stored for the forward pass;
- unsupported index ranges or resource estimates fall back before compilation.

If Pallas cannot pass correctness, supported-version, compilation, and
performance gates, stop and reassess.  Do not ship Pallas beside a second native
CUDA implementation.  A future JAX FFI/CUDA plan would replace the production
executor behind the same planner/recurrence contract.

Relevant upstream references are the [JAX Pallas
documentation](https://docs.jax.dev/en/latest/pallas/index.html), its
[quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html), and the
[JAX FFI documentation](https://docs.jax.dev/en/latest/ffi.html).

## Lowering-time dispatch

The private dispatcher has two independent decisions:

1. is the call *semantically and structurally eligible* for a wordwise kernel?
2. on which platform is the staged computation lowered?

Static eligibility includes:

- the operation is an ordinary signature or scalar FSSK signature;
- the active core/layout has a validated plan;
- the active truncation, dtype, index range, and local resource estimate are
  supported;
- ordinary-signature tree flags are false;
- block/seed/output mode has a tested implementation;
- the workload is in a benchmarked profitable region.

An initial feasibility prototype may use
`jax.lax.platform_dependent(cuda=wordwise, default=portable)`; the keyword is
`cuda`, not `gpu`.  It is not the release dispatcher.  The function traces all
branches when the platform is not concrete, can build GPU plans during a CPU
trace, and distinguishes CUDA from other platforms without distinguishing an
unsupported NVIDIA architecture.

The release dispatcher is therefore a small private JAX primitive with
platform-specific lowerings:

- its abstract rule describes only the canonical flat output shapes;
- its CPU/default lowering invokes the exact current portable implementation;
- its CUDA lowering queries lowering-target metadata and emits Pallas only for
  an allowlisted, feature-probed architecture/JAX/Pallas combination;
- an unknown architecture, unavailable target metadata, unsupported Pallas
  lowering, ROCm, TPU, multi-platform export, or unsupported dtype lowers the
  portable implementation;
- the CUDA lowering constructs the lazy host plan and passes every resulting
  device array as an explicit Pallas operand;
- its batching rule folds supported `vmap` axes into the canonical flat batch;
  differentiation is supplied by the wrapper described below.

The feasibility phase must prove that the supported JAX versions expose enough
target information at lowering time to make the CUDA decision safely.  If they
do not, native automatic dispatch is limited to concrete, single-device calls
whose actual target can be feature-probed; traced/AOT cases lower portable.  A
Pallas compilation failure on an unsupported GPU is not an acceptable fallback
mechanism.

This primitive is dispatch infrastructure, not a second numerical executor.
Its extra JAX-version sensitivity is covered by the minimum/maximum-version CI
matrix.  It also prevents a CUDA-only plan from adding CPU tracing and planning
cost.  A concrete eager fast path may bypass the primitive when its input has an
unambiguous device, but a Python-time default-backend guess must never determine
a traced computation.

The portable branch must be the exact current implementation rather than a new
reference recurrence.  Reference code exists for tests and derivation, not as a
third production executor.

## Automatic differentiation

Pallas forward execution must not be assumed to supply the required reverse
rule.  The first correctness bridge follows the established optional native CPU
pattern:

- wrap the native primal in `jax.custom_jvp`;
- compute the tangent with `jax.jvp` of the current portable implementation;
- let JAX transpose that exact tangent rule for reverse mode;
- keep differencing, projection, coefficient construction, shear transforms,
  and output assembly outside the custom operation where possible.

This gives a native forward value and a portable, mathematically identical
derivative.  It is sufficient for correctness, but not automatically sufficient
for automatic dispatch: value-and-gradient benchmarks must also pass.

Forward and differentiated allowlists are enforced separately.  In an ordinary
primal call, the wrapper may choose the native primal when the forward workload
passes its gate.  Under JVP/transpose, the custom JVP rule makes the complete
choice again: a gradient-approved workload uses native primal plus the portable
tangent; a workload outside the gradient allowlist uses portable primal plus
portable tangent.  JAX invokes the rule in place of the original primal during
differentiation, so this prevents a forward-approved but gradient-unprofitable
native call from leaking into `value_and_grad`.  Until this mechanism is tested,
one conservative allowlist governs both modes.

The second differentiation stage is a native adjoint when the portable tangent
is the bottleneck.

For ordinary signatures, derive the adjoint from the wordwise prefix recurrence
using terminal prefix values and backward reconstruction/checkpointing.  The
design should follow the memory principle of the wordwise CUDA implementation
described in [*Efficient Computation of Path Signatures on
GPUs*](https://arxiv.org/abs/2602.24066): do not retain the full time-by-prefix
trajectory merely to differentiate.

The scalar FSSK adjoint is separate.  It must return cotangents for projected
increments, `E`, `psi`, `phi`, and readout weights so outer JAX code continues
to differentiate with respect to `A`, `Lambda`, `b`, `dt`, and `tau_dt`.
Backward reconstruction through `E` or checkpointed segments must be derived
and numerically stress-tested before use; invertibility alone is not a stability
argument.

Gradient requirements include:

- forward-mode JVP;
- reverse gradients with respect to path nodes and increments;
- ordinary-signature gradients through a nontrivial starting point;
- FSSK gradients with respect to path, `A`, `Lambda`, `b`, `dt`, and readout
  lag where the present API supports them;
- gradients through ordered, partially symmetrized, and shear outputs;
- gradients under outer `jax.jit` and `jax.vmap`.

## Correctness strategy

The existing portable implementation is the primary behavioural oracle.  It is
supplemented by independent low-degree recurrences so a shared bug is not
mistaken for agreement.

### Plan and reference tests

Add focused private tests, for example:

```text
tests/wordwise/test_wordwise_plans.py
tests/wordwise/test_reference.py
tests/wordwise/test_dispatch.py
tests/wordwise/test_pallas_interpret.py
tests/wordwise/test_gpu_ordinary.py
tests/wordwise/test_gpu_fssk_q1.py
```

Plan tests verify:

- total word decoding against explicit Cartesian products;
- bidegree block order against `block_to_total_indices`;
- every retained predecessor is present;
- partial graph terminal nodes and multiplicities;
- exact block offsets and reconstruction;
- plan reuse between standard and shear cores;
- cache keys distinguish every numerically relevant structure;
- closed-form or early-terminating resource rejection occurs before allocation;
- `int32` and local-resource overflow select fallback.

### Ordinary-signature comparisons

Test all six core families over:

- `float32` and `float64` where the target enables them;
- unbatched, one-dimensional batch, and multiple batch axes;
- `axis=0`, `axis=-2`, and nontrivial normalized axes;
- node input and increment input;
- active truncations below a core's maximum capacity;
- the dimension-free default `Jax()` core with several input dimensions and
  explicit truncations;
- total truncations including low edge cases;
- bidegrees with one component zero as well as both positive;
- zero paths, constant paths, one increment, and random paths;
- `block_size=None`, full-size blocks, and multiple blocks;
- `accumulate=True` and `False`;
- unit and nontrivial starting points;
- `output_starting_point=True` and `False`;
- default and explicitly supplied cores.

Structural cross-checks include:

- ordered bidegree converted to total degree equals the matching total-degree
  signature on retained words;
- partially symmetrized output equals ordered output passed through
  `tensor_partially_symmetrize(..., first_on=False)`;
- inverse-transforming a shear result equals the standard-coordinate result;
- blockwise accumulation obeys the same Chen identity as the current route.

Compilation instrumentation must also show that standard and shear cores with
the same standard-coordinate layout share one recurrence executable, and that
two separately constructed but structurally identical cores do not cause a
second recurrence compilation.  Coordinate transforms may compile separately.

### Scalar FSSK comparisons

Test uniform and time-varying `dt`, dense and Jordan `Lambda`, state dimensions
`R = 1, 2, 4, ...`, multiple latent dimensions, blocking, initial states, and
readout lags.  Required identities are:

- current scalar total-degree `fssk_state` equals the new portable core-neutral
  route;
- `fssk_readout(fssk_state(...))` equals direct `fssk_vsig(...)`;
- ordered bidegree, partial symmetrization, and shear agree through the same
  conversions used for ordinary signatures;
- for the identity-kernel specialization (`Lambda = 0`, compatible identity
  projection and constant readout), scalar FSSK output recovers the ordinary
  signature in every supported core family;
- quotient-native partial results equal ordered results followed by
  `tensor_partially_symmetrize`, with the correct `first_on` value for state or
  signature output;
- readout before versus after a tensor coordinate transform agrees;
- `q > 1` tests remain unchanged and still reject unsupported cores.

Public-boundary regressions additionally cover:

- default versus explicit core and sequential-core resolution;
- `fssk_state_from_coef` with an explicit active bidegree below core capacity
  and with extra available coefficient depth;
- direct `fssk_readout`, including its standard-total tuple rule and explicit total-
  shear core;
- every `StateSpaceSignature` constructor and its `update_with_path`,
  `update_with_increment`, `states`, `vsig`, `readout`, and `reset` methods;
- conversion and validation of ordered, partially symmetrized, and shear
  initial states with `first_on=True`;
- every currently accepted scalar, length-one, step-vector, batched, and
  batch/step `dt` shape;
- independent broadcasting contributions from path, coefficients, and seed;
- invalid path/latent alphabet dimensions, incompatible core/state metadata,
  invalid truncations, and insufficient coefficient depth;
- preservation of the current `q > 1` default-core guard and error boundary.

Pallas interpret-mode tests run on CPU for every kernel shape family.  Separately
marked tests run on real NVIDIA devices and compare forced-portable,
forced-wordwise, and automatic dispatch.

### Transformation and composition tests

The single-device wordwise path must work inside:

- outer `jax.jit`;
- `jax.vmap` over one and multiple axes;
- `jax.lax.scan`;
- JVP and reverse-mode transformations.

The first release does not select the native wordwise path under `pmap`,
multi-device `NamedSharding`, or multi-platform export.  Its batching,
partitioning, or export guard must route those cases through the portable
implementation rather than erroring.  Multi-device native execution is a later
programme with an explicit partitioning contract.

The tests should detect unintended recompilation when only array values change.
Different active truncations are allowed to compile separately.

Comparisons use dtype-appropriate tolerances because the recurrence changes
floating-point association.  They do not demand bitwise identity.  Repeated
runs on identical inputs must nevertheless be deterministic; atomics and
nondeterministic reductions are absent by design.

Existing suites in `tests/development`, `tests/sss`, `tests/core/shear`,
`tests/core/symmetrized`, and `tests/volterra` remain mandatory regression
coverage.

## Benchmark and dispatch policy

Benchmarks are standalone and always synchronize results with
`block_until_ready`.  Pytest wall-clock assertions are prohibited.

For each case, record separately:

- host plan-construction time and plan bytes;
- JAX tracing/lowering time;
- device compilation time;
- first execution time;
- warmed execution median and lower/upper quantiles;
- output bytes and measured peak/temporary device memory;
- executable/code size where available;
- forward-only and value-and-gradient measurements;
- cost of standard-to-shear transformation separately from the recurrence.

Benchmark forced portable, forced wordwise, and automatic dispatch.  Private
forcing helpers are permitted in tests and benchmarks only; they are not public
API.

The workload matrix is stratified rather than a full Cartesian product.  A
closed-form output-size check removes cases that cannot fit with a fixed safety
margin in device memory; every axis and important interaction is still sampled,
and the reason for each omitted case is recorded.  In particular, batch
`10_000` is not combined blindly with the largest alphabet and degree.

The strata include:

- batch sizes `1`, a moderate training batch, and `10_000`;
- path lengths `16`, `64`, `256`, and `1024`;
- alphabets `2`, `3`, `4`, and `8` where memory permits;
- total degrees `2` through `8`;
- representative bidegrees, including `dims=(1, 2), trunc=(8, 4)`;
- all six supported core families;
- partially symmetrized graph-size buckets, not merely output coordinate
  counts;
- FSSK state dimensions `R = 1, 2, 4, 8, 16`;
- uniform and varying `dt`;
- dense and Jordan state matrices;
- terminal and blocked/trajectory outputs.

At least one Ampere-class and one Hopper-class NVIDIA GPU should be measured if
both are described as supported.  The package's minimum and maximum supported
JAX lines (currently `0.10` and `0.11`) must both be exercised.

Initial automatic-dispatch regions are conservative allowlists generated from
benchmark evidence.  Each warm case receives five untimed warmups and at least
30 synchronized measurements; first-call/compile cases use at least five fresh
processes.  Report medians, 10th/90th percentiles, and a bootstrap 95% confidence
interval for the paired ratio.  Do not delete individual outliers.  A run with
predeclared thermal-throttling or competing-load evidence is invalidated as a
whole and repeated.

The release pass/fail thresholds are:

- an automatically enabled GPU case has at least a 20% median warm speedup and
  a 95% confidence lower bound of at least 10%, both for its relevant forward
  or value-and-gradient mode;
- its measured peak device memory is no greater than the portable path beyond a
  2% measurement tolerance;
- host planning takes at most 100 ms and no more than 25% of the portable first-
  compile time for that case;
- native compile plus first execution either beats the portable first call or
  amortizes its excess within at most five warm calls;
- on CPU, warm runtime may regress by at most 3% in the median with an upper
  95% confidence bound of 5%; trace/compile and first-call time may regress by
  at most 5% or 25 ms, whichever allowance is larger;
- CPU peak process memory may regress by at most 3%, and no GPU wordwise plan
  bytes may be retained by a CPU-only call;
- package import and core construction build no wordwise plans and remain
  within the same 3% timing tolerance.

Forward and value-and-gradient allowlists are independent as specified in the
custom-JVP rule.  Unknown shapes, dtypes, graph sizes, devices, and software
versions use the portable route.  These thresholds may be tightened from the
Phase 0 evidence, but may not be relaxed during implementation without recording
and approving the reason in this plan.

Compilation is a first-class gate.  The release report publishes both first-call
and warm numbers and the measured amortization count.

## Implementation phases and exit gates

### Phase 0 — freeze baselines

- Capture current API, correctness, compile, runtime, and peak-memory baselines.
- Include the six ordinary core families and current standard total-degree FSSK
  `q == 1` and `q > 1` paths.
- Record exact hardware, driver, CUDA, Python, JAX, and jaxlib versions.

**Exit:** repeatable baseline artefacts and commands are checked into
`benchmarks/`; no implementation claim relies on an unsynchronized timing.

### Phase 1 — shared plans and reference recurrences

- Extract the shared free-development preparation, portable execution, and
  signature-finalization helpers without changing behaviour.
- Implement immutable layout plans, separate bounded plan/kernel caches,
  closed-form or early-terminating resource guards, and reconstruction.
- Implement small pure-JAX ordinary and scalar-FSSK wordwise references.
- Validate total decoding, bidegree ordering, and partial prefix graphs.
- Prototype target-capability discovery and the private platform-lowered
  primitive on every supported JAX line.

**Exit:** exhaustive low-degree tests agree with independent enumerations and
the current portable implementation; plan construction has explicit memory and
time/overflow tests; CPU preparation still lowers through the current path; and
unsupported CUDA targets can be selected away before Pallas compilation.

### Phase 2 — ordinary ordered Pallas kernels

- Implement one tiled ordered recurrence for total degree and bidegree.
- Keep scalar output construction and path normalization outside the kernel.
- Cover both configured and dimension-free `Jax()` total cores.
- Add interpret-mode and real-GPU tests.

**Exit:** ordered standard-coordinate results and JVPs agree for all input,
batch, axis, and active-truncation cases; no bidegree-specific recurrence has
been copied.

### Phase 3 — ordinary sequence semantics, seeds, and shear

- Implement blocking and accumulation modes.
- Reuse the existing tensor product for nontrivial starting points.
- Apply the existing shear transform once after standard-coordinate execution.

**Exit:** the complete `path_signature` contract passes for the four ordered
families, including outer JAX transformations and CPU fallback.

### Phase 4 — ordinary partial symmetrization

- Implement graph bucketing and quotient-native Pallas execution.
- Retain ordered plus `tensor_partially_symmetrize` only as a correctness or
  resource fallback.
- Measure direct quotient execution against dense ordered computation.

**Exit:** both partially symmetrized core families pass conversion and gradient
tests; dispatch is enabled only for buckets that beat the portable path.

### Phase 5 — core-neutral scalar FSSK state/readout

- Generalize `q == 1` state containers, validation, truncation, and readout to
  every supported core.
- Add explicit active truncation to `fssk_state_from_coef`, bind core/truncation
  in `StateSpaceSignature`, and preserve the direct-readout standard-total
  tuple rule.
- Use the resolved sequential core for the portable block/time scans.
- Keep `q > 1` on its current route.
- Avoid changing generic `td.vsig`.

**Exit:** portable results satisfy all cross-layout identities and existing
`q > 1` tests are unchanged in result, shape, guard behaviour, and performance;
every public FSSK state/readout boundary passes its compatibility tests.

### Phase 6 — ordered scalar FSSK Pallas kernels

- Share ordered decoding and launch infrastructure with ordinary signatures.
- Implement one recurrence with state-emitting and readout-emitting modes.
- Treat uniform coefficients without stepwise materialization.

**Exit:** ordered total/bidegree standard results, states, blocking, and
parameter gradients match the portable oracle on real GPUs.

### Phase 7 — scalar FSSK partial symmetrization and shear

- Reuse the proven predecessor graph.
- Validate quotient-native FSSK action before enabling it.
- Apply standard/shear transforms only at state/output boundaries.

**Exit:** all six core families pass state, readout, conversion, and gradient
tests; resource-heavy graph buckets fall back cleanly.

### Phase 8 — differentiation optimization

- Land `custom_jvp` correctness bridges first.
- Benchmark value-and-gradient separately.
- Derive and implement native adjoints only where needed for automatic dispatch.

**Exit:** no automatically selected path has a correctness failure or exceeds
the value-and-gradient time/memory gates; gradient memory is documented and
bounded.

### Phase 9 — automatic dispatch and regression audit

- Derive conservative allowlists/thresholds from the benchmark matrix.
- Exercise CPU, unsupported-device, dtype, overflow, and resource fallbacks.
- Audit import, core-construction, first-call, compilation, and warm CPU costs.
- Verify plan and executable reuse across structurally identical standard and
  shear cores.

**Exit:** every automatically dispatched case passes the fixed statistical
gates, and existing CPU users remain inside all CPU time/memory thresholds.

### Phase 10 — documentation and release

- Add concise GPU behaviour and device-placement examples to the README/API
  docs.
- State supported devices/dtypes and automatic fallback without exposing
  internal thresholds as an API promise.
- Cite the implementation note and the GPU paper for algorithmic detail.
- Remove development/debug commentary and audit terminology.

**Exit:** full CPU and GPU CI passes, benchmark artefacts are reproducible, and
the public docs describe functionality rather than implementation history.

## Native-dispatch release checklist

This checklist governs enabling automatic native execution; it does not block
the 0.1.0 portable-JAX release. Items verified without NVIDIA hardware are
marked complete. Native dispatch remains disabled until every hardware,
differentiation, and performance item is complete.

- [x] No public signature or default-core workflow was broken.
- [x] No alternative layout selector or new device/backend option was added.
- [x] All six existing core families work for ordinary signatures through the
      portable implementation and candidate interpreter path.
- [x] The dimension-free default `Jax()` core resolves its alphabet from the
      input and remains supported.
- [x] All six existing core families work for exact FSSK `q == 1` state and
      signature output through the portable implementation and candidate
      interpreter path.
- [x] Direct FSSK readout and every `StateSpaceSignature` method obey the new
      core/truncation contract; a tuple passed without `core` is interpreted as
      standard total degree.
- [x] FSSK `q > 1` and generic `td.vsig` remain on their current paths.
- [x] Standard and shear coordinates share compiled standard-coordinate
      recurrences.
- [x] Ordered and partially symmetrized layouts share one recurrence action and
      planner protocol.
- [x] `make_core` performs no wordwise precomputation.
- [x] Plan caches are bounded and keyed structurally.
- [x] Host plan and executable caches are separate; structurally identical
      standard/shear cores reuse the recurrence executable.
- [x] Unsupported/resource-heavy calls fall back without error.
- [x] Automatic native dispatch is disabled while no validated allowlist is
      installed.
- [ ] Unknown CUDA architectures and multi-device/exported calls select the
      portable route before Pallas compilation.
- [x] CPU import and core construction do not build wordwise plans.
- [ ] CPU compile, warm-runtime, and memory benchmark gates pass on the release
      matrix.
- [ ] Real NVIDIA correctness tests pass on every claimed architecture.
- [ ] Forward, JVP, reverse-gradient, nested-JIT, and batching tests pass.
- [ ] First-call, warm, memory, and value-and-gradient benchmarks are published.
- [ ] There are no atomics, nondeterministic reductions, captured Pallas arrays,
      or Python-unrolled time loops.
- [x] Documentation uses truncation, layout, and coordinates consistently.
- [x] Source comments explain current invariants and mathematics.

## Definition of done

This programme is complete only when ordinary signatures and exact scalar FSSK
signatures produce the existing native tensor types for every supported core;
the GPU path is selected transparently only where it is correct and measurably
beneficial; gradients are correct and performance-audited; shear and partial
symmetrization introduce no duplicate recurrence; CPU behaviour passes every
fixed regression gate; and every unsupported case follows the existing
implementation.
