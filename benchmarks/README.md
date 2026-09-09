# Bidegree efficiency benchmark

`bidegree_efficiency.py` compares the native bidegree implementation with the
total-degree implementation for ordinary signatures and Volterra signatures.
It is intentionally a standalone diagnostic rather than a pytest timing test:
wall-clock thresholds are unreliable across machines, JAX versions, and
backends.

The matched comparisons use a four-dimensional path split as `(2, 2)`:

| comparison | total payload | bidegree payload | maximum word length |
|---|---:|---:|---:|
| total 6 versus `(3, 3)` | 5,461 | 2,229 | 6 |
| total 8 versus `(4, 4)` | 87,381 | 31,381 | 8 |

The payload includes the scalar coordinate. The bidegree count is

`sum(comb(n + m, n) * 2**n * 2**m for n <= N, m <= M)`.

The optional total 6 versus `(4, 4)` comparison is explicitly reported as
non-equivalent: `(4, 4)` retains 31,381 coordinates and words of lengths 7 and
8, whereas total 6 retains only 5,461 coordinates.

Run a quick check with:

```console
PYTHONPATH=src python benchmarks/bidegree_efficiency.py \
    --preset smoke --cases matched6 --modes outer-jit
```

Run the standard benchmark and retain machine-readable results with:

```console
PYTHONPATH=src python benchmarks/bidegree_efficiency.py \
    --preset standard \
    --output benchmarks/results/bidegree_efficiency.json
```

The harness reports the following phases separately:

- core and kernel setup;
- whole-call lowering and compilation in `outer-jit` mode;
- the first synchronized execution;
- synchronized warm executions.

`direct` mode calls `tensordev.path_signature` or `tensordev.vsig` without an
outer JIT boundary, so its first call includes any compilation performed by
the implementation. `outer-jit` mode lowers and compiles the complete public
call explicitly before timing its execution. Every array leaf in the returned
PyTree is blocked before a timer stops.

For outer-JIT rows, the output also includes XLA cost and memory analysis,
StableHLO size, retained-coordinate counts, and actual output payload. XLA's
raw scan/while cost is preserved. The separately labelled `scan_scaled` fields
multiply that raw cost by the step count because the CPU cost model commonly
reports a loop body once; these fields are diagnostic estimates, not measured
timings.

For cold-start comparisons, `--clear-caches` clears JAX caches before every
row. Leave it disabled when measuring the normal behavior of multiple calls in
one process. Record the generated JSON metadata, JAX version, backend, device,
and dtype with any reported numbers.

# NVIDIA wordwise benchmark

`wordwise_gpu.py` validates and compares the alpha wordwise kernels with
portable JAX. It requires a JAX-visible NVIDIA GPU with CUDA compute capability
8.0 or newer. Public signature and scalar-FSSK calls opt in with
`execution="wordwise"`; automatic wordwise selection is disabled.

Run a small validation matrix with:

```console
PYTHONPATH=src python benchmarks/wordwise_gpu.py \
    --workloads ordinary fssk \
    --families total-standard bidegree-partial-standard \
    --fssk-outputs state readout \
    --warmups 5 --repeats 30 \
    --output benchmarks/results/wordwise_gpu.json
```

The JSON records synchronized native and portable timings, paired bootstrap
confidence intervals, correctness errors, plan sizes, and the device metadata
needed to reproduce the result. Native differentiation is not benchmarked;
transformed public calls continue to use portable JAX.
