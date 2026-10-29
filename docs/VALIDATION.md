# Validation record

Validated on 2026-10-21 using Python 3.12.13, **stock PyTorch 2.14.1+cu130**
(commit `5c4886908584029761b579af026dcfb627c84070`), CUDA toolkit 13.1.0,
NVIDIA driver 595.91.07, and an NVIDIA A100 80GB PCIe on `gilbreth-k023`.
The extension was compiled for `8.0;9.0+PTX`, with `MAX_JOBS=4`.

## Correctness and integration

`python -m pytest -q`: **396 passed, 1 skipped**. The skip requires two GPUs;
the allocation exposed one. PyTorch emitted a `torch.jit.script` deprecation
warning while initializing its differentiation machinery.

Coverage includes all matrix sizes 2 through 32 and all three supported dtypes,
empty batches, padding, transposed/expanded/negative views, launch boundaries,
scaling boundaries, large rotations, nonfinite inputs, and a nondefault CUDA
stream. Forward and backward use the inherited per-matrix error criteria against
FP64 references: `max(abs(actual-reference)) / (1 + max(abs(reference)))`, with
tolerances 0.001/0.005 for FP32 forward/backward and 0.05/0.1 for FP16/BF16.

Routing tests inspect profiler events to verify custom-op execution and native
fallback. They cover imported aliases, nested scopes, exception restoration,
thread isolation, backward outside the scope, and direct ATen bypass. Integration
tests cover `torch.library.opcheck`, higher derivatives, forward-over-reverse AD,
nested function transforms, tensor subclasses, autocast, and parameters.
`torch.compile` tests verify eligible custom-op capture and compiled forward and
backward; transform tests verify native graph capture.

The device header matches the integrated source byte for byte after namespace
substitution. **105 cases match the existing integrated build bit for bit in
both forward and backward**, spanning N=2…32, FP32/FP16/BF16, transposed inputs,
and batches 3 and 129. The comparison loaded the separately installed
PyTorch `2.16.0a0+git2a59301` build. Its library hashes match the prior source-build
validation:

| Library | SHA256 |
|---|---|
| `libtorch_cpu.so` | `74a702ef2065de36f87c131f115f3f7f48e2223a094b73abe9adb03455edc2b1` |
| `libtorch_cuda.so` | `821bb028254fed78deca79da66361c3c036d51b5a2a1a6d0e885e176ba918da5` |

Ruff checks and formatting checks pass. No PyTorch source build was needed to
build or test this package.

## Packaging

The editable install built successfully with `--no-build-isolation --no-deps`
against the preinstalled stock PyTorch and test dependencies. Independent builds
from a clean source export and from its sdist also succeeded. Each installed
copy passed **396 tests, with the same one-GPU skip**.

The clean source export contained no compiled extension, build directory, or
Python cache. It carried the pinned CUTLASS headers locally. Builds and tests ran
in temporary directories under Bubblewrap, with the original `flash-hypercons`,
PyTorch source, and FLA repositories hidden. Import paths were checked to ensure
tests loaded the independent installation. The sdist was built through
`setuptools.build_meta.build_sdist`; both installations used
`pip install --no-build-isolation --no-deps --target ...`.

The GPU test sandbox retained the node's writable runtime mounts. An earlier
read-only-root sandbox could build and import the extension but prevented CUDA
initialization; its CPU-only test run is not counted as GPU validation. The GPU
runs additionally emitted a native PyTorch kernel-cache-directory warning.

## Performance

The benchmark measures the direct `matrix_exp` API, including Python selection,
allocations, conversions, and autograd. It reports median per-call wall time
over three trials of ten calls, with five warmup calls and synchronization at
trial boundaries. Backend order alternates within trials. FP16/BF16 stock
comparisons compute in FP32 and cast back, including the casts in the timing.
Input and upstream-gradient hashes match across the two installed PyTorch builds.

```bash
python benchmarks/ops/matrix_exp.py \
  --n 4 8 16 32 --batch 1 128 65536 --dtype fp32 fp16 bf16 \
  --output results/benchmark-a100.json

# With the existing integrated PyTorch build's Python executable:
python benchmarks/ops/matrix_exp.py --backends torch \
  --n 4 8 16 32 --batch 1 128 65536 --dtype fp32 \
  --output results/benchmark-integrated-a100.json
```

Representative FP32 results at B=65,536, in microseconds:

| N | Operation | Stock 2.14.1 | flash-hypercons | Integrated 2.16 build | Speedup vs stock |
|---:|---|---:|---:|---:|---:|
| 4 | Forward | 1,406.51 | 22.83 | 12.82 | 61.61× |
| 4 | Forward + backward | 4,604.60 | 164.22 | 88.20 | 28.04× |
| 8 | Forward | 3,141.48 | 58.21 | 55.18 | 53.97× |
| 8 | Forward + backward | 10,974.34 | 208.52 | 199.41 | 52.63× |
| 16 | Forward | 7,659.91 | 370.30 | 341.20 | 20.69× |
| 16 | Forward + backward | 24,073.31 | 1,208.46 | 1,199.27 | 19.92× |
| 32 | Forward | 15,987.34 | 1,948.02 | 1,926.50 | 8.21× |
| 32 | Forward + backward | 82,186.25 | 8,723.40 | 8,504.83 | 9.42× |

The standalone package adds measurable overhead compared with the integrated
C++ route. For B=1, N=4, forward is 22.18 µs versus 7.69 µs, and forward plus
backward is 155.81 µs versus 90.49 µs. These are comparisons of different PyTorch
builds as well as different integration mechanisms; they do not isolate kernel
execution time. Scoped-mode overhead is not included. These short measurements
are local regression evidence, not a general performance guarantee.

The local ignored `results/` directory retains build/test logs, packaging reports,
extraction checks, and both benchmark JSON files. The JSON records software/GPU
metadata, extension and Python implementation hashes, input hashes, and every
timing trial.

## Remaining validation

**H100 runtime validation was not available.** Both nodes in the dedicated H100
partition were drained; the shared H100 partition required an account outside
this allocation. SM90 compilation succeeded, but that does not establish H100
runtime correctness or performance. The two-GPU device-guard test also remains
unrun. Broader version/platform compatibility and public wheels are not claimed.

## Benchmark plotting (2026-10-27)

On an A100-SXM4-40GB with stock PyTorch 2.14.1+cu130, **33 focused tests passed**:
15 benchmark/plotting checks and 18 rotation-regression cases through angle 1,000.
An end-to-end benchmark collected 48 rows for N=4/9, B=35, all three dtypes,
forward/forward-plus-backward, both backends, and skew-symmetric inputs with
target column norms 3 and 1,000. It generated all three plot types using the
test suite's normalized maximum-error metric. The final source hash matches the
report. Local data and figures are ignored under `.agents/forum/plotting-20261027/`
(`revised.json`, `revised-figures/`, Slurm job 12030994). These small runs validate
the plotting pipeline, not representative performance or tail-error statistics.

## Correctness-to-benchmark driver (2026-10-27)

The complete `python -m benchmarks.run matrix_exp` workflow passed on an
NVIDIA A100-SXM4-40GB using Python 3.12.13 and stock PyTorch 2.14.1+cu130
(Slurm job 12031058, exit 0). Its default correctness gate ran the numerical,
routing, and integration suites: **489 passed, 1 skipped**, including 467 passing
CUDA-marked tests. The skip requires two GPUs. Timing started only after pytest
finished successfully.

The subsequent benchmark recorded 48 rows for N=4/9, B=35, FP32/FP16/BF16,
forward/forward-plus-backward, and skew inputs with column norms 3/1,000.
Three timing trials and five precision samples per case exercised the pipeline;
these small counts do not establish representative performance or tail errors.
The driver saved the gate evidence and environment metadata in `report.json`
before exporting 36 PNG/SVG files across all three plot types. The canonical
report hash and all figure hashes were verified, and the figures were inspected.
Artifacts remain ignored under `.agents/forum/pipeline-20261027/run-12031058/`.

The **34 infrastructure tests passed** on CPU and in the A100 job. They cover
real pytest subprocess outcomes (failure, all-skipped, CPU-only, expected failure,
teardown failure, collection failure, and empty selection), stopping downstream
work, saving metadata before rendering, shared timing, and generic operator plot
metadata. PNG/PDF/SVG exports are covered. Ruff and whitespace checks also passed.


## Grouped benchmarks and unified visualization (2026-10-27)

After moving matrix-exp to `benchmarks/ops/` and combining the plotting modules
into `benchmarks/visualization.py`, **42 targeted CPU infrastructure and plotting
tests passed**. Three unchanged numerical-measurement tests were deselected;
input generation and numerical-measurement functions were checked against their
previous source. Tests cover group selection and metadata for ops/layers/modules,
report paths, the correctness gate, and all plot types and export formats.

The unified visualization entry point regenerated 36 PNG/SVG figures from the
saved A100 report with PyTorch imports explicitly disabled. The report hash and
all artifact hashes matched the manifest. Outputs are local under
`.reports/matrix_exp/visualization-check/`. No GPU timing or kernel behavior was
changed or revalidated for this organization change.

## Compact dtype summaries (2026-10-27)

**46 targeted CPU infrastructure and plotting tests passed**; three unchanged
numerical-measurement tests were deselected. Regression checks cover geometric
mean speedup bar heights, pooled precision samples and quantiles, combining
three dtypes and multiple families/workloads without extra pages, preserving raw
records, rejecting inconsistent summaries, and removing obsolete manifest-listed
figures while retaining unrelated files. Ruff and whitespace checks passed.

The saved A100 report was rendered again with PyTorch imports disabled: six
figures (three per mode), exported as 12 PNG/SVG files, now replace the previous
36 files in `.reports/matrix_exp/visualization-check/`. Report and artifact hashes
were verified and the forward speedup, CDF, and precision/speedup figures were
visually inspected. This reuses the earlier small skew-input measurement run;
it is a visualization check, not new GPU performance or kernel validation.

## Shared numerical utilities (2026-10-28)

The numerical assertion and matrix-exp benchmark now share
`flash_hypercons.testing.error_ratio`, preserving the normalized maximum-error
formula and existing forward/backward tolerances. **70 CPU tests passed** under
Python 3.12 / PyTorch 2.14.1+cu130, covering numerical helpers, benchmark evaluation,
plotting, and the benchmark pipeline. Checks include per-matrix reductions,
FP64 subtraction, complex inputs, nonfinite failures, empty reductions, shape
mismatches, and tolerance boundaries. Ruff and whitespace checks passed.
No GPU execution was available; kernel implementations were unchanged.

## Initial projected stream mixing implementation (2026-10-28)

Added `flash_hypercons.ops.stream_mix.stream_mix_project` with standalone CUDA
forward and backward for N=1..32. Its contract follows local
`hyperconnections/master` (`cf600d9f3f4a`) and `continuous-hc/master`
(`41ca9ae06735`); `expm-stream-kernel-v3` (`9dd5d6062db6`) supplies the compiler
prior. The operator supports fixed projection decay, independently typed
FP32/FP16/BF16 inputs, FP32 arithmetic, and output in the residual's dtype.
Uniform directions implement mean projection. Two-sided transitions and forced
sources are prepared by the caller; inflow/generator unprojected mixing remains
outside this API. Exact revisions and source hashes are in `provenance.json`.

On an A100 80GB PCIe (SM80, Slurm 12040946), with stock PyTorch 2.14.1+cu130,
Triton 3.8.0 and NVCC 13.1.80:

- **129 operator tests passed, 1 skipped**, including 10 comparisons with the
  captured v3/master Triton sources. Independent FP64 references cover outputs
  and all four gradients, decay, mixed dtypes, noncontiguous/expanded inputs,
  empty dimensions, grid-stride boundaries, and long reductions. Integration
  checks cover opcheck, CUDA forward/backward inside compiled graphs, higher
  derivatives and transforms, autocast, nondefault streams, and CUDA graph replay.
- **559 existing regression tests passed, 1 skipped**: 472 numerical, routing,
  numerical-helper and benchmark/pipeline tests, plus 87 PyTorch integration
  tests. Both skips require a second GPU. The integration run emitted one
  upstream `torch.jit.script` deprecation warning.
- **16 comparisons executing the actual master mixing method passed**, checking
  mean/learned directions, one/two-sided projection, four decay values, outputs,
  and gradients. These isolate the method; they are not end-to-end model tests.
- An independent wheel built from the source archive for SM80 and SM90+PTX
  passed six forward/all-gradient cases and a matrix-exp smoke check with
  Triton and both sibling packages made unavailable. The archive includes the
  operator, tests and benchmark, and excludes local forum/measurement artifacts.
  Runtime execution was validated only on SM80; H100 and multi-GPU behavior
  remain untested. The suggested `uvhypercon` environment was incomplete, so
  these runs used the existing `venvs/matexp-2141` environment.

Fixed per-batch normalized maximum-error tolerances were FP32 2e-5/1e-4,
FP16 2e-3/3e-3 and BF16 1e-2/2e-2 for forward/backward. Each gradient uses its
own input dtype's tolerance. An initial harness bug incorrectly applied the
residual's FP32 tolerance to a BF16 source gradient; correcting the dtype
selection did not change any tolerance.

The bounded forum campaign accepted two changes after the initial translation:
reuse each x column across all output rows in forward, and group parameter
gradient reductions by output row. Across 16 primary workloads (N=4/8/16/32,
B=128/1024, D=768, FP32/BF16), public forward + backward was **1.37–3.04x faster
than v3** (geometric mean 2.34x). Forward alone ranged from **0.88–1.83x**.
The final independently built wheel was also compared with master using FP32
transition/direction, BF16 sources, decay .99, and FP32/BF16 residuals:

| Workload, D=768 | Residual | Forward speedup vs master | Forward + backward speedup vs master |
| --- | --- | --- | --- |
| B=128, N=4/16 | FP32 | 1.60–1.62x | 2.41–2.43x |
| B=128, N=4/16 | BF16 | 1.58–1.60x | 2.53–2.54x |
| B=8192, N=4 | FP32 | 0.89x | 0.91x |
| B=8192, N=4 | BF16 | 0.65x | 0.76x |

Values below one are regressions: the B=8192 backward-inclusive latencies were
860 vs 786 microseconds for FP32 and 671 vs 509 microseconds for BF16. These
results do not support a universal speedup claim. Public timings include casts,
allocations and autograd, with five alternating-order trials of 50 calls after
warmup/autotuning. Separate CUDA-graph forward timings exclude host dispatch;
they are not interchangeable with public-call timings.

The final correctness → benchmark → plotting pipeline produced 24 measurement
rows and four PNG/SVG exports. Source, extension, report and figure hashes were
verified, and both figures were inspected. Ruff and whitespace checks passed.
Local evidence, compiler outputs, trial binaries and full timing distributions
are under `.agents/forum/stream-mix-projection-ptx/`; `KERNEL.md` records the
selected implementation and remaining performance limits. These artifacts are
ignored; reusable code, tests, benchmark and this validation record are in the
repository's normal layout.

## Large-batch stream mixing optimization (2026-10-28)

The resumed forum campaign tested four CUDA candidates (004–007) and selected
007, retaining the master-reconciled equations, dtype contract and all fixed
tolerances above. It vectorizes forward loads for N=4/8, fuses projection
moments with input gradients, and derives both parameter gradients from one
`H = G @ X.T` contraction. Padded shared-memory layouts reduce bank conflicts.
For N=16/32, FP32 uses the vendored CUTLASS SIMT GEMM; aligned FP16/BF16 operands
use WMMA with FP32 accumulation. Tensor-core operands are the already stored
X/G values. FP32 coefficients and intermediates are never converted to TF32.
Shape, alignment and architecture checks retain scalar/shared-memory fallbacks.

The complete primary grid was N=4/8/16/32, B=128/512/2048/8192/16384, D=768,
FP32/BF16 residuals, FP32 transition/direction, BF16 sources and decay .99.
Both the starting implementation (003) and final independent wheel were measured
on the same A100 80GB PCIe, with one fixed CPU affinity, one PyTorch CPU thread,
20 warmups and five alternating-backend trials of 30 synchronized public calls.
Input and upstream-gradient hashes match for every comparison. Timings include
casts, allocations and all four gradients; the table reports milliseconds.

| B=16384, D=768 | FP32 before → after | Speedup | BF16 before → after | Speedup |
| --- | --- | --- | --- | --- |
| N=4 | 1.672 → 1.093 | 1.53× | 1.298 → 0.662 | 1.96× |
| N=8 | 3.920 → 2.180 | 1.80× | 3.146 → 1.345 | 2.34× |
| N=16 | 11.402 → 5.181 | 2.20× | 8.964 → 2.954 | 3.04× |
| N=32 | 34.233 → 10.370 | 3.30× | 34.231 → 7.162 | 4.78× |

Across all 40 workloads, forward + backward improved **1.02–4.78×**, geometric
mean **1.83×**. Forward alone ranged from **0.996–1.40×**, geometric mean
**1.073×**; no measured case regressed by more than 0.4%. Small-call timings in
earlier trial processes showed host scheduling anomalies affecting both CUDA
and PyTorch backward, so the final comparison reran both builds with the same
explicit CPU affinity. Raw anomalous trials remain archived and are not the
basis of these claims. Separate device profiles reduced N32/B8192/BF16 backward
from 15.06 ms to 1.89 ms; those graph timings exclude forward and host dispatch.

The final wheel passed **190 operator tests, one multi-GPU skip**, including ten
captured Triton-source comparisons. Added coverage exercises the optimized
N=16/32 paths, FP16 tensor contractions, independent unaligned X/Y/G storage,
projection cancellation, large-batch tails, and graph replay after changing
inputs. The source-archive wheel was independently built for SM80 and SM90+PTX
and passed six output/all-gradient cases plus a matrix-exp smoke check with
Triton and both sibling packages unavailable. Runtime coverage remains A100
SM80 only. The gate emitted an upstream JIT deprecation warning and a PyTorch
AccumulateGrad stream-lifetime warning; graph correctness checks passed.

The existing regression suites passed **559 tests with one multi-GPU skip**,
bringing this validation to **749 passes and two hardware skips**. A further
32-workload timing sweep covered B=256/4096 and D=256/2048 for all four stream
counts and both residual dtypes; its 128 rows are in `large-batch/holdouts.json`.
These holdouts compare the selected kernel with PyTorch, not the old CUDA build.

The optional master-dispatcher comparison exposed a separate upstream failure:
at N16/B8192/D768, its big-NB backward calls `einsum(v, x)` with FP32 v and BF16
x and raises `expected scalar type Float but found BFloat16`. Its partial report
is preserved as an incomplete reference run, not a full-grid comparison. The
completed final pipeline compares CUDA with promoted PyTorch; the before/after
claims above compare the two CUDA implementations. The master dispatcher and
dependencies were captured at the recorded master revision; no sibling source
was changed. This also corrects the initial campaign's use of the small-NB
kernel directly as its master performance reference.

The unmodified dispatcher diagnostic passed output/all-gradient checks for
FP32 residuals at N16/B4096 and N32/B2048, and reproduced the mixed-BF16 backward
failure at both shapes. No BF16 numerical agreement is inferred from these
FP32 checks.

`torch.compile` was inspected and checked at representative N4/32 workloads.
Its generated forward uses two FP32 batched GEMMs plus conversion/intermediate
buffers; backward uses three batched GEMMs and four generated kernels. This
guided the shared contraction, but the measured compiled prototype was slower
and was not added to production dispatch. The reusable benchmark now accepts
`--compiled` and a local master dispatcher path.

Evidence is under `.agents/forum/stream-mix-projection-ptx/large-batch/`:
`final-summary.json`, `baseline/pinned-public.json`, `final-pipeline/`, trial
sources/profiles and package artifacts. The complete pipeline has 160 timing
rows and four PNG/SVG exports; both figures were inspected, and source,
extension, canonical report and figure hashes were verified. Current CUDA
SHA256 is `3b28df38258bb6a2ad7ab31f264d8e00b45bf082874eb8c95e16a230ed0eef07`;
the final wheel extension is
`04ae4010007367605e4c197fd6da067fa2a71bf1ce5fdc4f775d38e59dd5f719`.
The four-candidate optimization batch is complete. N16/32 forward execution,
additional feature widths and H100 execution remain useful future measurements.
