# Validation record

Current exp/mix policy: the experimental fused source is deprecated and
unregistered; the public helper composes separate operators for every dtype.
See [the deprecation record](#fused-expmix-deprecation-2026-10-29). Earlier
fusion measurements below describe their recorded implementations.

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

## N=4 exp/project/impulse forward fusion (2026-10-28)

Added `matrix_exp_stream_mix_project` in `flash_hypercons/ops/expm_stream_mix.py`, backed by
`csrc/expm_stream_mix.cu`. This fuses the adaptive small matrix exponential,
one-sided projected mixing, and the rank-one impulse source for N=4. Phi stays
in shared memory; the source is formed in registers. It preserves explicit
transition and impulse operand/result rounding, accepts the supplied perturbed
mean/V direction, and expects A to contain dt already. The precision contract
follows the CGHC m=1 path in the recorded master branches; no integral is computed.
No sibling model/layer code was changed.

This is a **forward-only optimization**. With autograd enabled and any input
requiring a gradient, the public function uses the existing differentiable
unfused operators. CPU, other N, FP64 and function transforms also use that
composition. There is no fused backward or training/full-model throughput claim.

Three launch layouts were evaluated on the allocated A100 80GB PCIe SM80 with
stock PyTorch 2.14.1+cu130, Python 3.12, and CUDA toolkit 13.1.80. The selected
kernel uses 128 threads per item generally, and 64 for BF16 residuals with
B>=8192 and D=768. Four items per block did not consistently improve on two and
hurt small batches. The selected vector paths use **56 registers/thread**, no
stack/local memory or register spills, and 128/256 bytes of shared memory on
SM80. Resource usage was checked in the final wheel, not only a trial build.

The full benchmark covers B=256,512,1024,2048,4096,8192,16384,21920,24576 at D=768,
with FP32 or BF16 residuals; A/W/Z/V are FP32 and transition/source rounding is
BF16. Synthetic generators are bounded skew matrices plus a negative diagonal;
V is a shared, normalized, perturbed mean. The baseline composes the existing
CUDA exp/mix operators with statically compiled intervening casts/source,
using `emulate_precision_casts=True`. Default Inductor can remove BF16 operand
rounding. An initial dynamic-shape compiled baseline was also slower because of
runtime integer division; it is preserved but excluded from the final comparison.

Representative CUDA graph replay latencies, in microseconds:

| B | Residual | Unfused compiled | Fused | Speedup |
|---:|:---|---:|---:|---:|
| 256 | FP32 | 17.050 | 5.530 | 3.08× |
| 256 | BF16 | 16.230 | 5.202 | 3.12× |
| 16,384 | FP32 | 443.351 | 278.354 | 1.59× |
| 16,384 | BF16 | 331.694 | 160.963 | 2.06× |
| 21,920 | FP32 | 587.377 | 369.920 | 1.59× |
| 21,920 | BF16 | 439.101 | 211.876 | 2.07× |
| 24,576 | FP32 | 655.708 | 413.573 | 1.59× |
| 24,576 | BF16 | 489.810 | 236.329 | 2.07× |

All 18 primary cases improved. Geometric mean graph speedups are **1.94× FP32**
and **2.38× BF16**; the respective ranges are 1.59–3.08× and 2.06–3.14×.
Measurements use ten warmups, ten captured calls replayed ten times per trial,
and seven event-timed trials with alternating backend order. They include the
GPU work of public calls, including conversions and intermediate/output memory
traffic; host dispatch and allocation overhead are excluded. Separate
synchronized public-call wall measurements include those host costs and also
improved every case: geometric means 2.45× FP32 and 2.90× BF16 on this host.
The generator, readout/module and backward are outside both timing scopes.
Additional D=256/2048 holdouts at B=256/8192 improved 1.51–3.67× versus the same
compiled baseline, using the general layout.

The final-wheel correctness gate passed **71 operator tests**. Coverage includes
all residual/mixing precision pairs, the BF16 cast boundaries, independent dense
FP64 exp/projector references, adaptive scaling/squaring, nonfinite isolation,
shared and nonunit directions, decay, noncontiguous/unaligned/negative inputs,
empty tensors, large batches, and odd batches around the dispatch boundary.
Five mixed-coefficient layout cases were also rerun with independent FP16/BF16
A/W/Z/V inputs against the final extension. Opcheck, dynamic fullgraph
compilation, changed-input graph replay on a
nondefault stream, forward AD, eager/compiled gradient fallback, and CPU FP64
first/second derivative checks passed. Tolerances were unchanged: normalized
per-batch forward error 2e-5 FP32, 2e-3 FP16 and 1e-2 BF16.

The existing suite passed **749 tests with two multi-GPU skips**, for **820
passes and two skips** total. Existing JIT-deprecation and AccumulateGrad stream
warnings remain; graph correctness checks passed. Ruff and whitespace checks
passed. An independent wheel built from the source archive for SM80 and
SM90+PTX passed six forward and three gradient-fallback cases with Triton and
both sibling packages blocked. Runtime validation is A100 only; H100 was
cross-compiled, not executed.

Reproduce with `python -m benchmarks.run expm_stream_mix -- --wall`. Raw evidence
is under `.agents/forum/expm-stream-mix-n4/`: `KERNEL.md`, trial snapshots,
`final-pipeline/` (108 timing rows and four inspected PNG/SVG exports),
`final-summary.json`, `holdouts.json`, `regression.log`, and `package/`.
Source, archive, wheel, imported extension, report and figure identities were
verified. Selected CUDA SHA256:
`4241f0d1aa99447d8e42c222109acb9e77c5c9dabb9bb49e833a7678307f1526`.
Final wheel extension SHA256:
`1ad6bebc6ef213eeecdcaa1bc1343084577de218cca38f21ac85340d782d2a9e`.
The three-candidate campaign is complete; backward fusion and model integration
remain separate future work.

## N=4/8/16/32 exp/project/impulse baseline (2026-10-29)

Evaluated the current implementation before changing any kernel, arithmetic,
precision policy, or dispatch. **Only N=4 has a fused CUDA implementation.**
The public `matrix_exp_stream_mix_project` call uses the existing eager CUDA
exp/mix composition at N=8/16/32. The benchmark now accepts `--n 4 8 16 32`,
labels the public call `current-op`, and records profiler-observed custom-op
dispatch outside the timed region. All 72 workload routes matched this contract.

The run used an A100 80GB PCIe on `gilbreth-k018`, Slurm job `12045071`, Python
3.12.13, stock PyTorch 2.14.1+cu130, and driver 595.91.07. The loaded extension
SHA256 was `1ad6bebc6ef213eeecdcaa1bc1343084577de218cca38f21ac85340d782d2a9e`,
identical to the previous final wheel. Production sources at revision `8e8d280`
were unchanged; no rebuild was performed.

The grid covers N=4/8/16/32; B=256,512,1024,2048,4096,8192,16384,21920,24576;
D=768; and FP32/BF16 residuals. A/W/Z/V are FP32, with explicit BF16 transition
and impulse operand/result rounding. Inputs use the previous bounded skew plus
negative diagonal distribution and a normalized, perturbed shared mean vector.
All N=4 input hashes match the previous benchmark. Larger N uses the same entry
distribution, which does not hold matrix norms constant across stream counts.

Three backends were measured on identical inputs: `current-op`, eager composition,
and static-shape compiled composition with `emulate_precision_casts=True`.
Ten warmups, ten captured calls, ten replays per trial, and seven trials with
alternating backend order produced 432 timing rows. CUDA graph timing includes
casts and intermediate/output memory traffic, excluding host dispatch and
allocation costs. Separate synchronized wall timing includes those costs.
Both modes use `no_grad`; generator construction, the model, and backward are
outside the measurement scope.

Geometric mean CUDA graph speedup is compiled-composition latency divided by
current-op latency, equally weighted across the nine batch sizes. Values below
one mean the current public call is slower than the compiled baseline.

| N | Current execution path | FP32 speedup | BF16 speedup |
|---:|:---|---:|---:|
| 4 | Fused CUDA | 1.937× | 2.386× |
| 8 | Eager composition | 0.759× | 0.712× |
| 16 | Eager composition | 0.794× | 0.763× |
| 32 | Eager composition | 0.826× | 0.812× |

Representative graph latencies at B=16,384 and D=768, in microseconds:

| N | Residual | Current op | Compiled composition |
|---:|:---|---:|---:|
| 4 | FP32 | 278.671 | 443.361 |
| 4 | BF16 | 160.901 | 331.633 |
| 8 | FP32 | 1124.639 | 857.283 |
| 8 | BF16 | 930.642 | 661.555 |
| 16 | FP32 | 2559.918 | 2035.517 |
| 16 | BF16 | 2283.428 | 1762.714 |
| 32 | FP32 | 5877.371 | 4853.258 |
| 32 | BF16 | 5534.617 | 4514.662 |

N=4 improves every measured case, reproducing the previous fusion result.
For N=8/16/32, graph latency tracks the eager baseline; the compiled composition
is 1.21–1.40× faster by geometric mean. This establishes a baseline for future
larger-N fusion, without measuring or predicting its benefit. Maximum within-case
IQR/median for current-op graph timing was 4.0%; all seven samples are retained.
Wall timings yield N=4 geometric mean speedups of 2.469× FP32 and 2.927× BF16;
the larger-N public fallback is slower than compiled composition in every case.

The full operator gate passed **80 tests, including 73 CUDA-marked tests**, with
no skips. Expanded coverage checks dense FP64 references and scaling at all four
N values, and the compiled baseline plus actual public dispatch at B=24,576 for
both primary residual dtypes. Existing tolerances remain unchanged. The one
existing TorchScript deprecation warning remains. **49 benchmark infrastructure
tests** also passed. A plot legend overlap found during inspection was corrected
in the shared renderer, then its tests were rerun and the saved report re-rendered;
no GPU measurements were rerun or modified for this presentation change.

Reproduce the correctness → measurement → plot pipeline with:

```bash
python -m benchmarks.run expm_stream_mix \
  --output-dir .reports/expm_stream_mix/n4-n32-20261029 --formats png svg \
  -- --n 4 8 16 32 --wall
```

Raw evidence is in `.reports/expm_stream_mix/n4-n32-20261029/`: `report.json`,
`correctness.json`, logs, `summary.csv`, `summary.json`, and four PNG/SVG exports.
The shared plots aggregate N values; use the per-N table above and the CSV for
the distinction between fused and fallback performance. Environment identity,
the exact launch script, and the summary/coverage checker are retained under
`.agents/forum/expm-stream-mix-n4/baseline-n4-n32-20261029/`.
`summarize.py` there regenerates the CSV/JSON summaries and verifies all pairings,
input hashes, dispatch records, and expected grid coverage. The allocation was
released after the completed sweep. No kernel optimization was attempted.

## Fused exp/project/impulse forward through N=32 (2026-10-29)

Extended `matrix_exp_stream_mix_project` from N=4 to **every N=2…32** on NVIDIA
CUDA. Non-power-of-two matrices pad locally to 4/8/16/32. The existing tiny
shuffle exponential handles padded sizes through 8; sizes 16/32 reuse the
standalone cooperative t18 polynomial with adaptive scaling and squaring.
Rounded Phi stays in shared memory. Mixing caches X columns in registers and
computes output rows sequentially, forming the rounded rank-one source locally.
There is no global Phi/Y intermediate. The prior N=4 arithmetic and launch
policy are retained.

This remains a **forward-only optimization** with the existing supplied mean/V
direction, projection decay and impulse precision contract. A already contains
dt. The source is the layer-output contribution to residual streams; this does
not implement `TemporalWriteback` or integral forcing. Gradient-enabled inputs,
function transforms, CPU/FP64, N=1 and N>32 retain the differentiable composition.
No module/layer or fused backward implementation was added.

The bounded campaign evaluated three mixing layouts. Streaming X into all
output accumulators used 240 registers/thread for N=32 and was substantially
slower than compiled composition. Distributing output rows across thread groups
reduced registers but duplicated X loads and still regressed. The selected third
layout caches X and processes rows sequentially. Its FP32 vector paths use
94/78/95 registers per thread for padded N=8/16/32, with 384/3332/12804 bytes of
shared memory on SM80. The N=32 scalar path for odd D or unaligned inputs still
uses 255 registers; it is covered for correctness, without a throughput claim.
These paths have zero stack/local memory in the final wheel's resource report.

Final validation used A100 80GB PCIe on `gilbreth-i001`, Slurm job `12045540`,
Python 3.12.13, stock PyTorch 2.14.1+cu130, CUDA toolkit 13.1.80, and driver
595.91.07. The primary grid and precision-preserving compiled baseline are
identical to the preceding baseline section: N=4/8/16/32, nine B values from
256 through 24,576, D=768, FP32/BF16 residuals, FP32 A/W/Z/V, and BF16 transition
and source rounding. All input hashes match that baseline. Profiler dispatch
confirmed the fused custom op, without separate exp/mix custom ops, for all 72
workloads. Timings compare identical inputs in a fresh process using the final
wheel extension, with no concurrent GPU benchmark.

Geometric mean CUDA graph speedups versus compiled composition, equally
weighted across the nine batch sizes (values below one indicate a regression):

| N | FP32 speedup | FP32 range | BF16 speedup | BF16 range |
|---:|---:|:---|---:|:---|
| 4 | 1.937× | 1.586–3.087× | 2.390× | 2.066–3.177× |
| 8 | 1.723× | 1.579–2.164× | 1.669× | 1.565–2.078× |
| 16 | 1.491× | 1.231–1.633× | 1.433× | 1.225–1.515× |
| 32 | 1.029× | 0.777–1.144× | 0.993× | 0.862–1.028× |

N=4 reproduces the earlier result. N=8/16 improve every primary case. **N=32
does not provide a uniform improvement over compiled composition**: the large
batches are near parity, while B=256 is slower. Representative N=32 graph
latencies, in microseconds:

| B | Residual | Fused | Compiled composition | Eager composition |
|---:|:---|---:|---:|---:|
| 256 | FP32 | 127.857 | 99.287 | 118.897 |
| 256 | BF16 | 102.779 | 88.627 | 108.872 |
| 16,384 | FP32 | 4632.013 | 4829.757 | 5860.496 |
| 16,384 | BF16 | 4444.017 | 4485.315 | 5497.231 |
| 24,576 | FP32 | 6943.447 | 7243.110 | 8788.961 |
| 24,576 | BF16 | 6658.611 | 6746.562 | 8264.642 |

Against eager composition (the former public fallback for N>4), N=32 geometric
mean graph speedups are 1.244× FP32 and 1.222× BF16; B=256 FP32 still regresses.
Seven event trials retain the same ten-warmup, ten-captured-call, ten-replay
protocol and alternating backend order. Maximum current-op graph IQR/median
is 4.1%. Graph timing includes public-call GPU conversions and memory traffic,
excluding host dispatch/allocation costs. Separate synchronized wall timing
includes those costs: FP32/BF16 geometric mean speedups are 2.448×/2.903× (N=4),
2.203×/2.245× (N=8), 1.832×/1.846× (N=16), and 1.085×/1.067× (N=32).
Generator construction, the model and backward are outside both timing scopes.

Additional five-trial graph holdouts cover N=8/16/32, B=256/8192, D=256/2048 and
both residual dtypes on the same validated wheel. Speedup ranges versus compiled
composition are 1.174–2.765× (N=8), 0.870–1.771× (N=16), and 0.658–1.249×
(N=32). In particular N=32, B=256, D=2048 is slower: 292.475/289.004 us fused
versus 213.074/190.198 us compiled for FP32/BF16. Feature width and batch size
matter; these measurements do not establish an optimal launch policy for every
supported shape. Testing a separate small-batch/wide-feature layout is a useful
next experiment, outside this completed three-candidate campaign.

The final-wheel gate passed **515 operator tests, including 508 CUDA tests**,
with no skips. Coverage includes all residual/mixing precision pairs at primary
N, every intermediate N, FP64 dense references, scaling/squaring, nonfinite
batch isolation, shared/nonunit V, decay, empty/layout/unaligned/negative inputs,
autocast, and B=24,576 comparisons with compiled composition. Opcheck, fullgraph
dynamic compilation with changed N/B/D, CUDA graphs on a nondefault stream,
inference mode, transform fallbacks, eager/compiled gradient fallback and CPU
FP64 first/second derivatives passed. Forward tolerances remain normalized
per-batch error 2e-5 FP32, 2e-3 FP16 and 1e-2 BF16.

The remaining suite passed **749 tests with two multi-GPU skips**, totaling
**1264 passes and two skips**. Existing TorchScript-deprecation and AccumulateGrad
stream warnings remain. An independent sdist-to-wheel build, installed away
from the source package, passed **24 forward and 21 gradient-fallback cases**
with Triton and both sibling packages blocked. Runtime validation is A100 only;
SM90+PTX was built but not executed. Integration review found no actionable
contract defects in the extension. Ruff, whitespace and the 54-path manifest
coverage checks passed.

Reproduce the full correctness → measurement → plot pipeline with:

```bash
python -m benchmarks.run expm_stream_mix \
  --output-dir .reports/expm_stream_mix/fused-n2-n32-20261029 --formats png svg \
  -- --n 4 8 16 32 --wall
```

That directory contains the 432-row report, full correctness gate, CSV/JSON
summaries, and four inspected PNG/SVG exports. Shared plots aggregate all N;
the table above and case-level CSV preserve the N=32 regressions. Trial sources,
rejected measurements, selected `KERNEL.md`, integration review, launch/build
scripts, final resource usage, regression log, holdouts and standalone package
evidence are under `.agents/forum/expm-stream-mix-n32/`. Source/archive/wheel,
loaded extension, paired inputs, complete grid and figure identities were
checked. Selected CUDA SHA256:
`799f6fe6ff42dead3d5d3715a87bd7014dcce658037492e8e4cc1076d51f0cd9`.
Final wheel extension SHA256:
`63cf6cba8bb6a4a836500c5e1df1ca8285590cfd8261ad35a3bddb15f327d27b`.
The completed GPU allocation was released.

## Exp/project/impulse backward through N=32 (2026-10-29)

Added ordinary CUDA backward to `matrix_exp_stream_mix_project` for every
N=2…32, independently supporting FP32/FP16/BF16 inputs and mixing dtype. Training
forward saves the mixing-rounded Phi in contiguous FP32 storage, while inference
keeps its previous allocation policy. The auxiliary costs `4*B*N*N` bytes and is
marked nondifferentiable; its unused gradient is not materialized. Original
inputs remain saved for version checking and differentiable higher-order formulas.
The public signature, mean/V projection, decay and impulse semantics are unchanged.

Backward reuses the structured augmented exponential to compute
`dA = L_exp(A.float().T, dPhi.float())`. The small cross moment `H = G X.T`
provides `r = H v`, `dPhi = H - r v.T`, and the direction gradient. It preserves
the mixing-dtype cast on dPhi before the exponential adjoint, the source-gradient
operand/product/reduction casts, and per-item V-gradient rounding before reducing
a shared V. State gradients return in X's dtype; every other gradient returns in
its corresponding input dtype. No global source Y or full source-gradient tensor
is materialized by this training route.

For N=4/8/16 and B>=32, the selected backward uses three core kernels: the
existing H contraction, combined state/source gradients, and parameter-gradient
finishing fused into the paired exponential. This avoids global dPhi and its
transpose/cast intermediates. Input preparation and a shared-V reduction can add
launches. N=32, intermediate N and small B reuse the existing stream-mixing and
Fréchet kernels with a fused impulse-gradient kernel. **N=32 does not use the
more deeply fused backward layout**, because that layout measured slower.

The three-layout campaign first established the separate-stage native path, then
combined state/source and parameter/Fréchet stages. The second layout caused
large stack allocations at N=16/32 (9344 bytes/thread for N=32 FP32 V2) and was
rejected there. Limiting output-row unrolling removed those allocations in the
third layout and improved N=4/8/16, but N=32 still favored the first layout.
The selected final-wheel FP32 state/source paths use 96/92/115 registers per
thread and 192/512/1536 bytes of shared memory at N=4/8/16, with zero stack/local
memory. The respective parameter/Fréchet kernels use 64/168/56 registers and
0/0/4292 shared bytes, also with zero stack/local memory. These counts describe
the listed SM80 paths, not every dtype, tail or architecture.

Final execution used A100 80GB PCIe on `gilbreth-k047`, Slurm job `12045911`,
Python 3.12.13, stock PyTorch 2.14.1+cu130, CUDA toolkit 13.1.80 and driver
595.91.07. The independently built wheel targets SM80 and SM90+PTX; execution
validation is A100 only. The benchmark uses N=4/8/16/32; B=256,512,1024,2048,
4096,8192,16384,21920,24576; D=768; FP32/BF16 residuals; FP32 A/W/Z/V; and BF16
mixing. All five inputs require gradients. Every backend receives identical
inputs and a fixed standard-normal upstream gradient in X's dtype. Fréchet
scaling depends on gradient magnitude, so other loss/gradient scales can change
the cost balance.

The baseline is static-shape compiled composition of the existing CUDA exp/mix
operators with `emulate_precision_casts=True`; eager composition is also measured.
CUDA graphs include the public forward and backward GPU work, Phi stores,
conversions and intermediates, excluding host dispatch/allocation costs. Separate
synchronized wall measurements include host costs. Both scopes exclude generator
construction, the model and optimizer. These are **forward-plus-backward** results,
not backward-only or full-model training throughput.

Geometric mean graph speedups over the nine batch sizes, versus compiled
composition (values below one mean slower):

| N | FP32 residual | FP32 range | BF16 residual | BF16 range |
|---:|---:|:---|---:|:---|
| 4 | 1.547× | 1.429–1.973× | 1.548× | 1.412–1.994× |
| 8 | 1.459× | 1.404–1.714× | 1.319× | 1.236–1.545× |
| 16 | 1.152× | 1.069–1.190× | 1.055× | 1.013–1.252× |
| 32 | 0.992× | 0.852–1.028× | 0.910× | 0.786–0.953× |

N=4/8/16 improve all primary cases. **N=32 remains a limitation versus compiled
composition**, particularly BF16 and small batches. The selected implementation
improves over the former public eager training route: geometric means are
2.303×/2.442× (N=4), 2.102×/2.029× (N=8), 1.587×/1.579× (N=16), and
1.315×/1.242× (N=32), respectively FP32/BF16. This does not establish a universal
benefit over a compiled model's existing composition.

Representative graph latencies at B=24,576, D=768, in microseconds:

| N | Residual | Current op | Compiled composition | Eager composition |
|---:|:---|---:|---:|---:|
| 4 | FP32 | 1412.782 | 2018.232 | 3051.039 |
| 4 | BF16 | 1032.724 | 1467.126 | 2396.846 |
| 8 | FP32 | 2835.722 | 4002.734 | 5915.587 |
| 8 | BF16 | 2364.703 | 2968.873 | 4693.207 |
| 16 | FP32 | 8210.032 | 9635.031 | 13430.610 |
| 16 | BF16 | 6447.739 | 6682.900 | 10126.479 |
| 32 | FP32 | 22586.428 | 23128.699 | 30695.701 |
| 32 | BF16 | 19906.201 | 18919.352 | 25777.439 |

At N=32, B=256, FP32/BF16 current-op latencies are 392.694/346.788 us versus
compiled 334.623/272.404 us. Seven trials retain the ten-warmup, ten-captured-call,
ten-replay protocol and alternating backend order. Maximum current-op graph
IQR/median is 1.6%. Wall geometric means versus compiled are 1.710×/1.721× (N=4),
1.630×/1.540× (N=8), 1.308×/1.246× (N=16), and 1.037×/0.981× (N=32).

The final-wheel correctness gate passed **886 tests, including 879 CUDA tests**,
with no skips. Coverage includes every N=2…32 and residual/mixing dtype pair,
independent dense FP64 gradients, skew-generator norms through 1000, low-precision
coefficients, nonunit/shared V, large B, layouts, nonfinite isolation and empty
inputs. The empty-D/nonfinite-A case retains the prior composition's NaN A
gradient. Opcheck, compiled gradients, changed-N/B/D dynamic compilation, graphs
on a nondefault stream, higher derivatives, forward-over-reverse and selective
gradients passed. The compiled baseline's gradients were checked explicitly.
Forward tolerances remain 2e-5/2e-3/1e-2 for FP32/FP16/BF16. Backward uses the
existing stream criteria 1e-4/3e-3/2e-2, considering both input dtype and the
explicit intermediate gradient-rounding dtype for dA/dW/dZ. No threshold was
relaxed after an experiment failed.

The remaining suite passed **749 tests with two multi-GPU skips**, totaling
**1635 passes and two skips**. Existing TorchScript-deprecation and AccumulateGrad
stream warnings remain. The isolated source-archive-built wheel passed **27
forward and 27 backward cases** with Triton and both sibling packages blocked.
Higher derivatives use differentiable formulas with recomputed Phi; unsupported
devices/dtypes/N and function transforms retain composition. Partial-gradient
calls are correct but currently compute all derivative blocks.

Five-trial training holdouts at B=256/8192 and D=256/2048 show compiled-baseline
speedup ranges of 1.268–2.185× (N=4), 1.001–1.810× (N=8), 0.867–1.520× (N=16),
and 0.710–1.022× (N=32), across both residual dtypes. N=16 BF16 regresses at
B=256,D=2048 (343.982 us versus compiled 298.107 us) and B=8192,D=256
(961.741 versus 936.407 us). N=32 BF16 at B=256,D=2048 takes 706.867 versus
compiled 502.088 us. Width and batch size remain important; the primary D=768
improvement does not establish a win for every supported shape.

Inference checks at N=4/8/16/32, B=256/8192, D=768 on identical inputs reproduce
the prior forward performance overall: previous/current latency geometric mean
0.999×, range 0.959–1.013× across 16 cases, on separate A100 allocations. These
calls still use only the original inference op. A final profiler check verified
that dynamically compiled public training calls use the training and backward
custom ops at all four primary N values. Integration review found no actionable
contract defects. Ruff, whitespace and the 54-path manifest coverage checks passed.

Reproduce training measurements with:

```bash
python -m benchmarks.run expm_stream_mix \
  --output-dir .reports/expm_stream_mix/backward-n2-n32-20261029 --formats png svg \
  -- --n 4 8 16 32 --backward --wall
```

The directory contains the full gate, 432 timing rows, `report-summary.csv`,
`report-summary.json`, and four inspected PNG/SVG exports. The shared plots pool
all N; the table and CSV retain N=32 regressions. Trial sources/binaries,
rejected measurements, selected `KERNEL.md`, source snapshots, build scripts,
resource usage, integration review and standalone package evidence are under
`.agents/forum/expm-stream-mix-backward/`. Final paired input/gradient hashes,
complete grid, dispatch, source/archive/wheel/extension and figure identities
were verified. Selected CUDA SHA256:
`0252563536e2209052567387e76136343138fb3c991f63b25dfc326d94bd07ac`.
Final wheel extension SHA256:
`6fad714fdf4eb45c2798f6926609fe84a7a7bb7d2fca356fdbb541b45aa030f6`.
The GPU allocation was released after all checks. A useful next experiment is
feature tiling with partial source-weight reductions for N=32/small-B wide-D
workloads; no further tuning was started in this three-layout campaign.

## Fused exp/mix deprecation (2026-10-29)

The fused exp/mix experiment is deprecated by user request. Its previously
validated multi-dtype CUDA source remains in `csrc/expm_stream_mix.cu`, with a
DEPRECATED banner, but `setup.py` excludes it. The three fused schemas and CUDA
registrations were removed, along with their Python fake/autograd registrations.
The retained public `matrix_exp_stream_mix_project` signature now composes
`matrix_exp`, explicit transition/source casts, and `stream_mix_project` for
all dtypes. Existing standalone backward implementations handle differentiation.
Compilation remains caller-controlled; the helper does not compile implicitly.

This is a simplification decision, not evidence that compiled composition wins
all shapes. The historical primary forward-plus-backward sweep favored fusion
at N=4/8/16, while N=32 was near parity in FP32 and favored compiled composition
in BF16. Those historical fused timings no longer describe the public helper.
The benchmark still compares public eager composition with direct eager and
compiled compositions, recording actual dispatch.

`torch.compile` optimizes operations surrounding the standalone custom kernels:
casts, rank-one source construction and backward reductions. Generated code
retained in the local forum cache shows separate `_matrix_exp` and
`_stream_mix_project` calls and generated kernels around them. Separate stages
can use different launch layouts; a complete attribution of N=32 timing to
individual bottlenecks was not performed. The rejected fusion experiments did
show that combining more work could increase register pressure and latency.

CUDA graph replay is distinct from compiler fusion. In the installed PyTorch,
`reduce-overhead` enables CUDA graphs; the default compiler configuration does
not enable them unless requested through configuration/environment. Our
`measure_graph` explicitly captured and replayed **every backend**, so the
historical graph timing comparison gives compiled composition no exclusive
replay advantage. A replay covers the sequence of kernels and intervening
operations; it does not keep Phi in registers across separate kernels.

The new wheel targets SM80 and SM90+PTX. Runtime validation uses A100 80GB PCIe
on `gilbreth-k037`, Slurm `12049530`, stock PyTorch 2.14.1+cu130, Python 3.12.13,
CUDA toolkit 13.1.80 and driver 595.91.07. SM90 was built, not executed. The
installed wheel passes 27 standalone forward and 27 backward cases with Triton
and sibling packages blocked. All three deprecated op names are absent, and
binary resource inspection confirms no fused exp/mix translation unit. The
source remains in the source-distribution inventory via `MANIFEST.in`.

The installed-wheel exp/mix gate passed **884 tests, including 876 CUDA tests**,
with no skips. It covers FP32/FP16/BF16 composition, every N=2…32, explicit
precision boundaries, forward/backward numerical references, selective and
higher derivatives, transforms, dynamic compilation, nondefault-stream graphs,
and profiler evidence for both standalone operators. A regression test asserts
that none of the deprecated op names are registered. The remaining suite passed
**749 tests with two multi-GPU skips**, totaling **1633 passes and two skips**.
Existing JIT-deprecation and AccumulateGrad stream warnings remain. No numerical
tolerance dictionary was changed. Tests using BF16 inputs apply the existing
BF16 criteria and intermediate gradient-rounding rules.

The active installed Python sources and extension match the workspace. The
extension SHA256 is
`cfe035c25d62db06680a3c63c72754b250828e225b20ba6f5ff5b65d3c575e66`;
the wheel SHA256 is
`c2ed3e4272c40c95f8a951631c508e18a980e8e8c31079466e81d579cf473fd8`.
Build, installed-wheel checks, source/binary identity, test logs and resource
inspection are retained in
`.agents/forum/expm-stream-mix-backward/deprecated/`. The wheel build regenerated
the active native objects; no deprecated object was linked. Integration review
found no actionable defects. Ruff, whitespace and manifest coverage checks pass.
The allocation was released. No new timing claim is made for this deprecation.
