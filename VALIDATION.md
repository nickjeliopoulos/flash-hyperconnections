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
python benchmarks/matrix_exp.py \
  --n 4 8 16 32 --batch 1 128 65536 --dtype fp32 fp16 bf16 \
  --output results/benchmark-a100.json

# With the existing integrated PyTorch build's Python executable:
python benchmarks/matrix_exp.py --backends torch \
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
