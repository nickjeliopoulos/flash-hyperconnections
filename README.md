# flash-hypercons

A collection of useful GPU primitives and operators for Hyperconnections-like architectures.
Including matrix exponential, stream-mixing variants, and fused architectural blocks.
Native CUDA C/C++ implementations that extend or replace PyTorch operators.

## Install

Install a CUDA build of PyTorch and a compatible CUDA toolkit first. The initial
validation target is Python 3.12, stock PyTorch 2.14.1, CUDA 13.x, and A100/H100.
Build against your installed PyTorch:

```bash
git submodule update --init third_party/cutlass
python -m pip install setuptools wheel ninja
export CUDA_HOME=/path/to/cuda
export TORCH_CUDA_ARCH_LIST='8.0;9.0+PTX'
python -m pip install -e '.[test]' --no-build-isolation
```

The architecture list defaults to `8.0;9.0+PTX`; override it for your build.
Set `MAX_JOBS` to limit compilation parallelism. CUDA compilation happens during
installation. Source distributions include CUTLASS headers and do not require Git
or another source checkout. Rebuild the extension after changing PyTorch builds.

## Usage

```python
import torch
import flash_hypercons as fh

x = torch.randn(128, 4, 4, device="cuda", requires_grad=True)
y = fh.matrix_exp(x)
y.sum().backward()

with fh.use_backend():
    y = torch.linalg.matrix_exp(x)
```

`matrix_exp(input)` selects the CUDA implementation only for strided NVIDIA CUDA
tensors of shape `[B, N, N]`, with `2 <= N <= 32`, in FP32, FP16, or BF16.
Empty batches and noncontiguous inputs are supported. The output retains the
input shape, dtype, and device. Other shapes, dtypes, layouts, and devices use
native PyTorch, including its errors for unsupported inputs.

`use_backend()` also routes `torch.matrix_exp`, `Tensor.matrix_exp`, and previously
imported Python aliases. It is a nestable, thread-local scope that restores the
previous behavior on exit. Importing the package does not activate routing.
Direct ATen/C++ calls are unaffected. Backward can run after leaving the scope.

The ordinary backward uses the CUDA Frechet kernel. Higher derivatives use a
differentiable augmented-matrix calculation. Forward AD, function
transforms, and tensor subclasses use native operations, promoting CUDA
low-precision inputs to FP32 and casting results back. These paths are not
accelerated. This package does not claim complete PyTorch API compatibility.

For compilation, use the direct function or keep the scope around execution:

```python
compiled = torch.compile(lambda x: torch.linalg.matrix_exp(x))
with fh.use_backend():
    y = compiled(x)
```

## Projected stream mixing

```python
from flash_hypercons.ops.stream_mix import stream_mix_project

# phi: [B,N,N], x/y: [B,N,D], v: [B,N], normalized by the caller
out = stream_mix_project(phi, x, y, v, projection_decay=0.99)
```

This computes `phi @ x + (projection_decay*v - phi @ v) (v.T @ x) + y`,
independently for each batch item. `projection_decay` defaults to 1 and must
lie in [0,1]. A uniform `v = 1/sqrt(N)` selects the stream mean. Shared
directions must be explicitly expanded to `[B,N]`; the operator does not
normalize `v` or broadcast inputs.

The contract follows `hyperconnections/master` and the residual storage used
by `continuous-hc/master`. All four inputs may independently be FP32, FP16,
or BF16, on the same device; the result retains **x's dtype**. Computation
accumulates in FP32 and casts only the result, including when `y` has a
different dtype. CUDA forward and first backward accelerate N=1…32, including
noncontiguous/expanded inputs and empty batch or feature dimensions. Gradients
for `phi`, `x`, `y`, and `v` retain their respective input dtypes. CPU, FP64,
larger N, and function transforms use differentiable PyTorch operations; higher
derivatives use differentiable backward formulas. Triton is not a dependency.

The primitive accepts an already prepared transition and source. For the
two-sided variant, the caller supplies `(I-vv.T) @ phi`. Forced CGHC supplies
its already transformed source as `y`, which is added unchanged. The inflow
and generator variants in `hyperconnections` prepare a complete transition
and use **unprojected** mixing; this projected API must not be substituted at
those call sites. Model/layer integration remains in the calling packages.

Run its correctness, measurement, and plotting workflow with:

```bash
python -m benchmarks.run stream_mix -- --n 4 8 16 32 --batch 128 512 2048 8192 16384 --d 768
```

The benchmark optionally accepts `--triton-source /path/to/ops/stream_mix.py`
for the local master dispatcher, `--compiled` for a compiled PyTorch reference,
and `--mixed --decay .99` for mixed residual/source dtypes and decay. Keep the
dispatcher beside its sibling kernels. Reports include source/extension
identities and identical input and upstream-gradient hashes. Measured performance
and validation coverage are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

## Development and Testing

See [AGENT_GOVERNANCE.md](docs/AGENT_GOVERNANCE.md) for repository conventions covering
operators, tests, benchmarks, plots, and local experiment artifacts.
See [MANIFEST.md](MANIFEST.md) for the Git-tracked file inventory and refresh command.

```bash
python -m pytest -q
python -m pip install ruff
ruff check .
python -m benchmarks.run matrix_exp -- --precision
```

GPU tests verify dispatch as well as numerical results. Tests use FP64 native
references and the numerical criteria from the integrated implementation.
Shared numerical helpers live in `flash_hypercons.testing`: `absolute_error`,
`error_ratio`, and `assert_close`. The ratio is maximum absolute error divided
by `1 + max(abs(expected))`; pass `dim=(-2, -1)` to compare each matrix separately.
The assertion takes an explicit `tolerance` on that ratio.
Benchmarks time the direct API, including allocations, conversions, and autograd overhead; compilation
and warmup are excluded. Results record the hardware and software environment.
The benchmark driver runs correctness tests before measurements, saves their
outcome with the report, then generates plots from the saved metadata. Outputs
default to the ignored `.reports/matrix_exp/` directory, including `report.json`
and `figures/`. Install
the `benchmark` extra for plotting. `--precision` adds FP64-reference errors,
precision CDFs, and precision-versus-speedup plots alongside speedup bars.
Each mode has one figure of each type, combining dtypes and aggregating matrix
families and workloads. Bar labels include latency; heights show speedup.
Benchmarks are grouped under `benchmarks/ops/`, `benchmarks/layers/`, and
`benchmarks/modules/`. All plot types, shared styling, and report visualization
live in `benchmarks/visualization.py`. See the
[benchmark guide](benchmarks/README.md) for optional dependencies, input families,
CPU-only plotting of saved JSON, and the data format for future operators.
See [VALIDATION.md](docs/VALIDATION.md) for the A100 results, independent installation
checks, extraction comparison, and remaining H100 and multi-GPU checks.

Python implementations live under `flash_hypercons/ops/matrix_exp.py`; CUDA sources
live under `csrc/`. Additional implementations can use the same public entry
points without introducing a backend registry.
