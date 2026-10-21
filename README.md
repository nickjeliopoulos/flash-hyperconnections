# flash-hypercons

Small batched CUDA matrix exponentials for PyTorch, with an optimized backward.
The initial implementation uses an adaptive degree-18 Taylor approximation with
scaling and squaring, FP32 arithmetic, and a structured Frechet derivative.

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

## Development

```bash
python -m pytest -q
python -m pip install ruff
ruff check .
python benchmarks/matrix_exp.py --output results/benchmark.json
```

GPU tests verify dispatch as well as numerical results. Tests use FP64 native
references and the numerical criteria from the integrated implementation.
Benchmarks time the direct API, including allocations, conversions, and autograd overhead; compilation
and warmup are excluded. Results record the hardware and software environment.
See [VALIDATION.md](VALIDATION.md) for the A100 results, independent installation
checks, extraction comparison, and remaining H100 and multi-GPU checks.

Python implementations live under `flash_hypercons/ops/matrix_exp/`; CUDA sources
live under `csrc/`. Additional implementations can use the same public entry
points without introducing a backend registry.

## Provenance

The CUDA implementation was extracted from the local PyTorch small-matrix
exponential integration. `provenance.json` identifies its base commit, exact
source hashes, and pinned CUTLASS revision. Kernel arithmetic and launch choices
are retained. The polynomial follows Bader, Blanes, and Casas (2019),
[Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation](https://doi.org/10.3390/math7121174).
See `LICENSE` and `licenses/` for attribution and redistribution notices.
