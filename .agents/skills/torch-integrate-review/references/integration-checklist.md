# Integration checklist

Select checks according to the changed behavior and claimed support. Inspect the
current contracts before applying these matrix-exp-specific expectations to future
operators.

## Public calls and scoped routing

- `matrix_exp` and `use_backend` share the same selector. Importing the package
  does not replace or globally activate native PyTorch functions.
- `TorchFunctionMode` routing covers `torch.linalg.matrix_exp`, `torch.matrix_exp`,
  tensor methods, and already-imported aliases. Nested scopes, exceptional exits,
  other modes/subclasses, and thread-local behavior preserve the intended semantics.
- Direct ATen/C++ calls remain outside Python routing. Native fallback cannot
  recurse back into the accelerated wrapper. Backward works after leaving the scope.
- Eligibility checks preserve layout, rank, matrix bounds, dtype, and NVIDIA versus
  ROCm behavior. No accidental flattening of extra batch dimensions or silent
  expansion of kernel coverage. Invalid arguments retain appropriate native errors.
- Native transform/subclass fallbacks preserve differentiation and subclass
  behavior. Low-precision promotion remains differentiable, including when `vmap`
  hides batch dimensions. Compilation must not bypass these fallback decisions.

## Operator schemas, fake tensors, and compilation

- Schemas describe real mutation/aliasing behavior, and namespaced implementations
  register for the intended device. Input validation protects raw pointer access.
- Fake outputs match runtime shape, dtype, device, layout/strides, and aliasing
  without reading tensor values or launching work. Symbolic sizes do not acquire
  unnecessary concrete-value assumptions.
- Fake/autograd registration occurs after loading schemas. Registration and import
  behavior do not depend on a patched PyTorch tree or collide with ATen operators.
- Run `torch.library.opcheck` on representative public-to-custom paths and raw
  operations where relevant. Test compiled forward/backward and inspect graph or
  profiler evidence that eligible inputs actually use the custom operations.
- Exercise scoped execution and execution after scope exit where appropriate;
  compare against eager results. Distinguish successful graph capture, compilation,
  and execution. Review export/dynamic-shape support when changed or claimed.

## Autograd and transformations

- The real-valued matrix-exp adjoint is `L_exp(A.mT, grad_output)`; verify the
  direction, shape, dtype, device, and saved-tensor handling.
- Ordinary backward may use the optimized Fréchet op. Higher derivatives retain a
  differentiable operator-based calculation rather than traversing an opaque kernel.
- Check tangents on both the original input and `grad_output`, including
  forward-over-reverse AD. Forward AD, `torch.func`/`vmap`, and tensor subclasses
  must not silently lose derivatives or fall into an unsupported custom operation.
- Include mixed transform compositions when affected. A valid native route is
  sufficient; adding accelerated JVP/vmap rules is not an automatic requirement.
- Verify autocast and low-precision behavior against the documented contract;
  distinguish FP32 computation with casts from native low-precision arithmetic.

## CUDA execution and numerical behavior

- Device guards select the tensor's device and restore prior state. Launches use
  its current stream, include launch checks, and do not introduce unnecessary host
  synchronization. Test a nondefault stream and noncurrent device when available.
- Tensor storage and temporaries live long enough for asynchronous execution;
  side-stream work has correct ordering and allocator lifetime handling.
- Noncontiguous/expanded inputs, lazy negative/conjugate bits where applicable,
  zero-sized batches, padding/unpadding, and output allocation match the contract.
- Check index bounds, partial blocks, launch-regime boundaries, shared-memory
  synchronization, per-matrix scaling, and nonfinite termination/isolation.
- Preserve numerical criteria across dtypes, all supported N, scaling boundaries,
  and large rotations. Changes to arithmetic, precision, fast-math flags, or
  launch choices require new evidence, not just compilation success.
- CUDA graph compatibility needs execution evidence when changed or claimed;
  the absence of explicit synchronization alone does not establish it.

## Packaging and compatibility

- Build against installed stock PyTorch with its extension build helpers. Check
  Python/PyTorch/toolkit/ABI/architecture assumptions against actual metadata and
  tested environments. Flag private API dependencies when their concrete version
  behavior conflicts with claimed support; do not assume universal compatibility.
- A clean source installation and sdist have all CUDA sources, pinned CUTLASS
  headers, and required notices. No sibling checkout, prebuilt local `.so`, cached
  artifact, or development-only dependency is required at runtime.
- Runtime dependencies stay separate from tests and benchmarks. Optional local
  forum artifacts and `results/` do not enter source distributions or wheels.
- Public API or custom-op schema changes account for existing callers and compiled
  graphs. Describe a real migration need without importing upstream release policy.

## Evidence and performance

- Standalone pytest tests exercise accelerated dispatch as well as native fallback;
  double-only tests must not stand in for FP32/FP16/BF16 kernel correctness.
- Keep build and runtime provenance: source/local-change identity, imported paths,
  extension identity, software versions, hardware, and exact test commands/outcomes.
- Performance comparisons use matching inputs, upstream gradients, timing scope,
  warmup, synchronization, and repetitions. Include wrapper/cast/allocation costs in
  end-to-end claims; retain slowdowns and distinguish forward from forward+backward.
- Historical validation applies only when relevant source and binaries still match.
  Clearly separate measured results, skipped checks, and inferred compatibility.
