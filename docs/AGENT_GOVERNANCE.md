# Agent governance

Keep `flash-hypercons` lean and organized around individual operators. Extend
the existing structure when adding an operator or implementation; introduce
shared abstractions only when there is a concrete need.

## Repository documentation

- Keep governance and validation records in `docs/`.
- Use [LEXICON.md](LEXICON.md) for shared vocabulary. Describe the ordinary
  additive update as the layer-output contribution, source update, or residual
  addition; reserve temporal write-back / `TemporalWriteback` for the distinct
  architecture that constructs temporal components.
- Keep [MANIFEST.md](MANIFEST.md) in `docs/` synchronized with Git's index and
  give every tracked path a concrete purpose. List untracked additions separately
  until they enter the index, and list submodules once. Run its read-only coverage
  check after updating the inventory. Keep the full inventory in this document.
- Paths in these documents are relative to the repository root unless stated
  otherwise.

## Operator code

- Place Python operator implementations and selection logic under
  `flash_hypercons/ops/<operator>.py` and native sources under `csrc/`.
  Keep `ops/` flat: use one module per operator, without per-operator
  subdirectories or `__init__.py` files.
- Keep implementation selection private behind the public operator API.
- Preserve documented PyTorch integration, fallback behavior, and numerical
  guarantees when changing kernels. Keep optional benchmark and plotting
  dependencies out of runtime imports.

## Tests

- Keep correctness and regression tests in `tests/`. This directory covers both
  individual operators and package-wide behavior, including routing, autograd,
  compilation, and installation where applicable.
- Follow the existing test layout. Add operator-specific files as needed without
  requiring a new directory hierarchy for each operator.
- Keep reference calculations, input cases, and tolerances specific to each
  operator. Share fixtures and small assertion helpers where their semantics
  agree; different operators need not use the same error metric or tolerance.
- Reuse `flash_hypercons/testing.py` for maximum absolute error, normalized
  maximum error, and assertions on that metric in both tests and benchmarks.
  Choose reduction dimensions explicitly for batched comparisons; matrix-exp
  uses `dim=(-2, -1)` so each matrix is checked independently.
- Extend existing parametrized tests with targeted cases before adding new test
  functions. As the suite grows, select a small smoke subset with pytest markers
  while retaining broader checks for full validation.
- Turn numerical bugs into focused regression tests. Numerical measurements in
  benchmarks support analysis and plots; they do not replace test assertions.
- Run checks appropriate to the change and record relevant environment details
  and limitations for GPU validation. Distinguish checks that passed from checks
  that could not run.

## Benchmarks and plots

- Group benchmarks under `benchmarks/ops/`, `benchmarks/layers/`, and
  `benchmarks/modules/`. Use one file per measured component, such as
  `benchmarks/ops/matrix_exp.py`, without per-component subdirectories or
  `__init__.py` files. Keep timing, input selection, and plot orchestration
  for that component together.
- Use `benchmarks/` primarily for performance measurement and plotting. Include
  numerical evaluation when needed to produce precision CDFs or compare
  precision with runtime.
- Keep all plot functions, shared style, and report visualization in the single
  `benchmarks/visualization.py` module. Ops, layers, and modules supply their
  timing and precision data to the same plotting functions.
- Keep report visualization compact: one speedup chart per mode, plus a CDF
  and precision-versus-speedup plot when precision data is present. Combine
  dtypes in each figure and pool matrix families and workloads. Use geometric
  means of matched speedups and pooled precision samples, with clear labels.
  Bar heights represent speedup; latency may appear as an annotation. Preserve
  detailed case measurements in JSON instead of generating per-case plot pages.
- Keep shapes, input generation, references, backend callables, and baseline
  semantics in each operator's benchmark. Share timing, environment capture,
  and report writing through small functions once another operator needs them;
  a new operator should not require copying the whole benchmark harness.
- Use a common measurement format with operator-specific case metadata and
  optional precision data. Shared plots should obtain labels, error metrics,
  and baseline descriptions from the report instead of assuming matrix-exp
  shapes or FP32 promotion. Avoid an operator registry or benchmark class
  hierarchy until repeated use demonstrates a need.
- Use appropriate baselines and label timing scope, units, error metrics, and
  input regimes. Preserve enough raw measurements and environment metadata to
  reproduce plots and understand comparisons.
- Validate correctness before accepting an optimization's speedup. Reuse pytest
  for this check; keep assertion logic out of benchmark runners, and do not
  treat an all-skipped GPU test run as successful validation.
- Use `python -m benchmarks.run <name> --group <ops|layers|modules>` for the
  correctness → measurement → saved report → plot workflow (`ops` is the
  default group). Each benchmark declares its test files and
  required device marker. Keep subset validation explicit in report metadata.
- Keep numerical evaluation focused on relevant inputs. For matrix exponential,
  the current benchmark range extends through a maximum absolute column sum
  (matrix 1-norm) of 1000, with the existing normalized maximum-error metric.
  Avoid adding alternative metrics or extreme-input machinery without a
  concrete requirement.

## Local artifacts

- Keep generated measurements and figures in ignored local output directories,
  using `.reports/<operator>/report.json` and `.reports/<operator>/figures/` by
  default for ops; layers and modules use `.reports/<group>/<name>/`.
  Track reusable scripts and concise validation records.
- Track reusable skills under `.agents/skills/`. Keep `.agents/forum/.gitkeep`
  as the directory placeholder; use the ignored contents of `.agents/forum/`
  for optional local kernel optimization experiments.
- Keep significant kernel design decisions and reproducible performance gains
  in `.agents/forum/<operator-topic>/KERNEL.md`, with links to forum entries and
  measurement evidence. Keep routine trials in the append-only journal. The
  `kernel-ptx-match-forum` skill defines bounded kernel search bootstrapped from
  a DSL/compiler prior (typically Triton/PTX), using the existing `kernel-forum`
  tooling.
