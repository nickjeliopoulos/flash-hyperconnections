# Repository manifest

This is the canonical inventory of paths tracked by this repository's Git index,
with a purpose for each path. Paths are relative to the repository root. The
CUTLASS submodule is listed once: its contents are tracked by its own repository.
Ignored builds, reports, and local experiments are outside this inventory.

Untracked source/documentation additions are identified separately below. Move
their rows into the tracked table when they enter the index; do not describe
pending files as already tracked. Mark tracked paths removed from the working
tree until their removal enters the index. Keep descriptions current when
responsibilities change. This file is the canonical inventory.
[Agent governance](AGENT_GOVERNANCE.md) and the [lexicon](LEXICON.md) define
repository and terminology conventions.

## Git-tracked files

| Path | Purpose |
|---|---|
| `.agents/forum/.gitkeep` | Keeps the local kernel-experiment directory present while experiment artifacts remain ignored. |
| `.agents/skills/kernel-forum/SKILL.md` | Defines the bounded kernel-optimization workflow and append-only experiment journal conventions. |
| `.agents/skills/kernel-forum/references/experiment-record.md` | Specifies the evidence to record for reproducible kernel experiments and handoffs. |
| `.agents/skills/kernel-forum/scripts/forum` | Implements the local forum CLI, including topic creation, append-only entries, queries, and file locking. |
| `.agents/skills/kernel-forum/scripts/test_forum.py` | Tests the forum CLI and journal behavior independently of CUDA operators. |
| `.agents/skills/kernel-ptx-match-forum/SKILL.md` | Guides compiler-prior CUDA kernel searches using generated code and a persistent design record. |
| `.agents/skills/torch-integrate-review/SKILL.md` | Defines reviews of stock-PyTorch compatibility, operator integration, and standalone packaging. |
| `.agents/skills/torch-integrate-review/references/integration-checklist.md` | Lists concrete checks for routing, registration, autograd, compilation, CUDA execution, and packaging. |
| `.gitignore` | Excludes build products, caches, environments, reports, and local experiment artifacts from Git. |
| `.gitmodules` | Declares the NVIDIA CUTLASS submodule path and upstream repository. |
| `LICENSE` | States the project copyright and MIT license terms. |
| `MANIFEST.in` | Selects documentation, CUDA sources, tests, benchmarks, and vendored headers for source distributions. |
| `README.md` | Introduces installation, public operator contracts, usage, and development workflows. |
| `benchmarks/README.md` | Documents benchmark invocation, correctness gates, measurement scope, saved reports, and plots. |
| `benchmarks/__init__.py` | Marks the source-tree benchmark utilities as a Python package. |
| `benchmarks/common.py` | Shares synchronized wall timing, CUDA graph replay timing, and atomic JSON report writing. |
| `benchmarks/layers/.gitkeep` | Reserves the benchmark directory for future layer measurements. |
| `benchmarks/modules/.gitkeep` | Reserves the benchmark directory for future module measurements. |
| `benchmarks/ops/expm_stream_mix.py` | Benchmarks exp/project/impulse forward and forward-plus-backward at N=4/8/16/32, recording dispatch, input/gradient hashes, and eager/compiled baselines. |
| `benchmarks/ops/matrix_exp.py` | Generates matrix-exp workloads and measures operator timing and optional numerical precision. |
| `benchmarks/ops/stream_mix.py` | Benchmarks projected stream mixing, including optional compiled and local Triton comparisons. |
| `benchmarks/pytest_gate.py` | Records pytest correctness evidence and requires actual execution of the requested device tests. |
| `benchmarks/run.py` | Runs the correctness-to-measurement-to-plot pipeline for ops, layers, and modules. |
| `benchmarks/visualization.py` | Validates matched reports and renders shared speedup, error-CDF, and precision/speedup figures. |
| `csrc/MatrixExpSmall.cuh` | Implements device-side small-matrix exponential and paired Fréchet arithmetic and launch templates. |
| `csrc/expm_stream_mix.cu` | Retains the deprecated experimental exp/project/impulse CUDA forward/backward source for reference; excluded from the build and operator registration. |
| `csrc/matrix_exp.cu` | Prepares CUDA matrix-exp/Fréchet inputs, selects kernels, and exposes extension entry points. |
| `csrc/register.cpp` | Registers namespaced custom-op schemas, CUDA implementations, and the native extension module. |
| `csrc/stream_mix.cu` | Implements projected stream-mixing CUDA forward/backward and shares its cross-moment contraction with exp/mix backward. |
| `docs/AGENT_GOVERNANCE.md` | Defines repository conventions for terminology, operator code, tests, benchmarks, and experiment records. |
| `docs/LEXICON.md` | Defines shared vocabulary, symbols, measurement terms, and the residual-addition/temporal-write-back distinction. |
| `docs/MANIFEST.md` | Provides the canonical Git-index inventory, a purpose for each path, and a coverage check. |
| `docs/VALIDATION.md` | Records measured correctness, performance, build provenance, and remaining validation limits. |
| `flash_hypercons/__init__.py` | Exports the package version and public operator entry points. |
| `flash_hypercons/ops/__init__.py` | Marks the flat operator implementation directory as a Python package. |
| `flash_hypercons/ops/expm_stream_mix.py` | Provides the exp/project/impulse API and explicit precision policy by composing the standalone matrix-exp and stream-mixing operators for every dtype. |
| `flash_hypercons/ops/matrix_exp.py` | Provides matrix-exp routing, fake/autograd registration, native fallbacks, and the scoped backend context. |
| `flash_hypercons/ops/stream_mix.py` | Provides the projected mixing API, reference formulas, validation, fake/autograd registration, and fallbacks. |
| `flash_hypercons/testing.py` | Shares absolute/normalized maximum-error calculations and assertions with explicit reduction dimensions. |
| `provenance.json` | Records source revisions, source hashes, semantic references, dependencies, and implementation origins. |
| `pyproject.toml` | Declares build metadata, package dependencies and extras, and pytest/Ruff configuration. |
| `setup.py` | Builds the active matrix-exp and stream-mixing CUDA extension with CUTLASS headers, excluding the deprecated fused source. |
| `tests/__init__.py` | Marks the test suite as a Python package for shared test helpers. |
| `tests/conftest.py` | Supplies deterministic test seeds and skips CUDA-marked tests when NVIDIA CUDA is unavailable. |
| `tests/reference.py` | Provides matrix-exponential and Fréchet reference calculations and numerical criteria. |
| `tests/test_benchmark_pipeline.py` | Checks correctness-gate enforcement, pipeline ordering, failure handling, and report/plot orchestration. |
| `tests/test_benchmarks.py` | Checks benchmark inputs, measurements, report matching, aggregation, precision metrics, and figure exports. |
| `tests/test_expm_stream_mix.py` | Checks exp/project/impulse composition, missing deprecated registrations, numerical/gradient behavior, precision boundaries, layouts, transforms, compilation, graphs, and large-batch baselines. |
| `tests/test_integration.py` | Checks matrix-exp registration, compilation, differentiation transforms, autocast, and CUDA graphs. |
| `tests/test_matrix_exp.py` | Checks matrix-exp numerics, matrix families, scaling, layouts, launch boundaries, streams, and devices. |
| `tests/test_routing.py` | Checks scoped matrix-exp routing, aliases, fallback selection, invalid inputs, and thread-local behavior. |
| `tests/test_stream_mix.py` | Checks projected mixing outputs/gradients, layouts, dtypes, dispatch, compilation, and CUDA integration. |
| `tests/test_testing.py` | Checks numerical helper reductions, tolerances, precision promotion, empty inputs, and nonfinite handling. |
| `third_party/cutlass` | Pins the NVIDIA CUTLASS submodule providing CUDA template/header dependencies; its files belong to its own Git index. |

## Pending additions

No pending additions.

| Path | Purpose |
|---|---|

## Check inventory coverage

Run this read-only check from the repository root after updating the tables or
staging path changes. It checks complete, duplicate-free tracked coverage and
nonempty purposes, and detects pending rows that need to move or be removed.
Additions need an authored purpose; regenerating a bare filename list would lose
that information. No staging or other Git mutation is performed by this check.

```bash
python -S - <<'PY_CHECK'
import re
import subprocess
from pathlib import Path

text = Path("docs/MANIFEST.md").read_text()
tracked_text = text.split("## Git-tracked files\n", 1)[1].split("## Pending additions\n", 1)[0]
pending_text = text.split("## Pending additions\n", 1)[1].split("## Check inventory coverage\n", 1)[0]
pattern = r"^\| `([^`]+)` \| (.*?) \|$"
tracked_rows = re.findall(pattern, tracked_text, re.MULTILINE)
pending_rows = re.findall(pattern, pending_text, re.MULTILINE)
tracked = set(filter(None, subprocess.check_output(
    ["git", "ls-files", "--cached", "--deduplicate", "-z"]
).decode().split("\0")))
listed = {path for path, _ in tracked_rows}
pending = {path for path, _ in pending_rows}
errors = []
if len(listed) != len(tracked_rows) or len(pending) != len(pending_rows):
    errors.append("Duplicate inventory rows")
if tracked - listed:
    errors.append(f"Missing tracked paths: {sorted(tracked - listed)}")
if listed - tracked:
    errors.append(f"Rows no longer tracked: {sorted(listed - tracked)}")
if pending & tracked:
    errors.append(f"Move pending rows into the tracked table: {sorted(pending & tracked)}")
for path, purpose in tracked_rows + pending_rows:
    if not purpose.strip():
        errors.append(f"Missing purpose: {path}")
for path in pending:
    if not Path(path).exists():
        errors.append(f"Missing pending file: {path}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Verified {len(tracked)} tracked paths with purposes; {len(pending)} pending additions listed.")
PY_CHECK
```
