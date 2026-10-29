# Benchmarks and figures

Install the optional plotting dependencies into the environment with PyTorch and
the compiled extension:

```bash
python -m pip install -e '.[test,benchmark]' --no-build-isolation
```

Run correctness tests, collect timings and numerical errors, save the report,
then export PNG, PDF, and SVG figures:

```bash
python -m benchmarks.run matrix_exp -- \
  --n 4 8 16 32 --batch 128 1024 --dtype fp32 fp16 bf16 \
  --precision --precision-samples 256
```

The driver runs the benchmark's declared pytest files in a subprocess before
starting a separate measurement process. For matrix-exp, these are the numerical,
routing, and PyTorch integration suites. Failures, collection errors, and runs
with no passing CUDA-marked tests stop the pipeline. Expected skips, such as a
multi-GPU test on a single GPU, are recorded. An exit code of zero with only
skipped or CPU tests is insufficient.

Ops default to `.reports/<name>/report.json` in the repository. Layers and
modules use `.reports/<group>/<name>/report.json`, so equal names in different
groups do not share an output directory. This directory is ignored by
Git. Repeated runs reuse it and replace the reports and logs; the driver clears
previous reports and the figure manifest before testing so a failed run cannot
present an earlier success as current. Use `--output-dir PATH` to retain a
separate run. Driver options precede `--`; benchmark options follow it.
Outputs within the benchmark directory are:

- `correctness.json` and `pytest.log`: test counts, scope, command, and output.
- `report.json` and `benchmark.log`: measurements, environment, correctness
  outcome, invocation, and timestamps. The complete report is saved before plots.
- `figures/`: appropriate figures for the data and a `figures.json` manifest.

Without `--precision`, the driver produces one speedup chart per mode. With it,
it also produces one precision CDF and one precision-versus-speedup plot per mode.
All dtypes share each figure; matrix families and workloads are aggregated.
A matrix-exp run with both modes produces six figures total (each exported in
the requested formats), regardless of the number of dtypes, families, or cases. `--formats png svg`
and `--quantile 0.99` are driver options. For quick iteration, `--gate-k EXPR`
selects existing pytest cases and records the gate as a **subset**, not full
validation. Full validation remains the default.

The operator script also remains directly runnable for standalone measurement;
its report explicitly records correctness as `not_run`:

```bash
python benchmarks/ops/matrix_exp.py --n 4 --batch 128 --precision \
  --plots .reports/matrix_exp/figures
```

The direct entry point also defaults to `.reports/matrix_exp/report.json`.
On that entry point, `--plots` enables precision collection automatically.
JSON stores individual timing trials, input/gradient hashes, environment metadata, and raw
per-matrix errors, so plotting never requires rerunning measurements:

```bash
python -m benchmarks.visualization .reports/matrix_exp/report.json \
  --formats png svg --quantile 0.99
```

Plot-only mode needs NumPy and Matplotlib, but does not import PyTorch or require
CUDA/the extension. For legacy matrix-exp reports missing case metadata, use
`python -m benchmarks.ops.matrix_exp --plot-from REPORT --plots DIRECTORY`;
that operator-specific path upgrades old reports before rendering speedup charts.
Incomplete reports and mismatched backend inputs are rejected. Outputs remain
local ignored artifacts; no generated figures or data need to be committed.

## Directory layout

```text
benchmarks/
  ops/matrix_exp.py
  ops/stream_mix.py
  ops/expm_stream_mix.py
  layers/                 # layer benchmarks as they are added
  modules/                # module benchmarks as they are added
  run.py                  # shared correctness → timing → report → plots driver
  common.py               # timing and JSON helpers
  pytest_gate.py          # correctness evidence
  visualization.py        # all plot types, style, report rendering, and plot CLI
```

The driver defaults to `--group ops`. Select `--group layers` or `--group modules`
when adding those benchmarks, for example
`python -m benchmarks.run NAME --group layers -- ...`. Each group uses the same correctness gate and report format. The
report's `benchmark` metadata records its group and name; the existing per-row
`operator` field holds the measured component's name for all three groups.

## Measurements and interpretation

The N=4 exp/project/impulse forward fusion uses:

```bash
python -m benchmarks.run expm_stream_mix -- --wall
```

Defaults cover B=256…24,576 and D=768 with FP32/BF16 residuals and BF16
transition/source boundaries. Baselines compose the existing CUDA exp and
stream-mixing operators, either eagerly or with static-shape `torch.compile`
for the intervening casts and rank-one source. Compilation uses
`emulate_precision_casts=True` to preserve BF16 operand rounding. Graph times
include casts and intermediate/output memory traffic but exclude host dispatch
and allocation overhead; optional wall times include public-wrapper overhead.
Both run under `no_grad`, exclude
generator construction and the module, and make no backward or full-model claim.

Projected stream mixing uses the same driver:

```bash
python -m benchmarks.run stream_mix -- \
  --n 4 8 16 32 --batch 128 512 2048 8192 16384 --d 768 --dtype fp32 bf16 --mixed --decay .99
```

Its gate runs `tests/test_stream_mix.py`. Optional `--triton-source PATH` loads
an existing small/big Triton kernel or `stream_mix.py` dispatcher without
importing its parent model package. Prefer the master dispatcher for comparisons
across batch sizes; keep its sibling kernel files and `numbers.py` beside it.
The report records dispatcher dependencies and source hashes. Use the master
version for nondefault decay. `--mixed` selects
FP32 transition/direction tensors, BF16 sources, and the requested residual
dtype. Timings include all four input gradients in forward + backward. This
benchmark emits timing data; the numerical precision options below belong to
matrix-exp. Stream-mixing accuracy is checked against an independent FP64
projector in its pytest suite.

`--compiled` adds `torch.compile(..., fullgraph=True)` of the same FP32-promoted
formula. Compilation and autotuning occur during warmup, outside measurements.
Reports record CPU affinity and PyTorch thread count; keep these settings fixed
when comparing separate builds. Triton precision is defined by the supplied
source: the master big-NB backward can enable TF32 for low-precision residuals,
even though the PyTorch reference and CUDA implementation preserve FP32
coefficients and intermediates.

The captured master revision `cf600d9f3f4a` fails in big-NB backward when `v`
is FP32 and `x` is BF16 because an `einsum` receives different operand dtypes.
The benchmark reports this failure; omit the optional reference to measure the
full mixed-dtype grid supported by this package. See `docs/VALIDATION.md` for
the tested reference scope.

The baseline is **native PyTorch from the installed build**. Use stock PyTorch for
published comparisons and check the recorded version/build. FP16/BF16 baseline
calls promote to FP32 and cast back; both casts are timed. Both implementations
receive identical input tensors and upstream gradients.

Timings retain the existing synchronized wall-clock method, alternating backend
order across trials. Forward includes the public Python wrapper and allocations.
Forward + backward includes the forward call and `autograd.grad`; it is **not a
backward-only latency**. Compilation and warmup are excluded. The bar charts show
side-by-side backend speedups grouped by dtype, with the baseline at 1×.
Each bar is the geometric mean of matched baseline/candidate median-latency
ratios across all measured families, sizes, batches, and norms. Each case has
equal weight. Labels include geometric-mean latency in microseconds; bar height
always represents speedup. Individual case timings remain in the JSON report.

Precision is collected outside timing, against native PyTorch FP64 (a numerical
reference, not exact arithmetic). Forward errors measure outputs; forward +
backward errors measure the input gradient, using the FP64 augmented-matrix
Fréchet reference `L_exp(A.T, grad_output)`. All references use the **stored,
dtype-rounded inputs**, so BF16 input quantization is not attributed to the kernel.

The metric matches the numerical test suite: per-matrix normalized maximum error:

```text
max(abs(actual - reference)) / (1 + max(abs(reference)))
```

Tests and benchmark evaluation share `flash_hypercons.testing.error_ratio`,
using `dim=(-2, -1)` for per-matrix reductions and detached FP64 arithmetic.
The same module supplies maximum absolute error and an assertion on the ratio;
operator-specific tolerances stay in the tests.

A zero reference uses absolute maximum error. Nonfinite outputs, references,
and errors are recorded as JSON `null` with separate status counts. CDFs retain
these failures in their denominator, instead of silently dropping them. Their finite part can therefore end below one.
Axes use a symmetric-log scale with a small linear region to display exact zeros.

`--precision-samples K` selects the first `min(B, K)` matrices of each timed input.
The backend still runs at the full timed batch size; only FP64 references are
chunked (`--reference-chunk`, default 32). This bounds reference memory while
preserving batch-dependent dispatch. Each sampled matrix has equal weight in the
CDF; cases with more samples contribute more. CDFs pool families, sizes, batches,
and norms within each dtype and mode. Each backend/dtype pair has a distinct
curve in the same figure; use separate runs to inspect individual norm regimes. A batch of one is one sample, not a meaningful distribution by itself.

Precision-versus-speedup plots show one point per backend/dtype pair, using the
pooled empirical p99 error and geometric-mean speedup by default
(`--quantile 1` selects maximum error). Quantiles use the inverse empirical CDF,
including failures as infinity. Nonfinite quantiles appear on an explicitly
labeled band. Baseline errors appear at speedup 1. The plot footer also reports
failed samples, including failures above a finite chosen quantile.

## Input families and stress runs

The default workload remains random Gaussian entries with standard deviation 0.1.
Use `--norm` for a target maximum absolute column sum (matrix 1-norm) from 0
through 1,000, before dtype rounding, and `--family` for random, skew-symmetric,
negative semidefinite, diagonal, or embedded rotation generators. Every requested family/norm/size/batch/dtype combination is measured.

```bash
python benchmarks/ops/matrix_exp.py --n 4 16 32 --batch 129 \
  --family random skew neg_psd diagonal --norm 0.1 1 3 6 10 \
  --precision --output .reports/matrix_exp/families/report.json --plots .reports/matrix_exp/families/figures

python benchmarks/ops/matrix_exp.py --n 2 4 8 9 16 32 --batch 129 \
  --family rotation --norm 1 3 10 100 1000 \
  --precision --output .reports/matrix_exp/rotations/report.json --plots .reports/matrix_exp/rotations/figures
```

Large positive-spectrum inputs can overflow low-precision outputs; failure counts
make this visible. Rotation inputs repeat the same generator across the batch,
so their forward CDF is a diagnostic, not an independent random population.
Their upstream gradients are random. These plots supplement the analytic
rotation regression in `tests/test_matrix_exp.py`.

## Reusing the plotting interface

All plot types live in `benchmarks/visualization.py` and accept an existing
Matplotlib Axes:

- `plot_cdf(ax, {backend: errors}, ...)`
- `plot_speedup(ax, labels, {backend: milliseconds}, baseline=..., ...)`
- `plot_precision_speedup(ax, points, baseline=..., ...)`

`visualization.py` also contains the shared `STYLE`, backend colors, error statistics,
measurement pairing, and `plot_report` adapter. Comparisons follow FLA's simple
Matplotlib blue/orange baseline/candidate palette and light grids, retaining the
former PyTorch PR's DejaVu Sans and editable SVG text. There is no dependency on
FLA, Triton, or pandas. Styling uses `rc_context` and does not persistently change
the caller's plotting configuration.

Future operators can supply the same report without importing matrix-exp code:

```python
from benchmarks.visualization import plot_report

# Include one row per case/backend, with matching inputs across backends.
report = {
    "status": "complete",
    "environment": {"gpu": "GPU name", "torch": "installed version"},
    "plotting": {
        "baseline": "torch",
        "backend_notes": {"torch": "Describe the actual baseline computation here."},
        "backend_colors": {"torch": "#1f77b4", "candidate": "#ff7f0e"},
    },
    "results": rows,
}
plot_report(report, ".reports/my_operator/figures")
```

Each row has this interface (latencies are milliseconds):

```python
{
    "case_id": "unique workload including dtype and mode",
    "label": "B=128 N=8",
    "operator": "my_operator",
    "dtype": "fp32",
    "mode": "forward",
    "family": "random",
    "case": {"batch": 128, "n": 8},  # optional operator-specific metadata
    "backend": "torch",  # other rows use candidate backend names
    "median_ms": 0.05,
    "input_sha256": "hash of the full timed input",
    "gradient_sha256": "hash of the full upstream gradient",
    "precision": {  # optional: omit for timing-only reports
        "metric": "normalized_max_error",
        "reference": "native PyTorch FP64",
        "target": "output",
        "samples": 3,
        "errors": [0.0, 0.00001, None],
        "input_sha256": "hash of the sampled input",
        "gradient_sha256": "hash of the sampled gradient",
    },
}
```

Hashes are optional for externally supplied reports, but matching supplied hashes
are checked. Precision metrics, targets, references, and sample counts must agree
across backends. Every backend must be present for every case in a figure group.
The adapter emits at most three figures per operator/mode, with dtypes combined
and no per-case pages. Precision must be available for all cases in a summary
or omitted from all of them. It writes `figures.json` with aggregation semantics,
a canonical report hash, and artifact hashes. After successful rendering, it
removes obsolete figures listed in the previous manifest; unrelated files remain. Use a separate output directory to preserve a previous run.

## Adding a benchmark

Add one `benchmarks/<group>/<name>.py` module with a CLI accepting `--output PATH`,
where `<group>` is `ops`, `layers`, or `modules`. Keep one file per benchmark;
there are no per-benchmark directories or `__init__.py` files.
Keep its imports lightweight so the driver can read its declarations before
initializing CUDA. Declare `TEST_FILES` (the relevant pytest paths) and
`REQUIRED_TEST_MARKER` (`"cuda"` for current kernels). The driver imports these
declarations directly; there is no registry or benchmark base class.

Reuse `common.measure` for synchronized wall-clock trials and
`common.write_report` for atomic JSON writes. Keep input generation, references,
baseline callables, and tolerances specific to the operator. Emit a report with
`status: "complete"`, explicit case labels/IDs, and the metadata above; precision
is optional. The driver adds correctness evidence and run metadata, saves the
report, then passes the saved data to `plot_report`. Shared plotting never needs
to import the operator. Legacy matrix-exp reports are upgraded by that operator's
`--plot-from` path rather than by generic plotting code.
