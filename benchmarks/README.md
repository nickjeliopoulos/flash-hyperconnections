# Benchmarks and figures

Install the optional plotting dependencies into the environment with PyTorch and
the compiled extension:

```bash
python -m pip install -e '.[test,benchmark]' --no-build-isolation
```

Run correctness tests, collect timings and numerical errors, save the report,
then export PNG, PDF, and SVG figures:

```bash
python -m benchmarks.run matrix_exp --output-dir results/matrix-exp-run -- \
  --n 4 8 16 32 --batch 128 1024 --dtype fp32 fp16 bf16 \
  --precision --precision-samples 256
```

The driver runs the operator's declared pytest files in a subprocess before
starting a separate measurement process. For matrix-exp, these are the numerical,
routing, and PyTorch integration suites. Failures, collection errors, and runs
with no passing CUDA-marked tests stop the pipeline. Expected skips, such as a
multi-GPU test on a single GPU, are recorded. An exit code of zero with only
skipped or CPU tests is insufficient.

The output directory must be new, so stale reports cannot look like a successful
run. Omitting `--output-dir` creates a timestamped directory under `results/`.
Driver options precede `--`; operator benchmark options follow it. Outputs are:

- `correctness.json` and `pytest.log`: test counts, scope, command, and output.
- `report.json` and `benchmark.log`: measurements, environment, correctness
  outcome, invocation, and timestamps. The complete report is saved before plots.
- `figures/`: appropriate figures for the data and a `figures.json` manifest.

Without `--precision`, the driver produces latency charts. With it, it also
produces precision CDFs and precision-versus-speedup plots. `--formats png svg`
and `--quantile 0.99` are driver options. For quick iteration, `--gate-k EXPR`
selects existing pytest cases and records the gate as a **subset**, not full
validation. Full validation remains the default.

The operator script also remains directly runnable for standalone measurement;
its report explicitly records correctness as `not_run`:

```bash
python benchmarks/matrix_exp.py --n 4 --batch 128 --precision \
  --output results/measurement.json --plots results/measurement-figures
```

On that direct entry point, `--plots` enables precision collection automatically.
JSON stores
individual timing trials, input/gradient hashes, environment metadata, and raw
per-matrix errors, so plotting never requires rerunning measurements:

```bash
python benchmarks/matrix_exp.py --plot-from results/matrix-exp-run/report.json \
  --plots results/replotted-figures --formats png svg --quantile 0.99
```

Plot-only mode needs NumPy and Matplotlib, but does not import PyTorch or require
CUDA/the extension. Older timing-only JSON reports can produce latency charts.
Incomplete reports and mismatched backend inputs are rejected. Outputs remain
local ignored artifacts; no generated figures or data need to be committed.

## Measurements and interpretation

The baseline is **native PyTorch from the installed build**. Use stock PyTorch for
published comparisons and check the recorded version/build. FP16/BF16 baseline
calls promote to FP32 and cast back; both casts are timed. Both implementations
receive identical input tensors and upstream gradients.

Timings retain the existing synchronized wall-clock method, alternating backend
order across trials. Forward includes the public Python wrapper and allocations.
Forward + backward includes the forward call and `autograd.grad`; it is **not a
backward-only latency**. Compilation and warmup are excluded. The bar charts show
side-by-side median latencies, annotated with baseline latency / candidate latency.

Precision is collected outside timing, against native PyTorch FP64 (a numerical
reference, not exact arithmetic). Forward errors measure outputs; forward +
backward errors measure the input gradient, using the FP64 augmented-matrix
Fréchet reference `L_exp(A.T, grad_output)`. All references use the **stored,
dtype-rounded inputs**, so BF16 input quantization is not attributed to the kernel.

The metric matches the numerical test suite: per-matrix normalized maximum error:

```text
max(abs(actual - reference)) / (1 + max(abs(reference)))
```

A zero reference uses absolute maximum error. Nonfinite outputs, references,
and errors are recorded as JSON `null` with separate status counts. CDFs retain
these failures in their denominator, instead of silently dropping them. Their finite part can therefore end below one.
Axes use a symmetric-log scale with a small linear region to display exact zeros.

`--precision-samples K` selects the first `min(B, K)` matrices of each timed input.
The backend still runs at the full timed batch size; only FP64 references are
chunked (`--reference-chunk`, default 32). This bounds reference memory while
preserving batch-dependent dispatch. Each sampled matrix has equal weight in the
CDF; cases with more samples contribute more. CDFs pool sizes/batches/norms within
each operator, dtype, mode, and family; use separate runs to inspect individual
norm regimes. A batch of one is one sample, not a meaningful distribution by itself.

Precision-versus-speedup plots use the per-case empirical p99 error by default
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
python benchmarks/matrix_exp.py --n 4 16 32 --batch 129 \
  --family random skew neg_psd diagonal --norm 0.1 1 3 6 10 \
  --precision --output results/families.json --plots results/families

python benchmarks/matrix_exp.py --n 2 4 8 9 16 32 --batch 129 \
  --family rotation --norm 1 3 10 100 1000 \
  --precision --output results/rotations.json --plots results/rotations
```

Large positive-spectrum inputs can overflow low-precision outputs; failure counts
make this visible. Rotation inputs repeat the same generator across the batch,
so their forward CDF is a diagnostic, not an independent random population.
Their upstream gradients are random. These plots supplement the analytic
rotation regression in `tests/test_matrix_exp.py`.

## Reusing the plotting interface

Each plot type lives in a separate file and accepts an existing Matplotlib Axes:

- `plot_cdf.plot_cdf(ax, {backend: errors}, ...)`
- `plot_latency.plot_latency(ax, labels, {backend: milliseconds}, baseline=..., ...)`
- `plot_precision_speedup.plot_precision_speedup(ax, points, baseline=..., ...)`

`plotting.py` contains the shared `STYLE`, backend colors, error statistics,
measurement pairing, and `plot_report` adapter. Comparisons follow FLA's simple
Matplotlib blue/orange baseline/candidate palette and light grids, retaining the
former PyTorch PR's DejaVu Sans and editable SVG text. There is no dependency on
FLA, Triton, or pandas. Styling uses `rc_context` and does not persistently change
the caller's plotting configuration.

Future operators can supply the same report without importing matrix-exp code:

```python
from benchmarks.plotting import plot_report

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
plot_report(report, "results/my-operator")
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
The adapter emits separate figures per group, paginates latency charts at eight
cases, and writes `figures.json` with a canonical report hash and artifact hashes.
Regenerate into a fresh directory when changing report groups or formats; the
manifest lists exactly the files generated by the latest invocation.

## Adding an operator

Add one `benchmarks/<operator>.py` module with a CLI accepting `--output PATH`.
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
