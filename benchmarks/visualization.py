"""Plot saved benchmark reports for ops, layers, and modules without PyTorch or CUDA."""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Editable text from the matrix-exp PR style; backend palette follows FLA/Matplotlib comparisons.
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
}
BACKEND_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def error_values(errors):
    """Decode JSON null as a failed (+inf) sample; reject missing/invalid data."""
    values = np.asarray([np.inf if e is None else e for e in errors], dtype=float)
    if values.ndim != 1 or not len(values) or np.any(np.isnan(values) | (values < 0)):
        raise ValueError("Errors must be a nonempty vector of nonnegative values or null/+inf")
    return values


def error_quantile(errors, quantile=0.99):
    """Inverse empirical CDF, including failures; no interpolation of infinities."""
    if not 0 < quantile <= 1:
        raise ValueError("Quantile must be in (0, 1]")
    values = np.sort(error_values(errors))
    return float(values[math.ceil(quantile * len(values)) - 1])


def format_error_axis(ax, which, upper):
    # A linear region preserves exact zero instead of clipping it to epsilon.
    getattr(ax, f"set_{which}scale")("symlog", linthresh=1e-10, linscale=0.5)
    getattr(ax, f"set_{which}lim")(0, upper)
    ax.grid(True, which="major", alpha=0.22, linestyle=":")


def plot_cdf(ax, series, *, colors=None, error_label="Normalized maximum error"):
    """Plot {backend: errors}; null/+inf failures stay in the CDF denominator.

    Zero errors are plotted exactly, with no epsilon substitution. Each sample
    has equal weight; callers decide which workloads to pool.
    """
    values = {name: error_values(errors) for name, errors in series.items()}
    upper = max([1e-9] + [float(v[np.isfinite(v)].max(initial=0)) * 1.2 for v in values.values()])
    for name, errors in values.items():
        unique, counts = np.unique(errors[np.isfinite(errors)], return_counts=True)
        cumulative = np.cumsum(counts) / len(errors)
        failed = int(np.isinf(errors).sum())
        label = f"{name} (n={len(errors):,}, failed={failed})"
        ax.step(
            np.r_[0, unique, upper],
            np.r_[0, cumulative, cumulative[-1] if len(cumulative) else 0],
            where="post",
            label=label,
            color=(colors or {}).get(name),
            linewidth=2,
        )
    format_error_axis(ax, "x", upper)
    ax.set(xlabel=error_label, ylabel="Fraction of samples", ylim=(0, 1.025))
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_speedup(ax, labels, timings, *, baseline, colors=None):
    """Plot baseline/candidate ratios; annotate each bar with speedup and latency."""
    if baseline not in timings:
        raise ValueError(f"Missing baseline: {baseline}")
    values = {name: np.asarray(times, dtype=float) for name, times in timings.items()}
    if not labels or any(v.shape != (len(labels),) or not np.all(np.isfinite(v) & (v > 0)) for v in values.values()):
        raise ValueError("Latency series must have one finite, positive value per case")
    width = 0.8 / len(values)
    positions = np.arange(len(labels))
    for i, (name, times) in enumerate(values.items()):
        bars = ax.bar(
            positions + (i - (len(values) - 1) / 2) * width,
            values[baseline] / times,
            width,
            label=name,
            color=(colors or {}).get(name),
        )
        ax.bar_label(
            bars,
            labels=[
                f"{ratio:.2f}×\n{latency * 1000:.3g} µs"
                for ratio, latency in zip(values[baseline] / times, times, strict=True)
            ],
            padding=3,
            fontsize=9,
        )
    ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel(f"Speedup vs {baseline} (higher is faster)")
    ax.margins(y=0.22)
    ax.grid(axis="y", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    return ax


def plot_precision_speedup(ax, points, *, baseline, colors=None, error_label="p99 normalized maximum error"):
    """Plot dicts with backend, label, speedup, error (null/+inf means failure).

    Failed quantiles appear on a labeled band above all finite measurements.
    The baseline's own numerical errors are shown at speedup 1.
    """
    if not points:
        raise ValueError("No precision/speedup points")
    finite = []
    for point in points:
        if not math.isfinite(point["speedup"]) or point["speedup"] <= 0:
            raise ValueError("Speedups must be finite and positive")
        error = point["error"]
        if error is not None:
            if math.isnan(error) or error < 0:
                raise ValueError("Errors must be nonnegative, null, or +inf")
            if math.isfinite(error):
                finite.append(error)
    upper = max([1e-9] + finite) * 3
    seen = set()
    failures = 0
    annotation_count = 0
    midpoint = math.sqrt(min(p["speedup"] for p in points) * max(p["speedup"] for p in points))
    for point in points:
        name, error = point["backend"], point["error"]
        failed = error is None or not math.isfinite(error)
        failures += int(failed)
        ax.scatter(
            point["speedup"],
            upper if failed else error,
            color=(colors or {}).get(name),
            marker="x" if failed else ("s" if point.get("is_baseline", name == baseline) else "o"),
            s=65,
            label=name if name not in seen else None,
            zorder=3,
        )
        seen.add(name)
        if len(points) <= 12 and not point.get("is_baseline", name == baseline):
            right = point["speedup"] >= midpoint
            above = annotation_count % 2 == 0
            ax.annotate(
                point["label"],
                (point["speedup"], upper if failed else error),
                xytext=(-5 if right else 5, 8 if above else -8),
                textcoords="offset points",
                ha="right" if right else "left",
                va="bottom" if above else "top",
                fontsize=8,
            )
            annotation_count += 1
    format_error_axis(ax, "y", upper * 2)
    if failures:
        ax.axhline(upper, color="#888888", linestyle=":")
        ax.text(
            0.02,
            upper,
            f"nonfinite quantile: {failures} point(s)",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            fontsize=9,
        )
    ax.axvline(1, color="#777777", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel=f"Speedup vs {baseline} (higher is faster)", ylabel=error_label)
    ax.legend(frameon=False)
    return ax


def matched_cases(rows, baseline):
    """Match backends by case ID and reject inconsistent inputs or precision data."""
    cases = defaultdict(dict)
    backends = list(dict.fromkeys(row["backend"] for row in rows))
    if baseline not in backends:
        raise ValueError(f"Missing baseline: {baseline}")
    for row in rows:
        key, backend = row["case_id"], row["backend"]
        if backend in cases[key]:
            raise ValueError(f"Duplicate case/backend: {key}, {backend}")
        if not math.isfinite(row["median_ms"]) or row["median_ms"] <= 0:
            raise ValueError("Latencies must be finite and positive")
        cases[key][backend] = row
    for key, case in cases.items():
        if set(case) != set(backends):
            raise ValueError(f"Missing backend measurements for {key}")
        ref = case[baseline]
        for row in case.values():
            for field in (
                "label",
                "operator",
                "dtype",
                "mode",
                "family",
                "case",
                "norm",
                "input_sha256",
                "gradient_sha256",
            ):
                if row.get(field) != ref.get(field):
                    raise ValueError(f"Mismatched {field} for {key}")
            precision, expected = row.get("precision"), ref.get("precision")
            if (precision is None) != (expected is None):
                raise ValueError(f"Missing precision measurements for {key}")
            if precision is not None:
                for field in ("metric", "reference", "target", "samples", "input_sha256", "gradient_sha256"):
                    if precision.get(field) != expected.get(field):
                        raise ValueError(f"Mismatched precision {field} for {key}")
                if len(error_values(precision["errors"])) != precision["samples"]:
                    raise ValueError(f"Incorrect precision sample count for {key}")
    return list(cases.values()), backends


def plot_report(report, output_dir, *, baseline=None, formats=("png", "pdf", "svg"), quantile=0.99):
    """Render one summary of each plot type per operator/mode, combining dtypes.

    Each dtype pools all measured families and workloads. Speedups give each
    matched case equal weight in log space; precision gives each sample equal
    weight. Raw report records are never changed.
    """
    if not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("Formats must be png, pdf, and/or svg")
    if not 0 < quantile <= 1:
        raise ValueError("Quantile must be in (0, 1]")
    if report.get("status", "complete") != "complete":
        raise ValueError("Benchmark report is incomplete")
    plotting = report.get("plotting", {})
    baseline = baseline or plotting.get("baseline", "torch")
    groups = defaultdict(lambda: defaultdict(list))
    for row in report["results"]:
        missing = {"operator", "case_id", "label", "dtype", "mode", "backend", "median_ms"} - row.keys()
        if missing:
            raise ValueError(f"Missing report fields: {sorted(missing)}")
        groups[(row["operator"], row["mode"])][row["dtype"]].append(row)
    if not groups:
        raise ValueError("Report contains no measurements")
    # Validate all measurements before writing any figures.
    prepared = {}
    for key, dtypes in groups.items():
        summaries, signatures, backend_sets, precision_presence = {}, set(), set(), set()
        for dtype, rows in dtypes.items():
            cases, backends = matched_cases(rows, baseline)
            backend_sets.add(frozenset(backends))
            summaries[dtype] = {}
            for name in backends:
                measurements = [case[name] for case in cases]
                summary = {
                    "latency_ms": math.exp(math.fsum(math.log(row["median_ms"]) for row in measurements) / len(cases)),
                }
                precision = [row["precision"] for row in measurements if row.get("precision") is not None]
                precision_presence.update(row.get("precision") is not None for row in measurements)
                if precision:
                    signatures.update(tuple(p[k] for k in ("metric", "reference", "target")) for p in precision)
                    summary["errors"] = [error for p in precision for error in p["errors"]]
                summaries[dtype][name] = summary
        if len(backend_sets) != 1:
            raise ValueError("Missing backend measurements for a dtype")
        if len(signatures) > 1:
            raise ValueError("Cannot pool different precision metrics, targets, or references")
        if len(precision_presence) > 1:
            raise ValueError("Missing precision for some workloads or dtypes")
        prepared[key] = summaries, next(iter(signatures), None)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "figures.json"
    previous = json.loads(manifest_path.read_text()).get("artifacts", {}) if manifest_path.exists() else {}
    artifacts = {}
    environment = report.get("environment", {})
    hardware = " · ".join(str(environment[k]) for k in ("gpu", "torch") if k in environment)

    def save(fig, stem, caption):
        fig.text(0.5, 0.01, caption + (f"\n{hardware}" if hardware else ""), ha="center", fontsize=8, color="#444444")
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        for extension in formats:
            path = root / f"{stem}.{extension}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        fig.clear()

    with mpl.rc_context(STYLE):
        for key, (summaries, signature) in prepared.items():
            operator, mode = key
            title = f"{operator} · {mode.replace('_', ' + ')}"
            stem = re.sub(r"[^a-zA-Z0-9_-]", "_", "-".join(key))
            backends = list(next(iter(summaries.values())))
            order = [baseline, *(name for name in backends if name != baseline)]
            colors = {name: BACKEND_COLORS[i % len(BACKEND_COLORS)] for i, name in enumerate(order)}
            colors.update(plotting.get("backend_colors", {}))
            fig = Figure(figsize=(8, 5.6))
            ax = fig.subplots()
            plot_speedup(
                ax,
                [dtype.upper() for dtype in summaries],
                {name: [summary[name]["latency_ms"] for summary in summaries.values()] for name in order},
                baseline=baseline,
                colors=colors,
            )
            ax.set_title(title, fontweight="bold")
            save(
                fig,
                f"{stem}-speedup",
                "Geometric means across all measured families and workloads; latency labels in µs."
                + (
                    "\n" + plotting["backend_notes"][baseline]
                    if plotting.get("backend_notes", {}).get(baseline)
                    else ""
                ),
            )
            if signature is None:
                continue
            metric, reference, target = signature
            error_label = f"{metric.replace('_', ' ')} ({target.replace('_', ' ')})"
            series = {
                f"{name} · {dtype.upper()}": summary[name]["errors"]
                for dtype, summary in summaries.items()
                for name in order
            }
            series_colors = {name: BACKEND_COLORS[i % len(BACKEND_COLORS)] for i, name in enumerate(series)}
            fig = Figure(figsize=(8, 5.6))
            ax = fig.subplots()
            plot_cdf(ax, series, colors=series_colors, error_label=error_label)
            ax.set_title(title, fontweight="bold")
            save(
                fig,
                f"{stem}-cdf",
                f"Reference: {reference}. Samples pooled across families and workloads; failures remain in denominator.",
            )
            points = [
                {
                    "backend": f"{name} · {dtype.upper()}",
                    "label": dtype.upper(),
                    "is_baseline": name == baseline,
                    "speedup": summary[baseline]["latency_ms"] / summary[name]["latency_ms"],
                    "error": error_quantile(summary[name]["errors"], quantile),
                }
                for dtype, summary in summaries.items()
                for name in order
            ]
            fig = Figure(figsize=(8, 5.6))
            ax = fig.subplots()
            plot_precision_speedup(
                ax, points, baseline=baseline, colors=series_colors, error_label=f"p{quantile * 100:g} {error_label}"
            )
            ax.set_title(title, fontweight="bold")
            failed = sum(e is None or not math.isfinite(e) for errors in series.values() for e in errors)
            save(
                fig,
                f"{stem}-precision-speedup",
                f"Reference: {reference}. Pooled sample quantiles; geometric mean speedups. {failed} failed samples.",
            )
    manifest = {
        "baseline": baseline,
        "quantile": quantile,
        "aggregation": {
            "group_by": ["operator", "mode", "dtype"],
            "speedup": "geometric_mean",
            "precision": "pooled_samples",
        },
        "artifacts": artifacts,
        "report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }
    # Remove only obsolete figures recorded by our previous successful render.
    for name in previous.keys() - artifacts.keys():
        if Path(name).name == name and Path(name).suffix in {".png", ".pdf", ".svg"}:
            (root / name).unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Saved benchmark report JSON")
    parser.add_argument("--output-dir", type=Path, help="Default: figures/ beside the report")
    parser.add_argument("--baseline", help="Override the baseline declared in the report")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf", "svg"])
    parser.add_argument("--quantile", type=float, default=0.99)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.report.parent / "figures"
    manifest = plot_report(
        json.loads(args.report.read_text()),
        output_dir,
        baseline=args.baseline,
        formats=args.formats,
        quantile=args.quantile,
    )
    print(f"Wrote {len(manifest['artifacts'])} figures to {output_dir}")


if __name__ == "__main__":
    main()
