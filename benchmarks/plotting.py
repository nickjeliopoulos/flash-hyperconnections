"""Shared style, measurement validation, and report rendering for any operator.

The per-plot modules accept plain arrays/dicts and an Axes. ``plot_report`` is
an optional adapter for benchmark reports; it never imports PyTorch or CUDA.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

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
    """Render report['results'], grouped by operator/dtype/mode/family.

    Rows require case_id, label, operator, dtype, mode, backend, median_ms.
    Optional precision: metric, reference, target, samples, errors. Matching
    input/gradient SHA256 fields are checked when supplied. See benchmarks/README.md.
    """
    if __package__:
        from .plot_cdf import plot_cdf
        from .plot_latency import plot_latency
        from .plot_precision_speedup import plot_precision_speedup
    else:
        from plot_cdf import plot_cdf
        from plot_latency import plot_latency
        from plot_precision_speedup import plot_precision_speedup

    if not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("Formats must be png, pdf, and/or svg")
    if not 0 < quantile <= 1:
        raise ValueError("Quantile must be in (0, 1]")
    if report.get("status", "complete") != "complete":
        raise ValueError("Benchmark report is incomplete")
    plotting = report.get("plotting", {})
    baseline = baseline or plotting.get("baseline", "torch")
    rows = []
    for raw in report["results"]:
        row = dict(raw)
        missing = {"operator", "case_id", "label", "dtype", "mode", "backend", "median_ms"} - row.keys()
        if missing:
            raise ValueError(f"Missing report fields: {sorted(missing)}")
        row.setdefault("family", "default")
        rows.append(row)
    if not rows:
        raise ValueError("Report contains no measurements")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("operator", "dtype", "mode", "family"))].append(row)
    # Validate every group before writing figures.
    prepared = {key: matched_cases(group, baseline) for key, group in groups.items()}
    for cases, _ in prepared.values():
        signatures = {
            tuple(case[baseline]["precision"][k] for k in ("metric", "reference", "target"))
            for case in cases
            if case[baseline].get("precision") is not None
        }
        if len(signatures) > 1:
            raise ValueError("Cannot pool different precision metrics, targets, or references")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    environment = report.get("environment", {})
    hardware = " · ".join(str(environment[k]) for k in ("gpu", "torch") if k in environment)

    def save(fig, stem, caption):
        fig.text(0.5, 0.01, caption + (f"\n{hardware}" if hardware else ""), ha="center", fontsize=8, color="#444444")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        for extension in formats:
            path = root / f"{stem}.{extension}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        fig.clear()

    with mpl.rc_context(STYLE):
        for key, (cases, backends) in prepared.items():
            operator, dtype, mode, family = key
            title = f"{operator} · {dtype.upper()} · {mode.replace('_', ' + ')} · {family}"
            stem = re.sub(r"[^a-zA-Z0-9_-]", "_", "-".join(key))
            order = [baseline, *(name for name in backends if name != baseline)]
            colors = {name: BACKEND_COLORS[i % len(BACKEND_COLORS)] for i, name in enumerate(order)}
            colors.update(plotting.get("backend_colors", {}))
            for offset in range(0, len(cases), 8):
                page = cases[offset : offset + 8]
                fig = Figure(figsize=(max(7, len(page) * 1.3), 5.6))
                ax = fig.subplots()
                plot_latency(
                    ax,
                    [case[baseline]["label"] for case in page],
                    {name: [case[name]["median_ms"] for case in page] for name in backends},
                    baseline=baseline,
                    colors=colors,
                )
                ax.set_title(title, fontweight="bold")
                save(
                    fig,
                    f"{stem}-latency-{offset // 8 + 1}",
                    f"Labels: {baseline} latency / backend latency. "
                    + report.get("method", {}).get("timer", "Reported median latency")
                    + (
                        "\n" + plotting["backend_notes"][baseline]
                        if plotting.get("backend_notes", {}).get(baseline)
                        else ""
                    ),
                )
            precision_cases = [case for case in cases if case[baseline].get("precision") is not None]
            if not precision_cases:
                continue
            signatures = {
                tuple(case[baseline]["precision"][k] for k in ("metric", "reference", "target"))
                for case in precision_cases
            }
            if len(signatures) != 1:
                raise ValueError("Cannot pool different precision metrics, targets, or references")
            metric, reference, target = signatures.pop()
            error_label = f"{metric.replace('_', ' ')} ({target.replace('_', ' ')})"
            series = {
                name: [error for case in precision_cases for error in case[name]["precision"]["errors"]]
                for name in backends
            }
            fig = Figure(figsize=(8, 5.6))
            ax = fig.subplots()
            plot_cdf(ax, series, colors=colors, error_label=error_label)
            ax.set_title(title, fontweight="bold")
            save(
                fig,
                f"{stem}-cdf",
                f"Reference: {reference}. Pooled samples across cases; failures remain in denominator.",
            )
            points = [
                {
                    "backend": name,
                    "label": case[name]["label"],
                    "speedup": case[baseline]["median_ms"] / case[name]["median_ms"],
                    "error": error_quantile(case[name]["precision"]["errors"], quantile),
                }
                for case in precision_cases
                for name in backends
            ]
            fig = Figure(figsize=(8, 5.6))
            ax = fig.subplots()
            plot_precision_speedup(
                ax, points, baseline=baseline, colors=colors, error_label=f"p{quantile * 100:g} {error_label}"
            )
            ax.set_title(title, fontweight="bold")
            failed = sum(
                sum(e is None or not math.isfinite(e) for e in case[name]["precision"]["errors"])
                for case in precision_cases
                for name in backends
            )
            save(
                fig,
                f"{stem}-precision-speedup",
                f"Reference: {reference}. Per-case empirical quantiles; {failed} failed samples. Timing uses full benchmark inputs.",
            )
    manifest = {
        "baseline": baseline,
        "quantile": quantile,
        "artifacts": artifacts,
        "report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }
    (root / "figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
