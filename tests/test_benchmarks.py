"""CPU checks for benchmark statistics, pairing, and portable figure exports."""

import copy
import json
import math

import pytest

from benchmarks.ops.matrix_exp import collect_precision, make_inputs, normalized_errors, upgrade_report


@pytest.fixture
def visualization():
    pytest.importorskip("matplotlib", reason="install the benchmark extra to test plotting")
    from benchmarks import visualization

    return visualization


def report_fixture():
    rows = []
    for backend, latency, errors in (("torch", 0.02, [0, 1e-6, 2e-6]), ("candidate", 0.01, [0, 1e-5, None])):
        rows.append(
            {
                "case_id": "case1",
                "label": "B=3",
                "operator": "example_op",
                "dtype": "fp32",
                "mode": "forward",
                "backend": backend,
                "median_ms": latency,
                "input_sha256": "same-inputs",
                "gradient_sha256": "same-gradients",
                "precision": {
                    "metric": "normalized_max_error",
                    "reference": "FP64",
                    "target": "output",
                    "samples": 3,
                    "errors": errors,
                },
            }
        )
    return {"status": "complete", "results": rows}


def test_error_quantile_keeps_failures_and_exact_zeros(visualization):
    assert visualization.error_quantile([0, 0, 1, None], 0.5) == 0
    assert visualization.error_quantile([0, 0, 1, None], 0.75) == 1
    assert math.isinf(visualization.error_quantile([0, 0, 1, None], 0.99))
    for invalid in ([], [-1], [float("nan")]):
        with pytest.raises(ValueError):
            visualization.error_values(invalid)


def test_cdf_ties_zeros_and_failures(visualization):
    from matplotlib.figure import Figure

    from benchmarks.visualization import plot_cdf

    ax = Figure().subplots()
    plot_cdf(ax, {"candidate": [0, 0, 1e-4, None]})
    line = ax.lines[0]
    assert list(line.get_ydata()) == [0, 0.5, 0.75, 0.75]
    assert list(line.get_xdata())[:3] == [0, 0, 1e-4]
    assert "failed=1" in line.get_label()
    ax = Figure().subplots()
    plot_cdf(ax, {"candidate": [None, None]})
    assert list(ax.lines[0].get_ydata()) == [0, 0]


def test_speedup_bars_use_named_baseline(visualization):
    from matplotlib.figure import Figure

    from benchmarks.visualization import plot_speedup

    ax = Figure().subplots()
    plot_speedup(ax, ["case"], {"candidate": [0.01], "torch": [0.02]}, baseline="torch")
    assert [patch.get_height() for patch in ax.patches] == [2, 1]
    assert [text.get_text() for text in ax.texts] == ["2.00×\n10 µs", "1.00×\n20 µs"]


@pytest.mark.parametrize("corruption", ["input", "gradient", "duplicate", "missing", "samples", "metric", "latency"])
def test_mismatched_measurements_rejected(visualization, corruption):
    rows = report_fixture()["results"]
    if corruption == "input":
        rows[1]["input_sha256"] = "different"
    elif corruption == "gradient":
        rows[1]["gradient_sha256"] = "different"
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[1]))
    elif corruption == "missing":
        rows[1]["case_id"] = "different-case"
    elif corruption == "samples":
        rows[1]["precision"]["errors"].pop()
    elif corruption == "metric":
        rows[1]["precision"]["metric"] = "different"
    else:
        rows[1]["median_ms"] = float("nan")
    with pytest.raises(ValueError):
        visualization.matched_cases(rows, "torch")


def test_export_all_plot_types_and_formats(visualization, tmp_path):
    report = report_fixture()
    original_style = visualization.mpl.rcParams.copy()
    manifest = visualization.plot_report(report, tmp_path)
    assert len(manifest["artifacts"]) == 9
    for name in manifest["artifacts"]:
        assert (tmp_path / name).stat().st_size > 100
    assert json.loads((tmp_path / "figures.json").read_text()) == manifest
    assert visualization.mpl.rcParams["font.size"] == original_style["font.size"]
    assert any("precision-speedup" in name for name in manifest["artifacts"])
    report["status"] = "running"
    with pytest.raises(ValueError, match="incomplete"):
        visualization.plot_report(report, tmp_path)


def test_legacy_timing_report(visualization, tmp_path):
    rows = [
        {"n": 4, "batch": 128, "dtype": "fp16", "mode": "forward", "backend": backend, "median_ms": latency}
        for backend, latency in (("torch", 0.02), ("candidate", 0.01))
    ]
    manifest = visualization.plot_report(upgrade_report({"results": rows}), tmp_path, formats=("svg",))
    assert len(manifest["artifacts"]) == 1
    assert "speedup" in next(iter(manifest["artifacts"]))


def test_visualization_cli_uses_saved_report(visualization, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report_fixture()))
    visualization.main([str(path), "--formats", "svg"])
    manifest = json.loads((tmp_path / "figures/figures.json").read_text())
    assert len(manifest["artifacts"]) == 3
    assert all((tmp_path / "figures" / name).is_file() for name in manifest["artifacts"])


def test_normalized_errors_zero_and_nonfinite():
    import torch

    actual = torch.tensor([[[0.0]], [[1.0]], [[float("nan")]], [[1.0]], [[3.0]]])
    expected = torch.tensor([[[0.0]], [[0.0]], [[1.0]], [[float("inf")]], [[2.0]]])
    errors, counts = normalized_errors(actual, expected)
    assert errors[:4] == [0, 1, None, None]
    assert errors[4] == pytest.approx(1 / 3)
    assert counts == {"finite": 3, "reference_nonfinite": 1, "actual_nonfinite": 1, "nonfinite_error": 0}
    json.dumps(errors, allow_nan=False)


def test_precision_uses_full_timing_batch_and_sampled_reference():
    import torch

    a = torch.diag_embed(torch.linspace(-0.2, 0.2, 10).reshape(5, 2))
    g = torch.ones_like(a) * 0.1
    calls = []

    def backend(x):
        calls.append((len(x), x.requires_grad))
        return torch.linalg.matrix_exp(x)

    records = collect_precision(a, g, {"candidate": backend}, samples=3, chunk_size=2)
    assert calls == [(5, False), (5, True)]
    assert set(records) == {("candidate", "forward"), ("candidate", "forward_backward")}
    for record in records.values():
        assert record["samples"] == len(record["errors"]) == 3
        assert max(record["errors"]) < 1e-6
        assert record["status_counts"]["finite"] == 3


def test_rotation_inputs_use_requested_column_norm():
    import torch

    a, g = make_inputs(3, 4, torch.bfloat16, "rotation", 1000, 42, device="cpu")
    assert torch.linalg.matrix_norm(a.float(), ord=1).tolist() == [1000] * 3
    torch.testing.assert_close(a, -a.mT, atol=0, rtol=0)
    assert g.dtype == a.dtype


def test_operator_metadata_controls_plots(visualization, tmp_path):
    report = report_fixture()
    report["plotting"] = {
        "baseline": "reference_backend",
        "backend_notes": {"reference_backend": "Reference uses native input precision."},
        "backend_colors": {"reference_backend": "#123456"},
    }
    for row in report["results"]:
        if row["backend"] == "torch":
            row["backend"] = "reference_backend"
        row["label"] = "channels=37"
        row["case"] = {"channels": 37}
        row["precision"]["metric"] = "absolute_error"
    manifest = visualization.plot_report(report, tmp_path, formats=("svg",))
    assert manifest["baseline"] == "reference_backend"
    speedup = next(tmp_path.glob("*-speedup.svg")).read_text()
    cdf = next(tmp_path.glob("*-cdf.svg")).read_text()
    assert "Reference uses native input precision." in speedup
    assert "FP32" in speedup and "#123456" in speedup
    assert "FP32 calculation" not in speedup and "matrix_exp" not in speedup
    assert "absolute error" in cdf
    report["results"][1]["case"] = {"channels": 38}
    with pytest.raises(ValueError, match="Mismatched case"):
        visualization.plot_report(report, tmp_path)


def test_measurement_alternates_backends_and_synchronizes(monkeypatch):
    from benchmarks import common

    events = []
    ticks = iter([0, 0.004, 1, 1.006, 2, 2.008, 3, 3.010])
    monkeypatch.setattr(common.time, "perf_counter", lambda: next(ticks))
    samples = common.measure(
        {"a": "a", "b": "b"}, events.append, lambda: events.append("sync"), warmup=1, repeat=2, trials=2
    )
    assert samples["a"] == pytest.approx([2, 5])
    assert samples["b"] == pytest.approx([3, 4])
    assert events == ["a", "b"] + ["sync", "a", "a", "sync", "sync", "b", "b", "sync"] + [
        "sync",
        "b",
        "b",
        "sync",
        "sync",
        "a",
        "a",
        "sync",
    ]


def test_report_combines_dtypes_and_pools_workloads(visualization, tmp_path, monkeypatch):
    report = {"results": []}
    for mode in ("forward", "forward_backward"):
        for dtype in ("fp32", "fp16", "bf16"):
            for size in range(12):  # No extra pages even with many workloads.
                for family, times in (("random", (0.02, 0.01)), ("skew", (0.16, 0.02))):
                    for row, latency in zip(report_fixture()["results"], times, strict=True):
                        row.update(
                            case_id=f"{mode}-{dtype}-{size}-{family}",
                            label=f"N={size}",
                            mode=mode,
                            dtype=dtype,
                            family=family,
                            median_ms=latency,
                        )
                        report["results"].append(row)
    original = copy.deepcopy(report)
    captured = {"bars": [], "cdf": [], "points": []}
    speedup, cdf, scatter = visualization.plot_speedup, visualization.plot_cdf, visualization.plot_precision_speedup

    def bars(ax, labels, timings, **kwargs):
        speedup(ax, labels, timings, **kwargs)
        captured["bars"].append((labels, [patch.get_height() for patch in ax.patches]))

    def distributions(ax, series, **kwargs):
        captured["cdf"].append(series)
        return cdf(ax, series, **kwargs)

    def points(ax, data, **kwargs):
        captured["points"].append(data)
        return scatter(ax, data, **kwargs)

    monkeypatch.setattr(visualization, "plot_speedup", bars)
    monkeypatch.setattr(visualization, "plot_cdf", distributions)
    monkeypatch.setattr(visualization, "plot_precision_speedup", points)
    # Previously generated per-dtype pages are retired; unrelated files survive.
    (tmp_path / "old-latency.svg").write_text("old")
    (tmp_path / "notes.svg").write_text("keep")
    (tmp_path / "figures.json").write_text(json.dumps({"artifacts": {"old-latency.svg": "hash"}}))
    manifest = visualization.plot_report(report, tmp_path, formats=("svg",), quantile=0.5)
    assert report == original
    assert len(manifest["artifacts"]) == 6
    assert not (tmp_path / "old-latency.svg").exists()
    assert (tmp_path / "notes.svg").read_text() == "keep"
    assert not any("latency" in name or "fp32" in name for name in manifest["artifacts"])
    assert len(captured["bars"]) == len(captured["cdf"]) == len(captured["points"]) == 2
    for labels, heights in captured["bars"]:
        assert labels == ["FP32", "FP16", "BF16"]
        assert heights == pytest.approx([1, 1, 1, 4, 4, 4])
    for series in captured["cdf"]:
        assert len(series) == 6
        for name, errors in series.items():
            assert len(errors) == 72
            assert errors.count(None) == (24 if name.startswith("candidate") else 0)
    for data in captured["points"]:
        assert len(data) == 6
        for point in data:
            assert point["speedup"] == pytest.approx(1 if point["is_baseline"] else 4)
            assert point["error"] == pytest.approx(1e-6 if point["is_baseline"] else 1e-5)


@pytest.mark.parametrize("corruption", ["precision", "metric", "backend"])
def test_summary_rejects_inconsistent_dtypes(visualization, tmp_path, corruption):
    report = report_fixture()
    additional = copy.deepcopy(report["results"])
    for row in additional:
        row.update(dtype="bf16", case_id="bf16-case")
        if corruption == "precision":
            row.pop("precision")
        elif corruption == "metric":
            row["precision"]["metric"] = "different"
        elif row["backend"] == "candidate":
            row["backend"] = "different"
    report["results"].extend(additional)
    with pytest.raises(ValueError):
        visualization.plot_report(report, tmp_path, formats=("svg",))
    assert not list(tmp_path.glob("*.svg"))
