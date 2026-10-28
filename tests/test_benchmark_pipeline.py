"""The correctness gate must execute tests before any measurements or plots."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks import run
from benchmarks.common import write_report


@pytest.mark.parametrize(
    ("source", "passed", "required_passed"),
    [
        ("@pytest.mark.accelerator\ndef test_ok(): pass", True, 1),
        ("@pytest.mark.accelerator\ndef test_bad(): assert False", False, 0),
        ("@pytest.mark.accelerator\ndef test_skip(): pytest.skip('no device')", False, 0),
        ("def test_cpu_only(): pass", False, 0),
        ("@pytest.mark.accelerator\n@pytest.mark.xfail\ndef test_expected(): assert False", False, 0),
        (
            "@pytest.fixture(autouse=True)\ndef cleanup():\n yield\n assert False\n"
            "@pytest.mark.accelerator\ndef test_bad_teardown(): pass",
            False,
            0,
        ),
        ("raise RuntimeError('collection failed')", False, 0),
        ("", False, 0),
    ],
    ids=["pass", "failure", "all-skipped", "cpu-only", "xfail", "teardown", "collection", "empty"],
)
def test_pytest_gate_records_real_execution(tmp_path, source, passed, required_passed):
    suite = tmp_path / "test_gate.py"
    suite.write_text("import pytest\n" + source + "\n")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nmarkers = accelerator: synthetic accelerator test\n")
    report = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(config),
            str(suite),
            "-p",
            "benchmarks.pytest_gate",
            "--benchmark-gate-report",
            str(report),
            "--benchmark-required-marker",
            "accelerator",
        ],
        cwd=run.ROOT,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert report.exists(), result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert (data["status"] == "passed") == passed
    assert data["required_passed"] == required_passed
    assert data["exit_code"] == result.returncode


@pytest.fixture
def pipeline(monkeypatch):
    module = SimpleNamespace(TEST_FILES=("tests/test_example.py",), REQUIRED_TEST_MARKER="cuda")
    monkeypatch.setattr(run, "load_benchmark", lambda name, group="ops": module)
    return module


@pytest.mark.parametrize("stage", ["gate-failure", "gate-skipped", "gate-missing", "benchmark-failure", "incomplete"])
def test_pipeline_stops_before_downstream_work(tmp_path, monkeypatch, pipeline, stage):
    calls = []
    output = tmp_path / "run"

    def command(args, log):
        calls.append(args)
        if "pytest" in args:
            if stage != "gate-missing":
                write_report(
                    output / "correctness.json",
                    {
                        "status": "failed" if stage == "gate-failure" else "passed",
                        "required_passed": 0 if stage == "gate-skipped" else 1,
                    },
                )
            return 1 if stage == "gate-failure" else 0
        if stage == "incomplete":
            write_report(output / "report.json", {"status": "running", "results": []})
            return 0
        return 1

    monkeypatch.setattr(run, "run_command", command)
    with pytest.raises(RuntimeError):
        run.run_pipeline("example", output, [])
    assert len(calls) == (1 if stage.startswith("gate") else 2)
    assert not (output / "figures").exists()


@pytest.mark.parametrize("group", ["ops", "layers", "modules"])
def test_pipeline_saves_gate_and_measurements_before_rendering(tmp_path, monkeypatch, pipeline, group):
    pytest.importorskip("matplotlib")
    from benchmarks import visualization

    output = tmp_path / "run"
    calls = []

    def command(args, log):
        calls.append(args)
        if "pytest" in args:
            write_report(output / "correctness.json", {"status": "passed", "required_passed": 2})
        else:
            write_report(
                output / "report.json",
                {
                    "status": "complete",
                    "plotting": {"baseline": "reference"},
                    "results": [
                        {
                            "operator": "example",
                            "case_id": "c",
                            "label": "channels=3",
                            "dtype": "fp32",
                            "mode": "forward",
                            "backend": backend,
                            "median_ms": latency,
                        }
                        for backend, latency in (("reference", 0.02), ("candidate", 0.01))
                    ],
                },
            )
        return 0

    real_plot = visualization.plot_report

    def plot(report, *args, **kwargs):
        assert report == json.loads((output / "report.json").read_text())
        assert report["correctness"]["status"] == "passed"
        assert report["correctness"]["scope"] == "subset"
        assert report["run"]["command"] == calls[1]
        assert report["benchmark"] == {"group": group, "name": "example"}
        return real_plot(report, *args, **kwargs)

    monkeypatch.setattr(run, "run_command", command)
    monkeypatch.setattr(visualization, "plot_report", plot)
    run.run_pipeline("example", output, ["--size", "3"], group=group, gate_k="example", formats=("svg",))
    assert calls[0][-2:] == ["-k", "example"]
    assert "--size" in calls[1]
    assert calls[1][2] == f"benchmarks.{group}.example"
    assert (output / "figures" / "figures.json").exists()
    assert len(list((output / "figures").glob("*.svg"))) == 1
    run.run_pipeline("example", output, ["--size", "3"], group=group, gate_k="example", formats=("svg",))
    assert len(calls) == 4


def test_failed_repeat_removes_stale_success(tmp_path, monkeypatch, pipeline):
    output = tmp_path / "run"
    for name in ("report.json", "correctness.json", "pytest.log", "benchmark.log", "figures/figures.json"):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status": "passed", "required_passed": 1}')
    (output / "notes.txt").write_text("Keep local notes")
    monkeypatch.setattr(run, "run_command", lambda args, log: 0)
    with pytest.raises(RuntimeError, match="Correctness gate failed"):
        run.run_pipeline("example", output, [])
    assert not (output / "report.json").exists()
    assert not (output / "figures/figures.json").exists()
    assert (output / "notes.txt").read_text() == "Keep local notes"


@pytest.mark.parametrize("operator", ["matrix_exp", "another_operator"])
@pytest.mark.parametrize("group", ["ops", "layers", "modules"])
def test_default_output_directory_is_per_operator(tmp_path, monkeypatch, operator, group):
    monkeypatch.setattr(run, "REPORTS_ROOT", tmp_path / ".reports")
    calls = []
    monkeypatch.setattr(run, "run_pipeline", lambda *args, **kwargs: calls.append((args, kwargs)))
    run.main([operator, "--group", group])
    root = tmp_path / ".reports" if group == "ops" else tmp_path / ".reports" / group
    assert calls[0][0][1] == root / operator
    assert calls[0][1]["group"] == group
    run.main([operator, "--group", group, "--output-dir", str(tmp_path / "custom")])
    assert calls[1][0][1] == tmp_path / "custom"


def test_load_flat_operator_benchmark():
    module = run.load_benchmark("matrix_exp")
    assert module.__name__ == "benchmarks.ops.matrix_exp"
    assert module.TEST_FILES and module.REQUIRED_TEST_MARKER == "cuda"
    for name, group in (("matrix_exp", "layers"), ("matrix_exp", "unknown"), ("../matrix_exp", "ops")):
        with pytest.raises(ValueError):
            run.load_benchmark(name, group)


@pytest.mark.parametrize("flag", ["--output=other.json", "--plots", "--plot-from=data.json"])
def test_driver_owns_output_and_plotting_order(tmp_path, pipeline, flag):
    with pytest.raises(ValueError, match="managed by the pipeline"):
        run.run_pipeline("example", tmp_path / "run", [flag])
    assert not (tmp_path / "run").exists()
