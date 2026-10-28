"""Run correctness tests, measurements, then visualize the saved benchmark report.

Example: python -m benchmarks.run matrix_exp -- --n 4 --batch 128
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import REPORTS_ROOT, write_report

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("ops", "layers", "modules")


def load_benchmark(operator, group="ops"):
    if group not in GROUPS:
        raise ValueError(f"Unknown benchmark group: {group}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", operator):
        raise ValueError("Benchmark must be a module name")
    if not (ROOT / "benchmarks" / group / f"{operator}.py").is_file():
        raise ValueError(f"No benchmark module for {group}/{operator}")
    module = importlib.import_module(f"benchmarks.{group}.{operator}")
    if not getattr(module, "TEST_FILES", None) or not getattr(module, "REQUIRED_TEST_MARKER", None):
        raise ValueError(f"{operator} does not declare its correctness tests")
    return module


def run_command(command, log_path):
    """Stream child output while keeping a log beside its measurements."""
    print("Running: " + " ".join(map(str, command)), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process,
    ):
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def run_pipeline(
    operator, output_dir, benchmark_args, *, group="ops", gate_k=None, formats=("png", "pdf", "svg"), quantile=0.99
):
    module = load_benchmark(operator, group)
    test_files = module.TEST_FILES
    marker = module.REQUIRED_TEST_MARKER
    # The driver owns output paths and plotting order. Other options belong to the operator.
    reserved = {"--output", "--plots", "--plot-from"}
    if any(arg.split("=", 1)[0] in reserved for arg in benchmark_args):
        raise ValueError("Use --output-dir on the driver; output and plotting flags are managed by the pipeline")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A failed repeat must not expose an earlier report or manifest as current.
    for name in ("report.json", "correctness.json", "pytest.log", "benchmark.log", "figures/figures.json"):
        (output_dir / name).unlink(missing_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    gate_path = output_dir / "correctness.json"
    gate_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *test_files,
        "-p",
        "benchmarks.pytest_gate",
        "--benchmark-gate-report",
        str(gate_path),
        "--benchmark-required-marker",
        marker,
    ]
    if gate_k:
        gate_command += ["-k", gate_k]
    code = run_command(gate_command, output_dir / "pytest.log")
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if code or gate.get("status") != "passed" or not gate.get("required_passed"):
        raise RuntimeError(
            f"Correctness gate failed or no {marker}-marked tests passed; see {output_dir / 'pytest.log'}"
        )
    gate.update(
        command=gate_command, test_files=list(test_files), selection=gate_k, scope="subset" if gate_k else "full"
    )
    write_report(gate_path, gate)
    report_path = output_dir / "report.json"
    command = [sys.executable, "-m", f"benchmarks.{group}.{operator}", *benchmark_args, "--output", str(report_path)]
    code = run_command(command, output_dir / "benchmark.log")
    if code:
        raise RuntimeError(f"Benchmark failed with exit code {code}; see {output_dir / 'benchmark.log'}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or not report.get("results"):
        raise RuntimeError("Benchmark did not produce a complete, nonempty report")
    if any(row.get("operator") != operator for row in report["results"]):
        raise RuntimeError("Benchmark report contains the wrong operator")
    report["correctness"] = gate
    report["benchmark"] = {"group": group, "name": operator}
    report["run"] = {"command": command, "started_at": started, "measured_at": datetime.now(timezone.utc).isoformat()}
    write_report(report_path, report)
    # Rendering consumes the persisted metadata, not live operator state.
    from .visualization import plot_report

    manifest = plot_report(
        json.loads(report_path.read_text()), output_dir / "figures", formats=formats, quantile=quantile
    )
    print(f"Saved {report_path} and {len(manifest['artifacts'])} figures", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("operator", help="Benchmark name, e.g. matrix_exp")
    parser.add_argument("--group", choices=GROUPS, default="ops", help="Benchmark group (default: ops)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: .reports/<name>/ for ops, .reports/<group>/<name>/ otherwise)",
    )
    parser.add_argument("--gate-k", help="Pytest -k selection for iteration; records the gate as a subset")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf", "svg"])
    parser.add_argument("--quantile", type=float, default=0.99)
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    if not 0 < args.quantile <= 1:
        parser.error("quantile must be in (0, 1]")
    report_root = REPORTS_ROOT if args.group == "ops" else REPORTS_ROOT / args.group
    output_dir = args.output_dir or report_root / args.operator
    try:
        run_pipeline(
            args.operator,
            output_dir,
            argv[split + 1 :],
            group=args.group,
            gate_k=args.gate_k,
            formats=args.formats,
            quantile=args.quantile,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
