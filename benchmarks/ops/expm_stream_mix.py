"""N4 fused exp/project/impulse forward versus the existing unfused CUDA ops."""

import argparse
import hashlib
import itertools
import json
import os
import platform
import statistics
from pathlib import Path

from benchmarks.common import REPORTS_ROOT, measure, measure_graph, write_report

TEST_FILES = ("tests/test_expm_stream_mix.py",)
REQUIRED_TEST_MARKER = "cuda"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384, 21920, 24576])
    parser.add_argument("--d", type=int, nargs="+", default=[768])
    parser.add_argument("--dtype", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32", "bf16"])
    parser.add_argument("--mixing-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--wall", action="store_true", help="Also report synchronized public-call wall time")
    parser.add_argument("--output", type=Path, default=REPORTS_ROOT / "expm_stream_mix" / "report.json")
    args = parser.parse_args()
    if min(*args.batch, *args.d, args.repeat, args.trials, args.warmup) < 1:
        parser.error("sizes and timing counts must be positive")

    import torch

    from flash_hypercons import _C, matrix_exp_stream_mix_project
    from flash_hypercons.ops.expm_stream_mix import _composed

    if not torch.cuda.is_available():
        parser.error("requires NVIDIA CUDA")
    torch.set_num_threads(1)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.backends.cuda.matmul.allow_tf32 = False
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    md = dtypes[args.mixing_dtype]
    root = Path(__file__).resolve().parents[2]
    sources = [
        "csrc/expm_stream_mix.cu",
        "csrc/MatrixExpSmall.cuh",
        "csrc/matrix_exp.cu",
        "csrc/stream_mix.cu",
        "flash_hypercons/ops/expm_stream_mix.py",
        "benchmarks/common.py",
        "benchmarks/ops/expm_stream_mix.py",
    ]
    report = {
        "schema_version": 3,
        "benchmark": {"group": "ops", "name": "expm_stream_mix"},
        "status": "running",
        "plotting": {
            "baseline": "unfused-compiled",
            "backend_notes": {
                "unfused-compiled": "Existing CUDA exp/mix plus compiled pointwise casts/impulse; precision casts preserved"
            },
        },
        "correctness": {"status": "not_run", "reason": "Use benchmarks.run for the pytest gate"},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "extension": str(_C.__file__),
            "extension_sha256": hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
            "sources": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "method": {
            "scope": "Forward under no_grad; public casts/allocations included; generator/module excluded",
            "graph": "CUDA events over graph replay; excludes CPU dispatch",
            "warmup": args.warmup,
            "calls_per_graph": args.repeat,
            "replays_per_trial": 10,
            "trials": args.trials,
            "wall": args.wall,
            "allow_tf32": False,
            "compile_options": {"emulate_precision_casts": True},
            "inputs": "Synthetic bounded skew plus negative diagonal A; FP32 w/z/perturbed shared mean v",
        },
        "results": [],
    }
    backends = {"unfused-eager": _composed, "fused": matrix_exp_stream_mix_project}
    with torch.no_grad():
        for b, d, dtype in itertools.product(args.batch, args.d, args.dtype):
            torch.compiler.reset()
            backends["unfused-compiled"] = torch.compile(
                _composed, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
            )
            torch.manual_seed(9127 + b + d)
            a = torch.randn(b, 4, 4, device="cuda").tanh() * 0.06
            a = a - a.mT - torch.diag_embed(torch.rand(b, 4, device="cuda") * 0.08)
            x = torch.randn(b, 4, d, device="cuda").to(dtypes[dtype])
            w = torch.randn(b, 4, device="cuda").tanh()
            z = torch.randn(b, d, device="cuda")
            v = torch.nn.functional.normalize(
                torch.full((1, 4), 0.5, device="cuda") + torch.randn(1, 4, device="cuda") * 0.01, dim=-1
            )
            values = (a, x, w, z, v)
            hashes = [
                hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for t in values
            ]

            def call(fn, values=values):
                return fn(*values, mixing_dtype=md)

            timings = {
                "forward_graph": measure_graph(
                    backends, call, warmup=args.warmup, repeat=args.repeat, trials=args.trials
                )
            }
            if args.wall:
                timings["forward_wall"] = measure(
                    backends, call, torch.cuda.synchronize, warmup=args.warmup, repeat=args.repeat, trials=args.trials
                )
            for mode, measured in timings.items():
                for backend, times in measured.items():
                    row = {
                        "operator": "expm_stream_mix",
                        "backend": backend,
                        "mode": mode,
                        "n": 4,
                        "batch": b,
                        "d": d,
                        "dtype": dtype,
                        "mixing_dtype": args.mixing_dtype,
                        "family": "impulse_projection",
                        "case_id": json.dumps([b, d, dtype, args.mixing_dtype, mode]),
                        "label": f"N=4 B={b} D={d}",
                        "median_ms": statistics.median(times),
                        "samples_ms": times,
                        "input_sha256": hashes,
                    }
                    report["results"].append(row)
                    print(f"{mode} B={b} D={d} {dtype} {backend}: {row['median_ms'] * 1000:.3f} us", flush=True)
            write_report(args.output, report)
    report["status"] = "complete"
    write_report(args.output, report)


if __name__ == "__main__":
    main()
