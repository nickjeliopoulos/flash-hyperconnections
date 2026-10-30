"""Public exp/project/impulse composition versus direct eager and compiled calls."""

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


def make_inputs(b, n, d, dtype, *, device="cuda"):
    """Use the original N=4 workload distribution at each requested stream count."""
    import torch

    torch.manual_seed(9127 + b + d)
    a = torch.randn(b, n, n, device=device).tanh() * 0.06
    a = a - a.mT - torch.diag_embed(torch.rand(b, n, device=device) * 0.08)
    x = torch.randn(b, n, d, device=device).to(dtype)
    w = torch.randn(b, n, device=device).tanh()
    z = torch.randn(b, d, device=device)
    v = torch.nn.functional.normalize(
        torch.full((1, n), n**-0.5, device=device) + torch.randn(1, n, device=device) * 0.01, dim=-1
    )
    return a, x, w, z, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", choices=[4, 8, 16, 32], default=[4, 8, 16, 32])
    parser.add_argument("--batch", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384, 21920, 24576])
    parser.add_argument("--d", type=int, nargs="+", default=[768])
    parser.add_argument("--dtype", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32", "bf16"])
    parser.add_argument("--mixing-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--wall", action="store_true", help="Also report synchronized public-call wall time")
    parser.add_argument("--backward", action="store_true", help="Measure forward plus gradients of all five inputs")
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
        "setup.py",
        "csrc/register.cpp",
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
                "current-op": "Public eager composition; experimental fusion is deprecated and unregistered.",
                "unfused-compiled": (
                    "CUDA exp/mix + compiled casts/impulse; precision casts preserved.\n"
                    "current-op dispatch is recorded for each workload."
                ),
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
            "scope": (
                "Forward plus backward for all five inputs; fixed upstream gradient; generator/module excluded"
                if args.backward else "Forward under no_grad; public casts/allocations included; generator/module excluded"
            ),
            "graph": "CUDA events over graph replay; excludes CPU dispatch",
            "warmup": args.warmup,
            "calls_per_graph": args.repeat,
            "replays_per_trial": 10,
            "trials": args.trials,
            "wall": args.wall,
            "allow_tf32": False,
            "compile_options": {"emulate_precision_casts": True},
            "inputs": "Synthetic bounded skew plus negative diagonal A; FP32 w/z/perturbed shared mean v",
            "stream_counts": args.n,
        },
        "results": [],
    }
    backends = {"unfused-eager": _composed, "current-op": matrix_exp_stream_mix_project}
    with torch.enable_grad() if args.backward else torch.no_grad():
        for n, b, d, dtype in itertools.product(args.n, args.batch, args.d, args.dtype):
            torch.compiler.reset()
            backends["unfused-compiled"] = torch.compile(
                _composed, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
            )
            values = make_inputs(b, n, d, dtypes[dtype])
            if args.backward:
                values = tuple(t.requires_grad_() for t in values)
                upstream = torch.randn_like(values[1])
                gradient_hash = hashlib.sha256(upstream.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            else:
                upstream, gradient_hash = None, None
            hashes = [
                hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for t in values
            ]

            def call(fn, values=values, upstream=upstream):
                out = fn(*values, mixing_dtype=md)
                return torch.autograd.grad(out, values, upstream) if args.backward else out

            # Observe actual routing outside the timed region.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                call(matrix_exp_stream_mix_project)
            public_ops = sorted(e.key for e in profile.key_averages() if e.key.startswith("flash_hypercons::"))
            public_path = (
                "fused_cuda"
                if any(name in public_ops for name in (
                    "flash_hypercons::_expm_stream_mix_project", "flash_hypercons::_expm_stream_mix_project_training"
                ))
                else "unfused_composition"
            )
            print(f"N={n} B={b} D={d} {dtype}: current-op path={public_path}", flush=True)
            mode_prefix = "forward_backward" if args.backward else "forward"
            timings = {
                mode_prefix + "_graph": measure_graph(
                    backends, call, warmup=args.warmup, repeat=args.repeat, trials=args.trials
                )
            }
            if args.wall:
                timings[mode_prefix + "_wall"] = measure(
                    backends, call, torch.cuda.synchronize, warmup=args.warmup, repeat=args.repeat, trials=args.trials
                )
            for mode, measured in timings.items():
                for backend, times in measured.items():
                    row = {
                        "operator": "expm_stream_mix",
                        "backend": backend,
                        "mode": mode,
                        "n": n,
                        "batch": b,
                        "d": d,
                        "dtype": dtype,
                        "mixing_dtype": args.mixing_dtype,
                        "family": "impulse_projection",
                        "case_id": json.dumps([n, b, d, dtype, args.mixing_dtype, mode]),
                        "label": f"N={n} B={b} D={d}",
                        "execution_path": public_path if backend == "current-op" else backend,
                        "public_ops": public_ops if backend == "current-op" else None,
                        "median_ms": statistics.median(times),
                        "samples_ms": times,
                        "input_sha256": hashes,
                        "gradient_sha256": gradient_hash,
                    }
                    report["results"].append(row)
                    print(f"{mode} N={n} B={b} D={d} {dtype} {backend}: {row['median_ms'] * 1000:.3f} us", flush=True)
            write_report(args.output, report)
    report["status"] = "complete"
    write_report(args.output, report)


if __name__ == "__main__":
    main()
