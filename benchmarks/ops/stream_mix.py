"""Projected stream mixing: public-call wall time and optional local Triton baseline."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import statistics
import sys
import types
from pathlib import Path

from benchmarks.common import REPORTS_ROOT, measure, write_report

TEST_FILES = ("tests/test_stream_mix.py",)
REQUIRED_TEST_MARKER = "cuda"


def load_triton(path):
    """Load a kernel or dispatcher in an isolated package for relative imports."""
    path = Path(path).resolve()
    package_name = "_stream_mix_benchmark_reference_" + hashlib.sha256(str(path.parent).encode()).hexdigest()[:12]
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package
    module_name = f"{package_name}.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in ("stream_mix_add", "stream_mix_add_small_nb", "stream_mix_add_big_nb"):
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"No stream mixing entry point in {path}")


def tensor_hash(t):
    import torch

    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 128, 1024])
    parser.add_argument("--d", type=int, nargs="+", default=[768])
    parser.add_argument("--dtype", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32", "bf16"])
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument(
        "--triton-source", type=Path, help="Optional captured Triton kernel or stream_mix.py dispatcher"
    )
    parser.add_argument("--mixed", action="store_true", help="FP32 Phi/v, BF16 source, selected residual dtype")
    parser.add_argument("--compiled", action="store_true", help="Also measure torch.compile of the promoted formula")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, default=REPORTS_ROOT / "stream_mix" / "report.json")
    args = parser.parse_args()
    if min(*args.n, *args.batch, *args.d, args.repeat, args.trials) < 1 or args.warmup < 0:
        parser.error("sizes, repeat and trials must be positive; warmup must be nonnegative")
    if not 0 <= args.decay <= 1:
        parser.error("decay must be in [0,1]")

    import torch

    import flash_hypercons as fh
    from flash_hypercons import _C
    from flash_hypercons.ops.stream_mix import _reference

    if not torch.cuda.is_available():
        parser.error("requires NVIDIA CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    backends = {"torch": _reference, "flash-hypercons": fh.stream_mix_project}
    seed = None
    if args.triton_source:
        backends["triton"] = load_triton(args.triton_source)
        seed = {
            "path": str(args.triton_source.resolve()),
            "sha256": hashlib.sha256(args.triton_source.read_bytes()).hexdigest(),
        }
        if args.triton_source.name == "stream_mix.py":
            seed["dependencies"] = {
                name: hashlib.sha256(args.triton_source.with_name(name).read_bytes()).hexdigest()
                for name in ("stream_mix_small_nb.py", "stream_mix_big_nb.py", "numbers.py")
            }
    report = {
        "schema_version": 3,
        "benchmark": {"group": "ops", "name": "stream_mix"},
        "status": "running",
        "plotting": {
            "baseline": "torch",
            "backend_notes": {"torch": "FP32 promoted projection formula; casts included"},
        },
        "correctness": {"status": "not_run", "reason": "Use benchmarks.run for the pytest correctness gate"},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "torch_num_threads": torch.get_num_threads(),
            "extension": str(Path(_C.__file__).resolve()),
            "extension_sha256": hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
            "cuda_source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[2] / "csrc/stream_mix.cu").read_bytes()
            ).hexdigest(),
            "python_source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[2] / "flash_hypercons/ops/stream_mix.py").read_bytes()
            ).hexdigest(),
            "triton_seed": seed,
        },
        "method": {
            "timer": "wall clock with synchronization at trial boundaries",
            "warmup_calls": args.warmup,
            "calls_per_trial": args.repeat,
            "trials": args.trials,
            "scope": "public wrapper including casts/allocations; all four gradients in forward_backward",
            "autotuning": "excluded by warmup",
            "torch_allow_tf32": False,
            "triton_precision": "Defined by the supplied source; dispatcher kernels may choose differently by dtype",
        },
        "results": [],
    }
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    for n, b, d, dtype_name in itertools.product(args.n, args.batch, args.d, args.dtype):
        torch.manual_seed(9041 + n * 100003 + b + d)
        dtype = dtype_map[dtype_name]
        p = (torch.randn(b, n, n, device="cuda") / n**0.5).to(torch.float32 if args.mixed else dtype)
        x = torch.randn(b, n, d, device="cuda", dtype=dtype)
        y = torch.randn_like(x).to(torch.bfloat16 if args.mixed else dtype)
        v = torch.nn.functional.normalize(torch.randn(b, n, device="cuda"), dim=-1).to(
            torch.float32 if args.mixed else dtype
        )
        grad = torch.randn_like(x)
        hashes = [tensor_hash(t) for t in (p, x, y, v, grad)]
        if args.compiled:
            # Each shape/dtype pair is a separate compilation workload. Warmup
            # below excludes forward and backward compilation from measurements.
            torch.compiler.reset()
            backends["torch-compiled"] = torch.compile(_reference, fullgraph=True)
        for mode in ("forward", "forward_backward"):
            inputs = tuple(t.detach().requires_grad_(mode == "forward_backward") for t in (p, x, y, v))

            def call(fn, inputs=inputs, grad=grad, mode=mode):
                out = fn(*inputs) if args.decay == 1 else fn(*inputs, args.decay)
                return torch.autograd.grad(out, inputs, grad) if mode == "forward_backward" else out

            samples = measure(
                backends, call, torch.cuda.synchronize, warmup=args.warmup, repeat=args.repeat, trials=args.trials
            )
            for backend, times in samples.items():
                row = {
                    "operator": "stream_mix",
                    "backend": backend,
                    "mode": mode,
                    "n": n,
                    "batch": b,
                    "d": d,
                    "dtype": dtype_name,
                    "family": "projection",
                    "decay": args.decay,
                    "mixed": args.mixed,
                    "case_id": json.dumps([n, b, d, dtype_name, mode, args.decay, args.mixed]),
                    "label": f"N={n} B={b} D={d}",
                    "median_ms": statistics.median(times),
                    "samples_ms": times,
                    "input_sha256": hashlib.sha256("".join(hashes[:4]).encode()).hexdigest(),
                    "gradient_sha256": hashes[4],
                    "input_gradient_sha256": hashes,
                }
                report["results"].append(row)
                print(f"{mode} N={n} B={b} D={d} {dtype_name} {backend}: {row['median_ms'] * 1000:.2f} us", flush=True)
            write_report(args.output, report)
    report["status"] = "complete"
    write_report(args.output, report)


if __name__ == "__main__":
    main()
