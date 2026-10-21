"""Wall-time benchmarks including Python wrappers, allocations, and autograd."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 128, 1024, 65536])
    parser.add_argument("--dtype", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--backends", nargs="+", choices=["torch", "flash-hypercons"], default=["torch", "flash-hypercons"]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if min(args.n + args.batch + [args.repeat, args.trials]) < 1 or args.warmup < 0:
        parser.error("sizes, repeat, and trials must be positive; warmup must be nonnegative")

    def stock(a):
        if a.dtype in (torch.float16, torch.bfloat16):
            return torch.linalg.matrix_exp(a.float()).to(a.dtype)
        return torch.linalg.matrix_exp(a)

    backends = {"torch": stock}
    extension = None
    if "flash-hypercons" in args.backends:
        import flash_hypercons as fh
        from flash_hypercons import _C

        backends["flash-hypercons"] = fh.matrix_exp
        import inspect

        extension = {
            "version": fh.__version__,
            "sha256": hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
            "python_sha256": hashlib.sha256(Path(inspect.getfile(fh.matrix_exp)).read_bytes()).hexdigest(),
        }

    props = torch.cuda.get_device_properties(0)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "cuda": torch.version.cuda,
            "gpu": props.name,
            "capability": list(torch.cuda.get_device_capability()),
            "device_memory_bytes": props.total_memory,
            "flash_hypercons": extension,
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "method": {
            "timer": "wall clock with synchronization at trial boundaries",
            "warmup_calls": args.warmup,
            "calls_per_trial": args.repeat,
            "trials": args.trials,
            "torch_low_precision": "FP32 matrix_exp, cast back to input dtype; casts included",
            "forward": "inference (requires_grad=False)",
        },
        "results": [],
    }
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    for n in args.n:
        for batch in args.batch:
            for dtype in args.dtype:
                # The same case gets identical inputs even when running a subset.
                seed = 9041 + n * 1000003 + batch * 3 + list(dtype_map).index(dtype)
                torch.manual_seed(seed)
                a = (torch.randn(batch, n, n, device="cuda") * 0.1).to(dtype_map[dtype])
                g = torch.randn_like(a) * 0.2
                input_hash = hashlib.sha256(a.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                gradient_hash = hashlib.sha256(g.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                for mode in ("forward", "forward_backward"):
                    x = a.detach().requires_grad_(mode == "forward_backward")

                    def call(fn, x=x, mode=mode, g=g):
                        y = fn(x)
                        if mode == "forward_backward":
                            return torch.autograd.grad(y, x, g)[0]
                        return y

                    for backend in args.backends:
                        for _ in range(args.warmup):
                            call(backends[backend])
                    samples = {name: [] for name in args.backends}
                    for trial in range(args.trials):
                        order = args.backends if trial % 2 == 0 else list(reversed(args.backends))
                        for backend in order:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            for _ in range(args.repeat):
                                call(backends[backend])
                            torch.cuda.synchronize()
                            samples[backend].append((time.perf_counter() - start) * 1000 / args.repeat)
                    for backend, times in samples.items():
                        row = {
                            "n": n,
                            "batch": batch,
                            "dtype": dtype,
                            "mode": mode,
                            "backend": backend,
                            "seed": seed,
                            "input_sha256": input_hash,
                            "gradient_sha256": gradient_hash,
                            "median_ms": statistics.median(times),
                            "trial_ms": times,
                        }
                        report["results"].append(row)
                        print(
                            f"{backend:16} n={n:2} batch={batch:6} {dtype} {mode:16} {row['median_ms']:.4f} ms",
                            flush=True,
                        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
