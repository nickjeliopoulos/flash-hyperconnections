"""Wall-time benchmarks including Python wrappers, allocations, and autograd."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import statistics
import sys
from pathlib import Path

if __package__:
    from ..common import REPORTS_ROOT, measure, write_report
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.common import REPORTS_ROOT, measure, write_report

TEST_FILES = ("tests/test_matrix_exp.py", "tests/test_routing.py", "tests/test_integration.py")
REQUIRED_TEST_MARKER = "cuda"


def tensor_hash(value):
    import torch

    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_inputs(batch, n, dtype, family, norm, seed, *, device="cuda"):
    """Generate one workload; normalization precedes input-dtype rounding."""
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(batch, n, n, generator=generator, device=device) * 0.1
    if family == "skew":
        a = a - a.mT
    elif family == "neg_psd":
        a = -(a @ a.mT)
    elif family == "diagonal":
        a = a.diagonal(dim1=-2, dim2=-1).diag_embed()
    elif family == "rotation":
        a.zero_()
        a[:, 0, 1], a[:, 1, 0] = -1, 1
    elif family != "random":
        raise ValueError(f"Unknown input family: {family}")
    if norm is not None:
        a = a * (norm / torch.linalg.matrix_norm(a, ord=1).clamp_min(1e-30))[:, None, None]
    a = a.to(dtype)
    g = torch.randn(batch, n, n, generator=generator, device=device, dtype=dtype) * 0.2
    return a, g


def normalized_errors(actual, expected):
    """Per-matrix normalized maximum errors, matching the numerical test suite."""

    actual, expected = actual.detach().double().flatten(1), expected.detach().double().flatten(1)
    actual_ok, expected_ok = actual.isfinite().all(1), expected.isfinite().all(1)
    error = (actual - expected).abs().amax(1) / (1 + expected.abs().amax(1))
    valid = actual_ok & expected_ok & error.isfinite()
    errors = [e if ok else None for e, ok in zip(error.cpu().tolist(), valid.cpu().tolist(), strict=True)]
    counts = {
        "finite": int(valid.sum().item()),
        "reference_nonfinite": int((~expected_ok).sum().item()),
        "actual_nonfinite": int((expected_ok & ~actual_ok).sum().item()),
        "nonfinite_error": int((expected_ok & actual_ok & ~valid).sum().item()),
    }
    return errors, counts


def collect_precision(a, g, backends, *, samples, chunk_size):
    """Compare both backends on the same prefix, outside all timed regions.

    FP64 native matrix_exp is the reference, including an augmented block for
    L_exp(A.T, g). Reference memory is bounded independently of timing batch size.
    """
    import torch

    count = min(samples, len(a))
    common = {
        "metric": "normalized_max_error",
        "reference": "native PyTorch FP64",
        "samples": count,
        "input_sha256": tensor_hash(a[:count]),
        "gradient_sha256": tensor_hash(g[:count]),
    }
    measurements = {
        (name, mode): {
            **common,
            "target": "output" if mode == "forward" else "input_gradient",
            "errors": [],
            "status_counts": {},
        }
        for name in backends
        for mode in ("forward", "forward_backward")
    }
    references, gradient_references = [], []
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        x = a[start:end]
        upstream = g[start:end]
        with torch.no_grad():
            double = x.double()
            expected = torch.ops.aten.linalg_matrix_exp.default(double)
            zero = torch.zeros_like(double)
            block = torch.cat((torch.cat((double.mT, upstream.double()), -1), torch.cat((zero, double.mT), -1)), -2)
            n = x.shape[-1]
            expected_grad = torch.ops.aten.linalg_matrix_exp.default(block)[..., :n, n:]
        references.append(expected)
        gradient_references.append(expected_grad.clone())
    expected, expected_grad = torch.cat(references), torch.cat(gradient_references)
    # Execute each measured backend at the original batch size: a future
    # implementation may select different kernels according to B.
    x = a.detach().requires_grad_()
    for name, fn in backends.items():
        forward_value = fn(a.detach())[:count].clone()
        y = fn(x)
        (grad,) = torch.autograd.grad(y, x, g)
        for mode, value, ref in (
            ("forward", forward_value, expected),
            ("forward_backward", grad[:count], expected_grad),
        ):
            errors, counts = normalized_errors(value, ref)
            measurements[name, mode].update(errors=errors, status_counts=counts)
    return measurements


def upgrade_report(report):
    """Supply labels for legacy matrix-exp reports; new reports are self-describing."""
    for row in report["results"]:
        row.setdefault("operator", "matrix_exp")
        row.setdefault("family", "random")
        row.setdefault(
            "case_id", json.dumps([row.get(k) for k in ("operator", "n", "batch", "dtype", "mode", "family", "norm")])
        )
        row.setdefault("label", f"N={row.get('n')} B={row.get('batch')}")
    return report


def render(report, args):
    from benchmarks.visualization import plot_report

    manifest = plot_report(report, args.plots, baseline=args.baseline, formats=args.formats, quantile=args.quantile)
    print(f"Wrote {len(manifest['artifacts'])} figures to {args.plots}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 128, 1024, 65536])
    parser.add_argument("--dtype", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--backends", nargs="+", choices=["torch", "flash-hypercons"], default=["torch", "flash-hypercons"]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, default=REPORTS_ROOT / "matrix_exp" / "report.json")
    parser.add_argument(
        "--family", nargs="+", choices=["random", "skew", "neg_psd", "diagonal", "rotation"], default=["random"]
    )
    parser.add_argument(
        "--norm", type=float, nargs="+", help="Target maximum absolute column sum, 0 to 1000, before dtype rounding"
    )
    parser.add_argument(
        "--precision", action="store_true", help="Collect per-matrix FP64-reference errors, outside timing"
    )
    parser.add_argument(
        "--precision-samples", type=int, default=256, help="First min(B, samples) matrices of each timed case"
    )
    parser.add_argument(
        "--reference-chunk", type=int, default=32, help="Maximum matrices per FP64 reference calculation"
    )
    parser.add_argument("--plots", type=Path, help="Figure output directory; enables precision collection for new runs")
    parser.add_argument(
        "--plot-from", type=Path, help="Render an existing JSON report without importing PyTorch or using CUDA"
    )
    parser.add_argument("--baseline", default="torch", help="Backend name used for plotted speedup ratios")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf", "svg"])
    parser.add_argument(
        "--quantile", type=float, default=0.99, help="Precision quantile for scatter plots; 1 means maximum"
    )
    args = parser.parse_args()
    if not 0 < args.quantile <= 1:
        parser.error("quantile must be in (0, 1]")
    if args.plot_from:
        if args.plots is None:
            parser.error("--plot-from requires --plots")
        render(upgrade_report(json.loads(args.plot_from.read_text())), args)
        return
    if min(args.n + args.batch + [args.repeat, args.trials]) < 1 or args.warmup < 0:
        parser.error("sizes, repeat, and trials must be positive; warmup must be nonnegative")
    if args.precision_samples < 1 or args.reference_chunk < 1:
        parser.error("precision sample and reference chunk counts must be positive")
    if args.norm is not None and any(not math.isfinite(v) or not 0 <= v <= 1000 for v in args.norm):
        parser.error("column norms must be between 0 and 1000")
    if args.norm is None and args.family != ["random"]:
        parser.error("Specify --norm for non-default input families")
    if "rotation" in args.family and min(args.n) < 2:
        parser.error("Rotation generators require N >= 2")
    if args.plots and args.baseline not in args.backends:
        parser.error("Plot baseline must be among --backends")
    for field in ("n", "batch", "dtype", "backends", "family", "norm"):
        values = getattr(args, field)
        if values is not None and len(values) != len(set(values)):
            parser.error(f"Duplicate --{field} values")

    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("Measurement requires NVIDIA CUDA; use --plot-from for CPU-only plotting")

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
        "schema_version": 3,
        "benchmark": {"group": "ops", "name": "matrix_exp"},
        "status": "running",
        "plotting": {
            "baseline": args.baseline,
            "backend_notes": {"torch": "FP16/BF16: FP32 matrix_exp plus casts, included in timing."},
        },
        "correctness": {
            "status": "not_run",
            "reason": "Standalone measurement; use benchmarks.run for the correctness gate",
        },
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
            "forward_backward": "forward plus autograd.grad; precision measures input-gradient error",
            "precision": {
                "enabled": bool(args.precision or args.plots),
                "max_samples_per_case": args.precision_samples,
                "sampling": "prefix of the exact timed inputs and upstream gradients",
                "reference": "native PyTorch FP64 matrix_exp; augmented block for Frechet derivative",
                "metric": "max(abs(actual-reference)) / (1 + max(abs(reference))) per matrix",
                "failures": "JSON null; included in CDF denominator and empirical quantiles",
                "reference_chunk": args.reference_chunk,
            },
        },
        "results": [],
    }
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    for family, norm in itertools.product(args.family, args.norm if args.norm is not None else [None]):
        for n, batch, dtype in itertools.product(args.n, args.batch, args.dtype):
            # The same case gets identical inputs even when running a subset.
            seed = 9041 + n * 1000003 + batch * 3 + list(dtype_map).index(dtype)
            a, g = make_inputs(batch, n, dtype_map[dtype], family, norm, seed)
            input_hash, gradient_hash = tensor_hash(a), tensor_hash(g)
            for mode in ("forward", "forward_backward"):
                x = a.detach().requires_grad_(mode == "forward_backward")

                def call(fn, x=x, mode=mode, g=g):
                    y = fn(x)
                    if mode == "forward_backward":
                        return torch.autograd.grad(y, x, g)[0]
                    return y

                samples = measure(
                    {name: backends[name] for name in args.backends},
                    call,
                    torch.cuda.synchronize,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    trials=args.trials,
                )
                for backend, times in samples.items():
                    row = {
                        "n": n,
                        "batch": batch,
                        "dtype": dtype,
                        "mode": mode,
                        "backend": backend,
                        "operator": "matrix_exp",
                        "family": family,
                        "norm": norm,
                        "case_id": json.dumps(["matrix_exp", n, batch, dtype, mode, family, norm]),
                        "label": f"N={n} B={batch}" + (f" norm={norm:g}" if norm is not None else ""),
                        "seed": seed,
                        "input_sha256": input_hash,
                        "gradient_sha256": gradient_hash,
                        "median_ms": statistics.median(times),
                        "trial_ms": times,
                    }
                    report["results"].append(row)
                    print(
                        f"{backend:16} n={n:2} batch={batch:6} {dtype} {mode:16} {family} norm={norm} {row['median_ms']:.4f} ms",
                        flush=True,
                    )
            if args.precision or args.plots:
                precision = collect_precision(
                    a,
                    g,
                    {name: backends[name] for name in args.backends},
                    samples=args.precision_samples,
                    chunk_size=args.reference_chunk,
                )
                for row in report["results"][-2 * len(args.backends) :]:
                    row["precision"] = precision[row["backend"], row["mode"]]
            write_report(args.output, report)
    report["status"] = "complete"
    write_report(args.output, report)
    if args.plots:
        render(report, args)


if __name__ == "__main__":
    main()
