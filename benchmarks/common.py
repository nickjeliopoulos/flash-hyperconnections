"""Small, operator-independent helpers for timing and saved reports."""

import json
import time
from pathlib import Path

REPORTS_ROOT = Path(__file__).resolve().parents[1] / ".reports"


def write_report(path, report):
    """Replace a report only after its complete JSON has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def measure(backends, call, synchronize, *, warmup, repeat, trials):
    """Time public calls, alternating backend order; return milliseconds per call."""
    for fn in backends.values():
        for _ in range(warmup):
            call(fn)
    samples = {name: [] for name in backends}
    names = list(backends)
    for trial in range(trials):
        for name in names if trial % 2 == 0 else reversed(names):
            synchronize()
            start = time.perf_counter()
            for _ in range(repeat):
                call(backends[name])
            synchronize()
            samples[name].append((time.perf_counter() - start) * 1000 / repeat)
    return samples


def measure_graph(backends, call, *, warmup, repeat, trials, replay=10):
    """CUDA-event milliseconds/call; capture public calls, excluding CPU dispatch."""
    import torch

    graphs = {}
    outputs = {}
    for name, fn in backends.items():
        for _ in range(warmup):
            call(fn)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeat):
                outputs[name] = call(fn)
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in backends}
    names = list(backends)
    for trial in range(trials):
        for name in names if trial % 2 == 0 else reversed(names):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replay):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / (repeat * replay))
    return samples
