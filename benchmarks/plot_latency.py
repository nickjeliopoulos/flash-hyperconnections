"""Side-by-side latency bars with speedups relative to a named baseline."""

import numpy as np


def plot_latency(ax, labels, timings, *, baseline, colors=None):
    """Plot {backend: milliseconds}, ordered to match labels, with ratio labels."""
    if baseline not in timings:
        raise ValueError(f"Missing baseline: {baseline}")
    values = {name: np.asarray(times, dtype=float) for name, times in timings.items()}
    if not labels or any(v.shape != (len(labels),) or not np.all(np.isfinite(v) & (v > 0)) for v in values.values()):
        raise ValueError("Latency series must have one finite, positive value per case")
    width = 0.8 / len(values)
    positions = np.arange(len(labels))
    for i, (name, times) in enumerate(values.items()):
        bars = ax.bar(
            positions + (i - (len(values) - 1) / 2) * width,
            times * 1000,
            width,
            label=name,
            color=(colors or {}).get(name),
        )
        if name != baseline:
            ax.bar_label(bars, labels=[f"{ratio:.2f}×" for ratio in values[baseline] / times], padding=3, fontsize=9)
    combined = np.concatenate(list(values.values()))
    if combined.max() / combined.min() > 100:
        ax.set_yscale("log")
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel("Median latency (µs; lower is better)")
    ax.margins(y=0.22)
    ax.grid(axis="y", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    return ax
