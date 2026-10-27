"""Numerical precision versus speedup for matched benchmark cases."""

import math

from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

if __package__:
    from .plotting import format_error_axis
else:
    from plotting import format_error_axis


def plot_precision_speedup(ax, points, *, baseline, colors=None, error_label="p99 normalized maximum error"):
    """Plot dicts with backend, label, speedup, error (null/+inf means failure).

    Failed quantiles appear on a labeled band above all finite measurements.
    The baseline's own numerical errors are shown at speedup 1.
    """
    if not points:
        raise ValueError("No precision/speedup points")
    finite = []
    for point in points:
        if not math.isfinite(point["speedup"]) or point["speedup"] <= 0:
            raise ValueError("Speedups must be finite and positive")
        error = point["error"]
        if error is not None:
            if math.isnan(error) or error < 0:
                raise ValueError("Errors must be nonnegative, null, or +inf")
            if math.isfinite(error):
                finite.append(error)
    upper = max([1e-9] + finite) * 3
    seen = set()
    failures = 0
    annotation_count = 0
    midpoint = math.sqrt(min(p["speedup"] for p in points) * max(p["speedup"] for p in points))
    for point in points:
        name, error = point["backend"], point["error"]
        failed = error is None or not math.isfinite(error)
        failures += int(failed)
        ax.scatter(
            point["speedup"],
            upper if failed else error,
            color=(colors or {}).get(name),
            marker="x" if failed else ("s" if name == baseline else "o"),
            s=65,
            label=name if name not in seen else None,
            zorder=3,
        )
        seen.add(name)
        if len(points) <= 12 and name != baseline:
            right = point["speedup"] >= midpoint
            above = annotation_count % 2 == 0
            ax.annotate(
                point["label"],
                (point["speedup"], upper if failed else error),
                xytext=(-5 if right else 5, 8 if above else -8),
                textcoords="offset points",
                ha="right" if right else "left",
                va="bottom" if above else "top",
                fontsize=8,
            )
            annotation_count += 1
    format_error_axis(ax, "y", upper * 2)
    if failures:
        ax.axhline(upper, color="#888888", linestyle=":")
        ax.text(
            0.02,
            upper,
            f"nonfinite quantile: {failures} point(s)",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            fontsize=9,
        )
    ax.axvline(1, color="#777777", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel=f"Speedup vs {baseline} (higher is faster)", ylabel=error_label)
    ax.legend(frameon=False)
    return ax
