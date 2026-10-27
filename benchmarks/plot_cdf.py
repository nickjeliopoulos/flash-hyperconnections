"""Empirical CDFs of per-sample numerical errors."""

import numpy as np

if __package__:
    from .plotting import error_values, format_error_axis
else:
    from plotting import error_values, format_error_axis


def plot_cdf(ax, series, *, colors=None, error_label="Normalized maximum error"):
    """Plot {backend: errors}; null/+inf failures stay in the CDF denominator.

    Zero errors are plotted exactly, with no epsilon substitution. Each sample
    has equal weight; callers decide which workloads to pool.
    """
    values = {name: error_values(errors) for name, errors in series.items()}
    upper = max([1e-9] + [float(v[np.isfinite(v)].max(initial=0)) * 1.2 for v in values.values()])
    for name, errors in values.items():
        unique, counts = np.unique(errors[np.isfinite(errors)], return_counts=True)
        cumulative = np.cumsum(counts) / len(errors)
        failed = int(np.isinf(errors).sum())
        label = f"{name} (n={len(errors):,}, failed={failed})"
        ax.step(
            np.r_[0, unique, upper],
            np.r_[0, cumulative, cumulative[-1] if len(cumulative) else 0],
            where="post",
            label=label,
            color=(colors or {}).get(name),
            linewidth=2,
        )
    format_error_axis(ax, "x", upper)
    ax.set(xlabel=error_label, ylabel="Fraction of samples", ylim=(0, 1.025))
    ax.legend(frameon=False, fontsize=9)
    return ax
