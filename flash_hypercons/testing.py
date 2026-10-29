"""Numerical comparison helpers shared by tests and benchmark evaluation.

Errors use detached FP64 (complex128 for complex inputs) arithmetic. ``dim=None``
reduces the entire tensor; pass ``dim=(-2, -1)`` for one error per matrix. Empty
reductions have zero error. Inputs must have matching shapes; dtypes may differ.
"""

import math

import torch


def _inputs(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: actual {tuple(actual.shape)}, expected {tuple(expected.shape)}")
    dtype = torch.complex128 if actual.is_complex() or expected.is_complex() else torch.float64
    return actual.detach().to(dtype), expected.detach().to(dtype)


def _maximum(value, dim):
    return value.sum(dim=dim) if not value.numel() else value.amax(dim=dim)


def absolute_error(actual, expected, *, dim=None):
    """Return max(abs(actual - expected)) as a tensor, reduced over ``dim``."""
    actual, expected = _inputs(actual, expected)
    return _maximum((actual - expected).abs(), dim)


def error_ratio(actual, expected, *, dim=None):
    """Return max absolute error / (1 + max(abs(expected))), reduced over ``dim``.

    This is the benchmark's ``normalized_max_error`` metric. A zero reference
    gives absolute error. Nonfinite values propagate as NaN/inf for the caller
    to reject or record as failed samples; no entries are silently omitted.
    """
    actual, expected = _inputs(actual, expected)
    return absolute_error(actual, expected, dim=dim) / (1 + _maximum(expected.abs(), dim))


def assert_close(actual, expected, *, tolerance, dim=None):
    """Assert every normalized maximum error is finite and <= ``tolerance``.

    The tolerance applies to ``error_ratio``, not elementwise torch.testing
    rtol/atol. Operators choose their own tolerances and reduction dimensions.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    error = error_ratio(actual, expected, dim=dim)
    torch.testing.assert_close(error, torch.zeros_like(error), atol=tolerance, rtol=0, equal_nan=False)
