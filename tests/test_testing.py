"""CPU checks for numerical comparison semantics shared by tests and reports."""

import pytest
import torch

from flash_hypercons.testing import absolute_error, assert_close, error_ratio

from .reference import check_error


def test_reductions_keep_bad_matrices_visible():
    actual = torch.tensor([[[4.0, 0], [0, 0]], [[1000.0, 1000], [1000, 1000]]], requires_grad=True).mT
    expected = torch.tensor([[[2.0, 0], [0, 0]], [[1000.0, 1000], [1000, 1000]]], dtype=torch.float64).mT
    dim = (-2, -1)
    assert absolute_error(actual, expected, dim=dim).tolist() == [2, 0]
    ratio = error_ratio(actual, expected, dim=dim)
    assert ratio.tolist() == pytest.approx([2 / 3, 0])
    assert ratio.dtype == torch.float64 and not ratio.requires_grad
    assert error_ratio(actual, expected).item() == pytest.approx(2 / 1001)
    assert_close(actual, expected, tolerance=0.01)  # A global scale hides the first matrix's error.
    with pytest.raises(AssertionError):
        assert_close(actual, expected, tolerance=0.01, dim=dim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_promotes_before_subtraction(dtype):
    actual = torch.tensor([60000.0], dtype=dtype)
    expected = -actual
    assert absolute_error(actual, expected).item() == 2 * actual.item()
    assert torch.isfinite(error_ratio(actual, expected))


def test_zero_reference_and_tolerance_boundary():
    actual, expected = torch.tensor([0.0, 0.5]), torch.zeros(2)
    assert error_ratio(actual, expected).item() == 0.5
    assert_close(actual, expected, tolerance=0.5)
    with pytest.raises(AssertionError):
        assert_close(actual, expected, tolerance=0.49)


@pytest.mark.parametrize(
    "actual,expected", [(float("nan"), 0), (float("inf"), 0), (0, float("inf")), (float("inf"), float("inf"))]
)
def test_nonfinite_values_fail(actual, expected):
    with pytest.raises(AssertionError):
        assert_close(torch.tensor([actual]), torch.tensor([expected]), tolerance=1)


@pytest.mark.parametrize("shape,error_shape", [((0, 2, 2), (0,)), ((3, 0, 2), (3,))])
def test_empty_reductions(shape, error_shape):
    value = torch.empty(shape)
    assert absolute_error(value, value).item() == 0
    assert error_ratio(value, value, dim=(-2, -1)).shape == error_shape
    assert_close(value, value, tolerance=0, dim=(-2, -1))


def test_complex_error_preserves_imaginary_component():
    actual, expected = torch.tensor([2j]), torch.tensor([1j])
    assert absolute_error(actual, expected).item() == 1
    assert error_ratio(actual, expected).item() == 0.5


@pytest.mark.parametrize("fn", [absolute_error, error_ratio, assert_close])
def test_comparisons_reject_broadcasting(fn):
    kwargs = {"tolerance": 1} if fn is assert_close else {}
    with pytest.raises(ValueError, match="Shape mismatch"):
        fn(torch.zeros(1, 2), torch.zeros(2), **kwargs)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="Tolerance"):
        assert_close(torch.ones(1), torch.ones(1), tolerance=tolerance)


@pytest.mark.parametrize(
    "dtype,forward,backward", [(torch.float32, 0.001, 0.005), (torch.float16, 0.05, 0.1), (torch.bfloat16, 0.05, 0.1)]
)
def test_matrix_exp_tolerances_are_preserved(dtype, forward, backward):
    expected = torch.zeros(1, 2, 2, dtype=torch.float64)
    for is_backward, threshold in ((False, forward), (True, backward)):
        check_error(torch.full_like(expected, threshold * 0.9, dtype=dtype), expected, backward=is_backward)
        with pytest.raises(AssertionError):
            check_error(torch.full_like(expected, threshold * 1.1, dtype=dtype), expected, backward=is_backward)
