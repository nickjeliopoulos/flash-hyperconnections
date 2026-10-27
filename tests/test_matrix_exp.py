import math

import pytest
import torch

import flash_hypercons as fh

from .reference import check_error, diagonal_reference, nilpotent_reference, reference

pytestmark = pytest.mark.cuda
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", range(2, 33))
@pytest.mark.parametrize("batch", [0, 1, 35])
def test_dimensions(dtype, n, batch):
    a = (torch.randn(batch, n, n, device="cuda") * 0.1).to(dtype).mT.detach().requires_grad_()
    g = torch.randn_like(a) * 0.2
    y = fh.matrix_exp(a)
    (grad,) = torch.autograd.grad(y, a, g)
    assert y.dtype == grad.dtype == dtype
    assert y.device == grad.device == a.device
    check_error(y, reference(a))
    check_error(grad, reference(a.mT, g), backward=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [4, 9, 16, 32])
@pytest.mark.parametrize("family", ["diagonal", "nilpotent", "nonnormal"])
def test_matrix_families(dtype, n, family):
    scales = torch.tensor([0.25, 1.0, 4.0], device="cuda")[:, None, None]
    diagonal = torch.linspace(-0.5, 0.5, n, device="cuda")
    diagonal[1] = diagonal[0]  # Repeated eigenvalues exercise the divided-difference limit.
    shift = torch.diag(torch.ones(n - 1, device="cuda"), diagonal=1)
    if family == "diagonal":
        a = scales * diagonal.diag()
        oracle = diagonal_reference
    elif family == "nilpotent":
        a = scales * shift
        oracle = nilpotent_reference
    else:
        # Bounded eigenvalues with large off-diagonal entries: unlike rotations,
        # these matrices are nonnormal, but their outputs still fit in FP16.
        a = diagonal.diag() + scales * shift
        oracle = reference
    a = a.to(dtype).requires_grad_()
    expected = oracle(a.detach())
    direction = torch.randn_like(a, dtype=torch.float32)
    direction = direction / torch.linalg.matrix_norm(direction, ord=1)[:, None, None]
    for magnitude in (0.0, 1e-3, 1.0, 100.0):
        g = (direction * magnitude).to(dtype)
        y = fh.matrix_exp(a)
        (grad,) = torch.autograd.grad(y, a, g)
        expected_grad = oracle(a.detach().mT, g)
        assert y.dtype == grad.dtype == dtype
        assert y.device == grad.device == a.device
        check_error(y, expected)
        check_error(grad, expected_grad, backward=True)
        if magnitude:
            # Also compare at unit upstream scale so small gradients do not
            # disappear into the absolute component of the error criterion.
            check_error(grad / magnitude, expected_grad / magnitude, backward=True)
        else:
            torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_launch_boundaries(dtype, n):
    a = (torch.randn(1, n, n, device="cuda") * 0.1).to(dtype)
    g = torch.randn_like(a) * 0.2
    expected, expected_grad = reference(a), reference(a.mT, g)
    baseline = None
    for batch in (1, 15, 16, 17, 31, 32, 33, 127, 128, 129):
        x = a.repeat(batch, 1, 1).requires_grad_()
        y = fh.matrix_exp(x)
        (grad,) = torch.autograd.grad(y, x, g.repeat(batch, 1, 1))
        check_error(y, expected.expand_as(y))
        check_error(grad, expected_grad.expand_as(grad), backward=True)
        if baseline is None:
            baseline = (y.detach(), grad)
        torch.testing.assert_close(y, baseline[0].expand_as(y), atol=0, rtol=0)
        torch.testing.assert_close(grad, baseline[1].expand_as(grad), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_scaling_and_nonfinite(dtype, n):
    theta = 3.010066362817634
    norms = torch.tensor(
        [
            0,
            0.1,
            theta * (1 - 1e-5),
            theta * (1 + 1e-5),
            2 * theta * (1 - 1e-5),
            2 * theta * (1 + 1e-5),
            12.04028,
            24.08056,
        ],
        device="cuda",
    )
    a = torch.randn(len(norms), n, n, device="cuda")
    # Skew-symmetric inputs exercise scaling without overflowing the FP16 output.
    a = a - a.mT
    a = (a * (norms / torch.linalg.matrix_norm(a, ord=1))[:, None, None]).to(dtype).requires_grad_()
    g = torch.randn_like(a) * 0.2
    y = fh.matrix_exp(a)
    (grad,) = torch.autograd.grad(y, a, g)
    check_error(y, reference(a))
    check_error(grad, reference(a.mT, g), backward=True)

    a = torch.zeros(4, n, n, device="cuda", dtype=dtype)
    a[:3, 0, 0] = torch.tensor([float("nan"), float("inf"), -float("inf")], device="cuda", dtype=dtype)
    a.requires_grad_()
    y = fh.matrix_exp(a)
    (grad,) = torch.autograd.grad(y, a, torch.ones_like(a))
    assert y[:3].isnan().all() and grad[:3].isnan().all()
    torch.testing.assert_close(y[3], torch.eye(n, device="cuda", dtype=dtype))
    torch.testing.assert_close(grad[3], torch.ones_like(grad[3]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_negative_and_expanded_views(dtype, n):
    for transpose in (False, True):
        base = (torch.randn(1, n, n, device="cuda") * 0.03).to(dtype)
        a = torch._neg_view(base.mT if transpose else base).expand(3, n, n).requires_grad_()
        g = torch._neg_view(torch.randn_like(a))
        y = fh.matrix_exp(a)
        (grad,) = torch.autograd.grad(y, a, g)
        check_error(y, reference(a))
        check_error(grad, reference(a.mT, g), backward=True)


def test_directional_difference():
    for n in (4, 8, 16, 32):
        a = (torch.randn(2, n, n, device="cuda") * 0.03).requires_grad_()
        h, g = torch.randn_like(a) * 0.1, torch.randn_like(a) * 0.2
        (gradient,) = torch.autograd.grad(fh.matrix_exp(a), a, g)
        step = 0.005
        finite_diff = ((fh.matrix_exp(a + step * h) - fh.matrix_exp(a - step * h)) * g).sum() / (2 * step)
        torch.testing.assert_close((gradient * h).sum(), finite_diff, atol=0.001, rtol=0.005)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [2, 4, 8, 9, 16, 32])
def test_large_rotations(dtype, n):
    # Regression for low-precision polynomial/squaring intermediates: exp(A)
    # must preserve lengths for a skew-symmetric generator, even after many
    # squarings. Cover tiny/scalar kernels, padding, and a partial launch block.
    for batch in (1, 129):
        for angle in (1, 3, 10, 25, 100, 1000):
            a = torch.zeros(batch, n, n, device="cuda", dtype=dtype)
            a[:, 0, 1], a[:, 1, 0] = -angle, angle
            a.requires_grad_()
            expected = torch.eye(n, device="cuda", dtype=torch.float64).expand_as(a).clone()
            expected[:, 0, 0] = expected[:, 1, 1] = math.cos(angle)
            expected[:, 0, 1], expected[:, 1, 0] = -math.sin(angle), math.sin(angle)
            y = fh.matrix_exp(a)
            g = (0.125 * torch.eye(n, device="cuda", dtype=dtype)).expand_as(a)
            (grad,) = torch.autograd.grad(y, a, g)
            check_error(y, expected)
            check_error(grad, 0.125 * expected.mT, backward=True)
            # Measure just the rotating columns in FP64, so unaffected identity
            # columns and low-precision reductions cannot hide norm drift.
            lengths = torch.linalg.vector_norm(y.double()[..., :2], dim=-2)
            torch.testing.assert_close(
                lengths,
                torch.ones_like(lengths),
                atol=0.01 if dtype == torch.bfloat16 else 0.002,
                rtol=0,
                msg=f"Rotation norm drift: {dtype=}, {n=}, {batch=}, {angle=}",
            )
            tolerance = 0.002 if dtype == torch.float32 else 0.1
            torch.testing.assert_close(
                y.double().mT @ y.double(),
                torch.eye(n, device="cuda", dtype=torch.float64).expand_as(a),
                atol=tolerance,
                rtol=0,
            )


def test_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = (torch.randn(3, 9, 9, device="cuda") * 0.1).requires_grad_()
        y = fh.matrix_exp(a)
        (grad,) = torch.autograd.grad(y.sum(), a)
    stream.synchronize()
    check_error(y, reference(a))
    check_error(grad, reference(a.mT, torch.ones_like(a)), backward=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_noncurrent_device():
    torch.cuda.set_device(0)
    a = torch.randn(2, 4, 4, device="cuda:1") * 0.1
    y = fh.matrix_exp(a)
    assert torch.cuda.current_device() == 0
    check_error(y, reference(a))
