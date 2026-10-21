from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import flash_hypercons as fh

from .reference import check_error, reference

IMPORTED_ALIAS = torch.linalg.matrix_exp


def op_names(fn):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        result = fn()
    return result, {event.key for event in prof.key_averages()}


@pytest.mark.parametrize("shape", [(2, 4, 4), (4, 4), (1, 2, 4, 4), (2, 33, 33), (2, 0, 0), (2, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_cpu_fallback(shape, dtype):
    a = torch.randn(shape, dtype=dtype) * 0.1
    expected = torch.linalg.matrix_exp(a)
    with fh.use_backend(), fh.use_backend():
        actual, names = op_names(lambda: fh.matrix_exp(a))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(IMPORTED_ALIAS(input=a), expected)
        torch.testing.assert_close(a.matrix_exp(), expected)
    assert not any(name.startswith("flash_hypercons::") for name in names)
    assert torch.linalg.matrix_exp is IMPORTED_ALIAS


@pytest.mark.parametrize("value", [torch.ones(2, 3), torch.ones(2, 2, dtype=torch.int64), torch.ones(2)])
def test_invalid_inputs(value):
    with pytest.raises((RuntimeError, TypeError)) as stock:
        torch.linalg.matrix_exp(value)
    with fh.use_backend(), pytest.raises(type(stock.value)):
        torch.linalg.matrix_exp(value)


def test_invalid_call_signature():
    with fh.use_backend(), pytest.raises(TypeError):
        torch.linalg.matrix_exp(torch.eye(2), out=torch.empty(2, 2))


@pytest.mark.cuda
def test_scopes_aliases_and_backward():
    a = (torch.randn(3, 9, 9, device="cuda") * 0.03).requires_grad_()
    original = (torch.linalg.matrix_exp, torch.matrix_exp, torch.Tensor.matrix_exp)
    for call in (
        lambda: torch.linalg.matrix_exp(a),
        lambda: IMPORTED_ALIAS(input=a),
        lambda: torch.matrix_exp(a),
        lambda: a.matrix_exp(),
    ):
        with fh.use_backend(), fh.use_backend():
            y, names = op_names(call)
        assert "flash_hypercons::_matrix_exp" in names
        grad, names = op_names(lambda y=y: torch.autograd.grad(y.sum(), a)[0])
        assert "flash_hypercons::_frechet" in names
        check_error(y, reference(a))
        check_error(grad, reference(a.mT, torch.ones_like(a)), backward=True)
    with pytest.raises(ValueError, match="scope"), fh.use_backend():
        raise ValueError("scope")
    _, names = op_names(lambda: torch.linalg.matrix_exp(a))
    assert "flash_hypercons::_matrix_exp" not in names
    assert original == (torch.linalg.matrix_exp, torch.matrix_exp, torch.Tensor.matrix_exp)
    with fh.use_backend():
        _, names = op_names(lambda: torch.ops.aten.linalg_matrix_exp.default(a))
    assert "flash_hypercons::_matrix_exp" not in names


@pytest.mark.cuda
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((4, 4), torch.float32),
        ((1, 2, 4, 4), torch.float32),
        ((2, 33, 33), torch.float32),
        ((2, 4, 4), torch.float64),
        ((2, 4, 4), torch.complex64),
        ((2, 1, 1), torch.float32),
    ],
)
def test_cuda_fallback(shape, dtype):
    a = torch.randn(shape, device="cuda", dtype=dtype) * 0.1
    expected = torch.linalg.matrix_exp(a)
    with fh.use_backend():
        actual, names = op_names(lambda: torch.linalg.matrix_exp(a))
    torch.testing.assert_close(actual, expected)
    assert "flash_hypercons::_matrix_exp" not in names


@pytest.mark.cuda
def test_thread_local_scope():
    from torch.overrides import _get_current_function_mode

    with fh.use_backend(), ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(_get_current_function_mode).result() is None


@pytest.mark.cuda
def test_tensor_subclass_fallback():
    class SubTensor(torch.Tensor):
        pass

    a = (torch.randn(2, 4, 4, device="cuda") * 0.03).as_subclass(SubTensor)
    y, names = op_names(lambda: fh.matrix_exp(a))
    assert isinstance(y, SubTensor)
    assert "flash_hypercons::_matrix_exp" not in names
    torch.testing.assert_close(y, torch.linalg.matrix_exp(a))
