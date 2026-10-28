"""Adaptive small-matrix exponentials and scoped PyTorch routing."""

import torch
from torch.overrides import TorchFunctionMode, has_torch_function

from flash_hypercons import _C  # noqa: F401

_NATIVE = torch.ops.aten.linalg_matrix_exp.default
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_FUNCTIONS = (torch.linalg.matrix_exp, torch.matrix_exp, torch.Tensor.matrix_exp)


def _eligible(input: torch.Tensor) -> bool:
    return (
        isinstance(input, torch.Tensor)
        and input.device.type == "cuda"
        and torch.version.hip is None
        and input.layout == torch.strided
        and input.ndim == 3
        and input.shape[-1] == input.shape[-2]
        and 1 < input.shape[-1] <= 32
        and input.dtype in _DTYPES
    )


def _native(input: torch.Tensor, *, promote: bool = False) -> torch.Tensor:
    if promote and input.is_cuda and input.dtype in (torch.float16, torch.bfloat16):
        return _NATIVE(input.float()).to(input.dtype)
    return _NATIVE(input)


def _frechet_reference(input: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    n = input.shape[-1]
    a, g = input.float(), direction.float()
    zero = torch.zeros_like(a)
    block = torch.cat((torch.cat((a, g), -1), torch.cat((zero, a), -1)), -2)
    return _NATIVE(block)[..., :n, n:].to(input.dtype)


@torch.library.register_fake("flash_hypercons::_matrix_exp")
def _matrix_exp_fake(input):
    torch._check(input.device.type == "cuda")
    torch._check(input.layout == torch.strided)
    torch._check(input.ndim == 3)
    torch._check(input.dtype in _DTYPES)
    torch._check(input.shape[-1] == input.shape[-2])
    torch._check(input.shape[-1] > 1)
    torch._check(input.shape[-1] <= 32)
    return input.new_empty(input.shape)


@torch.library.register_fake("flash_hypercons::_frechet")
def _frechet_fake(input, direction):
    torch._check(direction.shape == input.shape)
    torch._check(direction.device == input.device)
    torch._check(direction.dtype == input.dtype)
    torch._check(direction.layout == torch.strided)
    return _matrix_exp_fake(input)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _backward(ctx, grad_output):
    (input,) = ctx.saved_tensors
    if (
        torch.is_grad_enabled()
        or has_torch_function((input, grad_output))
        or torch.autograd.forward_ad.unpack_dual(input).tangent is not None
        or torch.autograd.forward_ad.unpack_dual(grad_output).tangent is not None
    ):
        return _frechet_reference(input.mT, grad_output)
    return torch.ops.flash_hypercons._frechet(input.mT, grad_output)


torch.library.register_autograd("flash_hypercons::_matrix_exp", _backward, setup_context=_setup_context)


def matrix_exp(input: torch.Tensor) -> torch.Tensor:
    """Compute exp(input), accelerating CUDA [B, N, N] tensors with 2 <= N <= 32.

    The CUDA implementation accepts float32, float16, and bfloat16. Other
    inputs use native PyTorch. Forward AD, function transforms, and tensor
    subclasses also use native operations; their CUDA low-precision inputs
    are promoted to float32 for the calculation and cast back on return.
    """
    # Check transforms before shape eligibility: vmap can hide batch dimensions.
    if isinstance(input, torch.Tensor) and (
        type(input) not in (torch.Tensor, torch.nn.Parameter)
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad.unpack_dual(input).tangent is not None
    ):
        return _native(input, promote=True)
    if not _eligible(input):
        return _native(input)
    return torch.ops.flash_hypercons._matrix_exp(input)


class _MatrixExpMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _FUNCTIONS:
            if len(args) == 1 and not kwargs:
                return matrix_exp(args[0])
            if not args and set(kwargs) == {"input"}:
                return matrix_exp(kwargs["input"])
        return func(*args, **kwargs)


def use_backend() -> TorchFunctionMode:
    """Route Python matrix-exp calls through flash-hypercons within a with block.

    Scopes nest and restore normal routing on exit, including exceptional exit.
    Enter the scope around calls to compiled functions as well as eager calls.
    Direct ATen/C++ calls are unaffected.
    """
    return _MatrixExpMode()


__all__ = ["matrix_exp", "use_backend"]
