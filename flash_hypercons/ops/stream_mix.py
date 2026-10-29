"""Stream mixing with a supplied preserved direction, using FP32 accumulation."""

import torch

from flash_hypercons import _C  # noqa: F401

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _check(phi, x, y, v, *, cuda=False):
    for t in (phi, x, y, v):
        torch._check(t.layout == torch.strided)
        torch._check(t.device == x.device)
        torch._check(t.dtype in (_DTYPES if cuda else (*_DTYPES, torch.float64)))
    torch._check(x.ndim == 3)
    torch._check(x.shape[1] >= 1)
    torch._check(phi.ndim == 3)
    torch._check(phi.shape == (x.shape[0], x.shape[1], x.shape[1]))
    torch._check(y.shape == x.shape)
    torch._check(v.ndim == 2)
    torch._check(v.shape == (x.shape[0], x.shape[1]))
    if cuda:
        torch._check(x.device.type == "cuda")
        torch._check(x.shape[1] <= 32)


def _promote(*values):
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in values) else torch.float32
    return (t.to(dtype) for t in values)


def _reference(phi, x, y, v, projection_decay=1.0):
    with torch.autocast(device_type=x.device.type, enabled=False):
        p, state, source, direction = _promote(phi, x, y, v)
        alpha = (direction.unsqueeze(-1) * state).sum(-2)
        c = projection_decay * direction - (p @ direction.unsqueeze(-1)).squeeze(-1)
        return (p @ state + c.unsqueeze(-1) * alpha.unsqueeze(-2) + source).to(x.dtype)


def _reference_backward(phi, x, v, grad, projection_decay=1.0):
    with torch.autocast(device_type=x.device.type, enabled=False):
        p, state, direction, g = _promote(phi, x, v, grad)
        alpha = (direction.unsqueeze(-1) * state).sum(-2)
        c = projection_decay * direction - (p @ direction.unsqueeze(-1)).squeeze(-1)
        beta = (c.unsqueeze(-1) * g).sum(-2)
        dp = g @ (state - direction.unsqueeze(-1) * alpha.unsqueeze(-2)).mT
        dx = p.mT @ g + direction.unsqueeze(-1) * beta.unsqueeze(-2)
        rho = (g * alpha.unsqueeze(-2)).sum(-1)
        dv = projection_decay * rho - (p.mT @ rho.unsqueeze(-1)).squeeze(-1) + (state * beta.unsqueeze(-2)).sum(-1)
        return dp.to(phi.dtype), dx.to(x.dtype), dv.to(v.dtype)


def _transformed(values):
    return torch._C._are_functorch_transforms_active() or any(
        type(t) not in (torch.Tensor, torch.nn.Parameter)
        or torch.autograd.forward_ad.unpack_dual(t).tangent is not None
        for t in values
    )


@torch.library.register_fake("flash_hypercons::_stream_mix_project")
def _fake(phi, x, y, v, projection_decay=1.0):
    torch._check(0 <= projection_decay <= 1)
    _check(phi, x, y, v, cuda=True)
    return x.new_empty(x.shape)


@torch.library.register_fake("flash_hypercons::_stream_mix_project_backward")
def _fake_backward(phi, x, v, grad, projection_decay=1.0):
    torch._check(0 <= projection_decay <= 1)
    torch._check(grad.dtype == x.dtype)
    _check(phi, x, grad, v, cuda=True)
    return phi.new_empty(phi.shape), x.new_empty(x.shape), v.new_empty(v.shape)


def _setup_context(ctx, inputs, output):
    phi, x, y, v, ctx.projection_decay = inputs
    ctx.y_dtype = y.dtype
    ctx.save_for_backward(phi, x, v)


def _backward(ctx, grad):
    phi, x, v = ctx.saved_tensors
    if not any(ctx.needs_input_grad[i] for i in (0, 1, 3)):
        return None, None, grad.to(ctx.y_dtype), None, None
    # FakeTensor subclasses used while tracing AOTAutograd are ordinary custom-op
    # inputs here. Public tensor subclasses/func transforms already use _reference.
    if torch.is_grad_enabled() or any(
        torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in (phi, x, v, grad)
    ):
        dp, dx, dv = _reference_backward(phi, x, v, grad, ctx.projection_decay)
    else:
        dp, dx, dv = torch.ops.flash_hypercons._stream_mix_project_backward(phi, x, v, grad, ctx.projection_decay)
    return tuple(
        value if needed else None
        for value, needed in zip((dp, dx, grad.to(ctx.y_dtype), dv), ctx.needs_input_grad[:4], strict=True)
    ) + (None,)


torch.library.register_autograd("flash_hypercons::_stream_mix_project", _backward, setup_context=_setup_context)


def stream_mix_project(
    phi: torch.Tensor, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, projection_decay: float = 1.0
) -> torch.Tensor:
    """Return Phi @ x + (projection_decay*v - Phi @ v) (v^T x) + y.

    Shapes are Phi [B,N,N], x/y [B,N,D], and v [B,N], on one device.
    v is supplied without normalization; use unit v for an orthogonal projection.
    Uniform v=1/sqrt(N) selects the stream mean. Expand a shared v to [B,N].
    All inputs may use independent floating dtypes. projection_decay is a fixed
    scalar in [0,1], default 1, matching hyperconnections master.
    The result has x's dtype. FP16/BF16 arithmetic accumulates in FP32.

    NVIDIA CUDA N=1..32 in FP32/FP16/BF16 uses standalone CUDA forward/backward.
    CPU, FP64, larger N, and function transforms use differentiable PyTorch.
    Higher derivatives use differentiable backward formulas. Empty batches and
    feature dimensions and noncontiguous inputs are supported. No Triton runtime
    dependency, implicit broadcasting, or module/layer integration is required.
    """
    torch._check(0 <= projection_decay <= 1)
    _check(phi, x, y, v)
    values = (phi, x, y, v)
    if _transformed(values):
        return _reference(*values, projection_decay)
    if x.is_cuda and torch.version.hip is None and x.shape[1] <= 32 and all(t.dtype in _DTYPES for t in values):
        return torch.ops.flash_hypercons._stream_mix_project(*values, projection_decay)
    return _reference(*values, projection_decay)


__all__ = ["stream_mix_project"]
