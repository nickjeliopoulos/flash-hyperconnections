"""Composition of small matrix exponential, projected mixing, and rank-one impulse."""

import torch

from .matrix_exp import matrix_exp
from .stream_mix import _DTYPES, stream_mix_project


def _check(a, x, w, z, v, decay, mixing_dtype):
    torch._check(0 <= decay <= 1)
    torch._check(mixing_dtype in (*_DTYPES, torch.float64))
    for t in (a, x, w, z, v):
        torch._check(t.layout == torch.strided)
        torch._check(t.device == x.device)
        torch._check(t.dtype in (*_DTYPES, torch.float64))
    torch._check(x.ndim == 3)
    b, n, d = x.shape
    torch._check(n >= 1)
    torch._check(a.shape == (b, n, n))
    torch._check(w.shape == (b, n))
    torch._check(z.shape == (b, d))
    torch._check(v.ndim == 2)
    torch._check(v.shape[1] == n)
    torch._check(v.shape[0] == 1 or v.shape[0] == b)


def _composed(a, x, w, z, v, projection_decay=1.0, mixing_dtype=torch.float32):
    # Explicit casts describe the model's autocast boundaries, independently of
    # the surrounding autocast context. A already includes the time step.
    with torch.autocast(device_type=x.device.type, enabled=False):
        dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
        phi = matrix_exp(a.to(dtype)).to(mixing_dtype)
        y = w.to(mixing_dtype).unsqueeze(-1) * z.to(mixing_dtype).unsqueeze(-2)
        return stream_mix_project(phi, x, y, v.expand(x.shape[:2]), projection_decay)


def matrix_exp_stream_mix_project(
    a: torch.Tensor,
    x: torch.Tensor,
    write_weights: torch.Tensor,
    layer_output: torch.Tensor,
    v: torch.Tensor,
    projection_decay: float = 1.0,
    *,
    mixing_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute exp(a), projected stream mixing, and a rank-one impulse.

    a [B,N,N] includes the time step; x [B,N,D], write_weights [B,N],
    layer_output [B,D], v [B,N] or [1,N]. v is supplied without normalization.
    Uniform v=1/sqrt(N) preserves the mean when projection_decay=1.

    Phi = exp(a.float()).to(mixing_dtype); Y is the outer product of weights
    and layer_output, with both operands AND its result rounded to mixing_dtype.
    This is the additive layer-output contribution to the residual streams.
    It does not implement the LM TemporalWriteback architecture's temporal
    convolutions, orthogonalization, or multiple output components.
    Return stream_mix_project(Phi, x, Y, v, projection_decay), in x.dtype.
    FP64 a uses FP64 exp in the fallback. Set mixing_dtype=torch.bfloat16
    for the BF16 transition/source boundaries in continuous-hc's mixed mode.
    The surrounding autocast context does not change this explicit policy.

    All dtypes use separate matrix_exp and stream_mix_project operations. Their
    existing CUDA kernels and autograd implementations handle eligible inputs;
    other devices/dtypes, subclasses and transforms retain their native fallbacks.
    The experimental fused CUDA implementation is deprecated and unregistered.
    This wrapper does not compile implicitly. Callers can use torch.compile with
    options={"emulate_precision_casts": True} to preserve the explicit low-precision
    boundaries while optimizing the composition. Empty/noncontiguous inputs are supported.
    """
    values = (a, x, write_weights, layer_output, v)
    _check(*values, projection_decay, mixing_dtype)
    return _composed(*values, projection_decay, mixing_dtype)


__all__ = ["matrix_exp_stream_mix_project"]
