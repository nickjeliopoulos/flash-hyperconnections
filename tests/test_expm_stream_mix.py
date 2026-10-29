"""N4 exp/project/impulse fusion and its explicit precision boundaries."""

import pytest
import torch

from flash_hypercons import matrix_exp_stream_mix_project
from flash_hypercons.ops.expm_stream_mix import _composed
from flash_hypercons.testing import assert_close

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
TOL = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}


def inputs(b=7, n=4, d=131, dtype=torch.float32, device="cuda", shared=True):
    torch.manual_seed(7402 + b + d)
    a = torch.randn(b, n, n, device=device) * 0.08
    x = torch.randn(b, n, d, device=device).to(dtype)
    w = torch.randn(b, n, device=device)
    z = torch.randn(b, d, device=device)
    v = torch.full((1 if shared else b, n), n**-0.5, device=device)
    v = torch.nn.functional.normalize(v + 0.01 * torch.randn_like(v), dim=-1)
    return a, x, w, z, v


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mixing_dtype", DTYPES)
@pytest.mark.parametrize("b,d", [(1, 1), (7, 131), (256, 768), (1025, 256), (24576, 17)])
def test_fused_forward(dtype, mixing_dtype, b, d):
    values = inputs(b=b, d=d, dtype=dtype)
    actual = matrix_exp_stream_mix_project(*values, 0.99, mixing_dtype=mixing_dtype)
    expected = _composed(*values, 0.99, mixing_dtype)
    assert actual.dtype == dtype and actual.is_contiguous()
    assert_close(actual, expected, tolerance=TOL[dtype], dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["transposed", "expanded", "negative", "unaligned", "sliced"])
def test_fused_layouts(layout):
    values = inputs(shared=False)
    if layout == "transposed":
        values = tuple(t.mT.contiguous().mT for t in values)
    elif layout == "expanded":
        values = tuple(t[:1].expand_as(t) for t in values)
    elif layout == "negative":
        values = tuple(torch._neg_view(t) for t in values)
    elif layout == "unaligned":
        values = inputs(d=768, shared=False)
        values = tuple(torch.cat((t.new_zeros(1), t.flatten()))[1:].view(t.shape) for t in values)
    else:
        values = tuple(torch.stack((t, t), -1)[..., 0] for t in values)
    # Independent input dtypes, including BF16 layer-output preparation.
    a_dtype = torch.bfloat16 if layout == "negative" else torch.float16
    coefficient_types = (a_dtype, values[1].dtype, torch.float16, torch.bfloat16, torch.bfloat16)
    values = tuple(t.to(dtype) for t, dtype in zip(values, coefficient_types, strict=True))
    for decay in (0.0, 1.0):
        actual = matrix_exp_stream_mix_project(*values, decay, mixing_dtype=torch.bfloat16)
        assert_close(actual, _composed(*values, decay, torch.bfloat16), tolerance=2e-5, dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("b,d", [(0, 0), (0, 17), (3, 0)])
def test_fused_empty(b, d):
    result = matrix_exp_stream_mix_project(*inputs(b=b, d=d))
    assert result.shape == (b, 4, d) and result.is_contiguous()


@pytest.mark.cuda
def test_fused_scaling_and_dense_reference():
    a, x, w, z, v = inputs(b=9, d=128, shared=False)
    a *= torch.tensor([0, 0.1, 1, 10, 30, 40, 50, 60, 80], device="cuda")[:, None, None]
    # Skew matrices exercise several squarings without exponential overflow.
    a = a - a.mT
    actual = matrix_exp_stream_mix_project(a, x, w, z, v, 0.9)
    p = torch.linalg.matrix_exp(a.double())
    projection = v.double().unsqueeze(-1) * v.double().unsqueeze(-2)
    expected = (p @ (torch.eye(4, device="cuda") - projection) + 0.9 * projection) @ x.double()
    expected += (w.unsqueeze(-1) * z.unsqueeze(-2)).double()
    assert_close(actual, expected, tolerance=2e-5, dim=(-2, -1))
    assert_close(actual, _composed(a, x, w, z, v, 0.9), tolerance=2e-5, dim=(-2, -1))


@pytest.mark.cuda
def test_fused_nonfinite_isolation():
    values = inputs(b=5, d=32)
    values[0][1, 0, 0] = float("nan")
    values[0][3, 2, 1] = float("inf")
    out = matrix_exp_stream_mix_project(*values)
    assert torch.isnan(out[[1, 3]]).all()
    assert torch.isfinite(out[[0, 2, 4]]).all()


@pytest.mark.cuda
def test_fused_registration_compile_graph_stream():
    values = inputs(b=33, d=128, shared=False)
    op = torch.ops.flash_hypercons._expm_stream_mix_project
    torch.library.opcheck(op, (*values, 0.99, torch.bfloat16))

    def fn(*args):
        return matrix_exp_stream_mix_project(*args, 0.99, mixing_dtype=torch.bfloat16)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        expected = fn(*values)
    assert any(e.key == "flash_hypercons::_expm_stream_mix_project" for e in prof.key_averages())
    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(*values), expected)
    torch.testing.assert_close(compiled(*inputs(b=19, d=257, shared=False)), fn(*inputs(b=19, d=257, shared=False)))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(*values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn(*values)
        values[0].mul_(2)
        values[1].add_(0.3)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result, fn(*values))


@pytest.mark.cuda
def test_fused_autocast_and_gradient_fallback():
    values = tuple(t.requires_grad_() for t in inputs(b=2, d=33))
    with torch.no_grad():
        expected = matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16)
    assert_close(actual, expected, tolerance=2e-5, dim=(-2, -1))
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, values, grad)
    ref_grads = torch.autograd.grad(_composed(*values, mixing_dtype=torch.bfloat16), values, grad)
    for got, want in zip(actual_grads, ref_grads, strict=True):
        torch.testing.assert_close(got, want)
    compiled = torch.compile(matrix_exp_stream_mix_project, fullgraph=True, options={"emulate_precision_casts": True})
    result = compiled(*values, mixing_dtype=torch.bfloat16)
    assert_close(result, expected, tolerance=2e-5, dim=(-2, -1))
    for got, want in zip(torch.autograd.grad(result, values, grad), ref_grads, strict=True):
        assert_close(got, want, tolerance=1e-4)


@pytest.mark.parametrize("n,dtype", [(4, torch.float32), (4, torch.float64), (8, torch.float32)])
def test_fused_cpu_fallback(n, dtype):
    values = tuple(t.to(dtype).requires_grad_() for t in inputs(b=2, n=n, d=7, device="cpu"))

    def fn(*args):
        return matrix_exp_stream_mix_project(*args, mixing_dtype=dtype)

    torch.testing.assert_close(fn(*values), _composed(*values, mixing_dtype=dtype))
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(fn, values, fast_mode=True)
        assert torch.autograd.gradgradcheck(fn, values, fast_mode=True)


@pytest.mark.cuda
def test_fused_transforms_and_other_n():
    for n in (4, 8):
        values = inputs(b=2, n=n, d=7)
        tangent = tuple(torch.randn_like(t) for t in values)
        got = torch.func.jvp(matrix_exp_stream_mix_project, values, tangent)
        want = torch.func.jvp(_composed, values, tangent)
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("bad", ["shape", "dtype", "decay", "mixing"])
def test_fused_invalid(bad):
    values = list(inputs(b=2, d=7, device="cpu"))
    kwargs = {}
    if bad == "shape":
        values[3] = values[3][:, :1]
    elif bad == "dtype":
        values[0] = values[0].long()
    elif bad == "decay":
        kwargs["projection_decay"] = -0.1
    else:
        kwargs["mixing_dtype"] = torch.int32
    with pytest.raises(RuntimeError):
        matrix_exp_stream_mix_project(*values, **kwargs)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_large_and_compiled_baseline(dtype):
    values = inputs(b=24576, d=768, dtype=dtype)
    compiled = torch.compile(_composed, fullgraph=True, options={"emulate_precision_casts": True})
    expected = _composed(*values, mixing_dtype=torch.bfloat16)
    assert_close(compiled(*values, mixing_dtype=torch.bfloat16), expected, tolerance=TOL[dtype], dim=(-2, -1))
    assert_close(
        matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16),
        expected,
        tolerance=TOL[dtype],
        dim=(-2, -1),
    )


@pytest.mark.cuda
def test_fused_mean_and_nonunit_direction():
    a, x, w, z, v = inputs(b=9, d=256)
    v.fill_(0.5)
    x = x[:, :1].expand_as(x).contiguous()
    w.zero_()
    torch.testing.assert_close(matrix_exp_stream_mix_project(a, x, w, z, v), x, atol=2e-6, rtol=2e-6)
    v.mul_(1.7)
    assert_close(matrix_exp_stream_mix_project(a, x, w, z, v), _composed(a, x, w, z, v), tolerance=2e-5, dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("b", [8191, 8192, 8193])
def test_fused_dispatch_boundary(b):
    values = inputs(b=b, d=768, dtype=torch.bfloat16, shared=False)
    # Force the scalar load path in the two-token CTA, including an odd batch.
    values = tuple(torch.cat((t.new_zeros(1), t.flatten()))[1:].view(t.shape) for t in values)
    actual = matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16)
    assert_close(actual, _composed(*values, mixing_dtype=torch.bfloat16), tolerance=TOL[torch.bfloat16], dim=(-2, -1))
