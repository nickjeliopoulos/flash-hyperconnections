"""Exp/project/impulse composition, explicit precision boundaries, and compilation."""

import pytest
import torch

from flash_hypercons import matrix_exp_stream_mix_project
from flash_hypercons.ops.expm_stream_mix import _composed
from flash_hypercons.testing import assert_close

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
TOL = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}
BWD_TOL = {torch.float32: 1e-4, torch.float16: 3e-3, torch.bfloat16: 2e-2}


def assert_dispatch(ops, *, training=False):
    required = {"flash_hypercons::_matrix_exp", "flash_hypercons::_stream_mix_project"}
    if training:
        required |= {"flash_hypercons::_frechet", "flash_hypercons::_stream_mix_project_backward"}
    assert required <= ops
    assert not any(name.startswith("flash_hypercons::_expm_stream_mix_project") for name in ops)


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
@pytest.mark.parametrize(
    "b,n,d",
    [(b, n, d) for n in (4, 8, 16, 32) for b, d in ((1, 1), (7, 131), (256, 768), (1025, 256), (24576, 17))]
    + [(5, n, 17) for n in range(2, 33) if n not in (4, 8, 16, 32)],
)
def test_forward(dtype, mixing_dtype, b, n, d):
    values = inputs(b=b, n=n, d=d, dtype=dtype)
    actual = matrix_exp_stream_mix_project(*values, 0.99, mixing_dtype=mixing_dtype)
    expected = _composed(*values, 0.99, mixing_dtype)
    assert actual.dtype == dtype and actual.is_contiguous()
    assert_close(actual, expected, tolerance=TOL[dtype], dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["transposed", "expanded", "negative", "unaligned", "sliced"])
@pytest.mark.parametrize("n", [4, 8, 15, 16, 31, 32])
def test_layouts(layout, n):
    values = inputs(n=n, dtype=torch.bfloat16, shared=False)
    if layout == "transposed":
        values = tuple(t.mT.contiguous().mT for t in values)
    elif layout == "expanded":
        values = tuple(t[:1].expand_as(t) for t in values)
    elif layout == "negative":
        values = tuple(torch._neg_view(t) for t in values)
    elif layout == "unaligned":
        values = inputs(n=n, d=768, dtype=torch.bfloat16, shared=False)
        values = tuple(torch.cat((t.new_zeros(1), t.flatten()))[1:].view(t.shape) for t in values)
    else:
        values = tuple(torch.stack((t, t), -1)[..., 0] for t in values)
    # Independent input dtypes, including BF16 layer-output preparation.
    a_dtype = torch.bfloat16 if layout == "negative" else torch.float16
    coefficient_types = (a_dtype, values[1].dtype, torch.float16, torch.bfloat16, torch.bfloat16)
    values = tuple(t.to(dtype) for t, dtype in zip(values, coefficient_types, strict=True))
    for decay in (0.0, 1.0):
        actual = matrix_exp_stream_mix_project(*values, decay, mixing_dtype=torch.bfloat16)
        assert_close(actual, _composed(*values, decay, torch.bfloat16), tolerance=TOL[values[1].dtype], dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("b,d", [(0, 0), (0, 17), (3, 0)])
@pytest.mark.parametrize("n", [3, 4, 8, 16, 32])
def test_empty(b, d, n):
    result = matrix_exp_stream_mix_project(*inputs(b=b, n=n, d=d, dtype=torch.bfloat16))
    assert result.shape == (b, n, d) and result.is_contiguous()


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_scaling_and_dense_reference(n):
    a, x, w, z, v = inputs(b=9, n=n, d=128, dtype=torch.bfloat16, shared=False)
    a *= torch.tensor([0, 0.1, 1, 10, 30, 40, 50, 60, 80], device="cuda")[:, None, None]
    # Skew matrices exercise several squarings without exponential overflow.
    a = a - a.mT
    actual = matrix_exp_stream_mix_project(a, x, w, z, v, 0.9)
    p = torch.linalg.matrix_exp(a.double())
    projection = v.double().unsqueeze(-1) * v.double().unsqueeze(-2)
    expected = (p @ (torch.eye(n, device="cuda") - projection) + 0.9 * projection) @ x.double()
    expected += (w.unsqueeze(-1) * z.unsqueeze(-2)).double()
    assert_close(actual, expected, tolerance=TOL[x.dtype], dim=(-2, -1))
    assert_close(actual, _composed(a, x, w, z, v, 0.9), tolerance=TOL[x.dtype], dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("n", [3, 4, 8, 16, 31, 32])
def test_nonfinite_isolation(n):
    values = inputs(b=5, n=n, d=32, dtype=torch.bfloat16)
    values[0][1, 0, 0] = float("nan")
    values[0][3, 2, 1] = float("inf")
    out = matrix_exp_stream_mix_project(*values)
    assert torch.isnan(out[[1, 3]]).all()
    assert torch.isfinite(out[[0, 2, 4]]).all()


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 32])
@pytest.mark.parametrize("dtype", DTYPES)
def test_compile_graph_stream(n, dtype):
    torch.compiler.reset()
    values = inputs(b=33, n=n, d=128, dtype=dtype, shared=False)
    # Mixed residual/transition dtypes still use the separate operators.
    mixing_dtype = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16

    def fn(*args):
        return matrix_exp_stream_mix_project(*args, 0.99, mixing_dtype=mixing_dtype)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        expected = fn(*values)
    assert_dispatch({e.key for e in prof.key_averages()})
    compiled = torch.compile(fn, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
    torch.testing.assert_close(compiled(*values), expected)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        torch.testing.assert_close(compiled(*values), expected)
    assert_dispatch({e.key for e in prof.key_averages()})
    other = inputs(b=19, n=32 if n != 32 else 3, d=257, dtype=dtype, shared=False)
    torch.testing.assert_close(compiled(*other), fn(*other))
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
@pytest.mark.parametrize("n", [4, 8, 15, 16, 31, 32])
def test_autocast_and_compiled_gradients(n):
    torch.compiler.reset()
    values = tuple(t.requires_grad_() for t in inputs(b=2, n=n, d=33, dtype=torch.bfloat16))
    with torch.no_grad():
        expected = matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16), expected)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16)
    assert_close(actual, expected, tolerance=TOL[values[1].dtype], dim=(-2, -1))
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, values, grad)
    ref_grads = torch.autograd.grad(_composed(*values, mixing_dtype=torch.bfloat16), values, grad)
    for i, (got, want) in enumerate(zip(actual_grads, ref_grads, strict=True)):
        tolerance = max(BWD_TOL[got.dtype], BWD_TOL[torch.bfloat16] if i in (0, 2, 3) else 0)
        assert_close(got, want, tolerance=tolerance)
    compiled = torch.compile(matrix_exp_stream_mix_project, fullgraph=True, options={"emulate_precision_casts": True})
    result = compiled(*values, mixing_dtype=torch.bfloat16)
    assert_close(result, expected, tolerance=TOL[values[1].dtype], dim=(-2, -1))
    for i, (got, want) in enumerate(zip(torch.autograd.grad(result, values, grad), ref_grads, strict=True)):
        tolerance = max(BWD_TOL[got.dtype], BWD_TOL[torch.bfloat16] if i in (0, 2, 3) else 0)
        assert_close(got, want, tolerance=tolerance)


@pytest.mark.parametrize("n,dtype", [(4, torch.float32), (4, torch.float64), (8, torch.float32)])
def test_cpu_fallback(n, dtype):
    values = tuple(t.to(dtype).requires_grad_() for t in inputs(b=2, n=n, d=7, device="cpu"))

    def fn(*args):
        return matrix_exp_stream_mix_project(*args, mixing_dtype=dtype)

    torch.testing.assert_close(fn(*values), _composed(*values, mixing_dtype=dtype))
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(fn, values, fast_mode=True)
        assert torch.autograd.gradgradcheck(fn, values, fast_mode=True)


@pytest.mark.cuda
def test_transforms_and_other_n():
    for n in (1, 4, 8, 15, 16, 32, 33):
        values = inputs(b=2, n=n, d=7, dtype=torch.bfloat16)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = matrix_exp_stream_mix_project(*values)
        ops = {e.key for e in profile.key_averages()}
        assert not any(name.startswith("flash_hypercons::_expm_stream_mix_project") for name in ops)
        if 2 <= n <= 32:
            assert_dispatch(ops)
        assert_close(actual, _composed(*values), tolerance=TOL[values[1].dtype], dim=(-2, -1))
        tangent = tuple(torch.randn_like(t) for t in values)
        got = torch.func.jvp(matrix_exp_stream_mix_project, values, tangent)
        want = torch.func.jvp(_composed, values, tangent)
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("bad", ["shape", "dtype", "decay", "mixing"])
def test_invalid(bad):
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
@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_and_compiled_baseline(n, dtype):
    from benchmarks.ops.expm_stream_mix import make_inputs

    torch.compiler.reset()
    values = make_inputs(24576, n, 768, dtype)
    compiled = torch.compile(_composed, fullgraph=True, options={"emulate_precision_casts": True})
    expected = _composed(*values, mixing_dtype=torch.bfloat16)
    assert_close(compiled(*values, mixing_dtype=torch.bfloat16), expected, tolerance=TOL[dtype], dim=(-2, -1))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16)
    ops = {e.key for e in profile.key_averages()}
    assert_dispatch(ops)
    assert_close(actual, expected, tolerance=TOL[dtype], dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("n", [3, 4, 8, 16, 31, 32])
def test_mean_and_nonunit_direction(n):
    a, x, w, z, v = inputs(b=9, n=n, d=256, dtype=torch.bfloat16)
    v.fill_(n**-0.5)
    x = x[:, :1].expand_as(x).contiguous()
    w.zero_()
    torch.testing.assert_close(matrix_exp_stream_mix_project(a, x, w, z, v), x, atol=2e-6, rtol=2e-6)
    v.mul_(1.7)
    assert_close(matrix_exp_stream_mix_project(a, x, w, z, v), _composed(a, x, w, z, v), tolerance=TOL[x.dtype], dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("b", [8191, 8192, 8193])
def test_unaligned_large_batches(b):
    values = inputs(b=b, d=768, dtype=torch.bfloat16, shared=False)
    # Unaligned residual storage around the former fused launch boundary.
    values = tuple(torch.cat((t.new_zeros(1), t.flatten()))[1:].view(t.shape) for t in values)
    actual = matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16)
    assert_close(actual, _composed(*values, mixing_dtype=torch.bfloat16), tolerance=TOL[torch.bfloat16], dim=(-2, -1))


@pytest.mark.cuda
@pytest.mark.parametrize("n", range(2, 33))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mixing_dtype", DTYPES)
def test_backward(n, dtype, mixing_dtype):
    values = tuple(t.requires_grad_() for t in inputs(b=33, n=n, d=133, dtype=dtype, shared=n % 2 == 0))
    out = matrix_exp_stream_mix_project(*values, 0.93, mixing_dtype=mixing_dtype)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, values, grad)
    reference = _composed(*values, 0.93, mixing_dtype)
    assert_close(out, reference, tolerance=TOL[dtype], dim=(-2, -1))
    expected = torch.autograd.grad(reference, values, grad)
    for i, (got, want, original) in enumerate(zip(actual, expected, values, strict=True)):
        # dA/dw/dz traverse explicit low-precision gradient cast boundaries.
        tolerance = max(BWD_TOL[original.dtype], BWD_TOL[mixing_dtype] if i in (0, 2, 3) else 0)
        assert got.dtype == original.dtype and got.shape == original.shape
        assert_close(got, want, tolerance=tolerance, dim=tuple(range(1, got.ndim)))


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 32])
@pytest.mark.parametrize("dtype", DTYPES)
def test_training_compile_and_graph(n, dtype):
    torch.compiler.reset()
    values = tuple(t.requires_grad_() for t in inputs(b=33, n=n, d=37, dtype=dtype))
    grad = torch.randn_like(values[1])

    def step():
        out = matrix_exp_stream_mix_project(*values, 0.99, mixing_dtype=torch.bfloat16)
        return torch.autograd.grad(out, values, grad)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        expected = step()
    ops = {e.key for e in prof.key_averages()}
    assert_dispatch(ops, training=True)
    compiled = torch.compile(matrix_exp_stream_mix_project, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
    result = torch.autograd.grad(compiled(*values, 0.99, mixing_dtype=torch.bfloat16), values, grad)
    for got, want in zip(result, expected, strict=True):
        torch.testing.assert_close(got, want)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        torch.autograd.grad(compiled(*values, 0.99, mixing_dtype=torch.bfloat16), values, grad)
    assert_dispatch({e.key for e in prof.key_averages()}, training=True)
    other = tuple(t.requires_grad_() for t in inputs(b=19, n=16 if n != 16 else 4, d=17, dtype=dtype))
    other_grad = torch.randn_like(other[1])
    result = torch.autograd.grad(compiled(*other, 0.99, mixing_dtype=torch.bfloat16), other, other_grad)
    reference = torch.autograd.grad(matrix_exp_stream_mix_project(*other, 0.99, mixing_dtype=torch.bfloat16), other, other_grad)
    for got, want in zip(result, reference, strict=True):
        torch.testing.assert_close(got, want)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = step()
        with torch.no_grad():
            values[0].mul_(1.2)
            values[1].add_(0.2)
            grad.mul_(0.7)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    expected = step()
    for got, want in zip(result, expected, strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_second_derivatives(n, dtype):
    values = tuple(t.requires_grad_() for t in inputs(b=2, n=n, d=7, dtype=dtype, shared=False))
    grad = torch.randn_like(values[1], requires_grad=True)
    directions = tuple(torch.randn_like(t) for t in values)
    results = []
    for fn in (matrix_exp_stream_mix_project, _composed):
        first = torch.autograd.grad(fn(*values, 0.9), values, grad, create_graph=True)
        loss = sum((g * direction).sum() for g, direction in zip(first, directions, strict=True))
        results.append(torch.autograd.grad(loss, (*values, grad), allow_unused=True))
    for got, want in zip(*results, strict=True):
        assert (got is None) == (want is None)
        if got is not None:
            assert_close(got, want, tolerance=BWD_TOL[dtype])


@pytest.mark.cuda
@pytest.mark.parametrize("index", range(5))
def test_training_selective_gradient(index):
    values = list(inputs(b=3, n=8, d=31, dtype=torch.bfloat16))
    values[index].requires_grad_()
    grad = torch.randn_like(values[1])
    got = torch.autograd.grad(matrix_exp_stream_mix_project(*values), values[index], grad)[0]
    want = torch.autograd.grad(_composed(*values), values[index], grad)[0]
    assert_close(got, want, tolerance=BWD_TOL[got.dtype])


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 15, 16, 32])
@pytest.mark.parametrize("layout", ["transposed", "expanded", "negative", "unaligned"])
def test_training_layouts(n, layout):
    values = inputs(b=33, n=n, d=130, dtype=torch.bfloat16, shared=False)
    if layout == "transposed":
        values = tuple(t.mT.contiguous().mT for t in values)
    elif layout == "expanded":
        values = tuple(t[:1].expand_as(t) for t in values)
    elif layout == "negative":
        values = tuple(torch._neg_view(t) for t in values)
    else:
        values = tuple(torch.cat((t.new_zeros(1), t.flatten()))[1:].view(t.shape) for t in values)
    values = tuple(t.requires_grad_() for t in values)
    grad = torch.randn_like(values[1]).mT.contiguous().mT
    actual = torch.autograd.grad(matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16), values, grad)
    expected = torch.autograd.grad(_composed(*values, mixing_dtype=torch.bfloat16), values, grad)
    for i, (got, want, original) in enumerate(zip(actual, expected, values, strict=True)):
        assert_close(got, want, tolerance=2e-2 if i in (0, 2, 3) else BWD_TOL[original.dtype], dim=tuple(range(1, got.ndim)))


@pytest.mark.cuda
@pytest.mark.parametrize("b,d", [(0, 0), (0, 17), (3, 0)])
@pytest.mark.parametrize("n", [3, 4, 16, 32])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_training_empty(b, d, n, nonfinite):
    values = inputs(b=b, n=n, d=d, dtype=torch.bfloat16)
    if nonfinite and b:
        values[0][1, 0, 0] = float("nan")
    values = tuple(t.mT.contiguous().mT.requires_grad_() for t in values)
    got = torch.autograd.grad(matrix_exp_stream_mix_project(*values).sum(), values)
    want = torch.autograd.grad(_composed(*values).sum(), values)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, equal_nan=True)


@pytest.mark.cuda
@pytest.mark.parametrize("n", [3, 4, 16, 32])
def test_training_nonfinite_isolation(n):
    values = inputs(b=33, n=n, d=128, dtype=torch.bfloat16, shared=False)
    values[0][1, 0, 0] = float("nan")
    values[0][3, 1, 0] = float("inf")
    values = tuple(t.requires_grad_() for t in values)
    grad = torch.randn_like(values[1])
    actual = torch.autograd.grad(matrix_exp_stream_mix_project(*values), values, grad)
    expected = torch.autograd.grad(_composed(*values), values, grad)
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        finite = torch.tensor([b for b in range(33) if b not in (1, 3)], device="cuda")
        assert_close(got[finite], want[finite], tolerance=BWD_TOL[got.dtype], dim=tuple(range(1, got.ndim)))
        if i in (0, 1, 4):
            assert torch.isnan(got[[1, 3]]).all()
        else:
            assert_close(got, want, tolerance=BWD_TOL[got.dtype], dim=tuple(range(1, got.ndim)))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_forward_over_reverse(dtype):
    values = tuple(t.requires_grad_() for t in inputs(b=2, n=8, d=7, dtype=dtype))
    g, tangent = torch.randn_like(values[1]), torch.randn_like(values[1])
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(g, tangent)
        actual = torch.autograd.grad(matrix_exp_stream_mix_project(*values), values, dual)
        expected = torch.autograd.grad(_composed(*values), values, dual)
        for got, want in zip(actual, expected, strict=True):
            got_p, got_t = torch.autograd.forward_ad.unpack_dual(got)
            want_p, want_t = torch.autograd.forward_ad.unpack_dual(want)
            assert got_t is not None
            assert_close(got_p, want_p, tolerance=BWD_TOL[dtype])
            assert_close(got_t, want_t, tolerance=BWD_TOL[dtype])


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_training_scaling(n):
    a, x, w, z, v = inputs(b=33, n=n, d=13, dtype=torch.bfloat16, shared=False)
    a = a - a.mT
    norm = a.abs().sum(-2).amax(-1)
    targets = torch.tensor([0.01, 1, 4, 16, 64, 256, 1000], device="cuda")[torch.arange(33, device="cuda") % 7]
    a = a * (targets / norm)[:, None, None]
    values = tuple(t.requires_grad_() for t in (a, x, w, z, v * 1.3))
    grad = torch.randn_like(x)
    actual = torch.autograd.grad(matrix_exp_stream_mix_project(*values, 0.9), values, grad)
    reference = tuple(t.detach().double().requires_grad_() for t in values)
    expected = torch.autograd.grad(_composed(*reference, 0.9, torch.float64), reference, grad.double())
    for got, want in zip(actual, expected, strict=True):
        assert_close(got, want, tolerance=BWD_TOL[got.dtype], dim=tuple(range(1, got.ndim)))


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_training_low_precision_coefficients(n, dtype):
    values = tuple(t.to(dtype).requires_grad_() for t in inputs(b=65, n=n, d=256))
    grad = torch.randn_like(values[1])
    actual = torch.autograd.grad(matrix_exp_stream_mix_project(*values, mixing_dtype=dtype), values, grad)
    expected = torch.autograd.grad(_composed(*values, mixing_dtype=dtype), values, grad)
    for got, want in zip(actual, expected, strict=True):
        assert_close(got, want, tolerance=BWD_TOL[dtype], dim=tuple(range(1, got.ndim)))


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_large(n, dtype):
    from benchmarks.ops.expm_stream_mix import make_inputs

    values = tuple(t.requires_grad_() for t in make_inputs(8193, n, 768, dtype))
    grad = torch.randn_like(values[1])
    actual = torch.autograd.grad(matrix_exp_stream_mix_project(*values, mixing_dtype=torch.bfloat16), values, grad)
    expected = torch.autograd.grad(_composed(*values, mixing_dtype=torch.bfloat16), values, grad)
    for i, (got, want, original) in enumerate(zip(actual, expected, values, strict=True)):
        tolerance = max(BWD_TOL[original.dtype], BWD_TOL[torch.bfloat16] if i in (0, 2, 3) else 0)
        assert_close(got, want, tolerance=tolerance, dim=tuple(range(1, got.ndim)))


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_compiled_baseline(n, dtype):
    from benchmarks.ops.expm_stream_mix import make_inputs

    torch.compiler.reset()
    values = tuple(t.requires_grad_() for t in make_inputs(256, n, 768, dtype))
    grad = torch.randn_like(values[1])
    compiled = torch.compile(_composed, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True})
    expected = torch.autograd.grad(_composed(*values, mixing_dtype=torch.bfloat16), values, grad)
    actual = torch.autograd.grad(compiled(*values, mixing_dtype=torch.bfloat16), values, grad)
    for i, (got, want, original) in enumerate(zip(actual, expected, values, strict=True)):
        tolerance = max(BWD_TOL[original.dtype], BWD_TOL[torch.bfloat16] if i in (0, 2, 3) else 0)
        assert_close(got, want, tolerance=tolerance, dim=tuple(range(1, got.ndim)))


def test_deprecated_fused_operators_are_unregistered():
    for name in ("_expm_stream_mix_project", "_expm_stream_mix_project_training", "_expm_stream_mix_project_backward"):
        assert not hasattr(torch.ops.flash_hypercons, name)
