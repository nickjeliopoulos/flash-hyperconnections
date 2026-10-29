"""Projection operator checks against an independently formed dense projector."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from flash_hypercons import stream_mix_project
from flash_hypercons.testing import assert_close

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
TOLERANCES = {torch.float32: (2e-5, 1e-4), torch.float16: (2e-3, 3e-3), torch.bfloat16: (1e-2, 2e-2)}


def make_inputs(b, n, d, dtype, device="cuda", *, mixed=False, layout="contiguous", unit=True):
    p = torch.randn(b, n, n, device=device, dtype=torch.float32) / n**0.5
    x = torch.randn(b, n, d, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    v = torch.randn(b, n, device=device)
    if unit:
        v = torch.nn.functional.normalize(v, dim=-1)
    values = [
        p.to(torch.float32 if mixed else dtype),
        x.to(dtype),
        y.to(dtype),
        v.to(torch.float32 if mixed else dtype),
    ]
    if layout == "transposed":
        values = [t.transpose(-1, -2).contiguous().transpose(-1, -2) for t in values]
    elif layout == "expanded":
        values = [t[:1].expand_as(t) for t in values]
    elif layout == "negative":
        values = [torch._neg_view(t) for t in values]
    elif layout == "sliced":
        values = [torch.stack((t, t), dim=-1)[..., 0] for t in values]
    return tuple(t.detach().requires_grad_() for t in values)


def dense_reference(phi, x, y, v, projection_decay=1.0):
    p, state, source, direction = (t.double() for t in (phi, x, y, v))
    projection = direction.unsqueeze(-1) * direction.unsqueeze(-2)
    identity = torch.eye(x.shape[1], device=x.device, dtype=torch.float64)
    return (projection_decay * projection + p @ (identity - projection)) @ state + source


def check_forward_backward(inputs, fn=stream_mix_project, decay=1.0, *, grad=None):
    phi, x, y, v = inputs
    out = fn(*inputs) if decay == 1.0 else fn(*inputs, decay)
    expected = dense_reference(*inputs, decay)
    fwd_tol, bwd_tol = TOLERANCES[x.dtype]
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert out.is_contiguous()
    assert_close(out, expected, tolerance=fwd_tol, dim=(-2, -1))
    if grad is None:
        grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, inputs, grad)
    ref_inputs = tuple(t.detach().double().requires_grad_() for t in inputs)
    reference_grads = torch.autograd.grad(dense_reference(*ref_inputs, decay), ref_inputs, grad.double())
    for actual, reference, original in zip(actual_grads, reference_grads, inputs, strict=True):
        assert actual.dtype == original.dtype
        assert_close(actual, reference, tolerance=TOLERANCES[original.dtype][1], dim=tuple(range(1, actual.ndim)))
    torch.testing.assert_close(actual_grads[2], grad.to(y.dtype), atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "n,d", [(1, 1), (2, 31), (3, 127), (4, 128), (7, 129), (8, 257), (16, 768), (31, 33), (32, 2048)]
)
def test_project_forward_backward(dtype, n, d):
    check_forward_backward(make_inputs(3, n, d, dtype))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["transposed", "expanded", "negative", "sliced"])
def test_project_layouts_and_mixed_coefficients(dtype, layout):
    check_forward_backward(make_inputs(5, 4, 131, dtype, mixed=True, layout=layout, unit=False))


@pytest.mark.cuda
@pytest.mark.parametrize("b,n,d", [(0, 4, 13), (5, 4, 0), (65537, 1, 1), (4097, 16, 1), (16384, 4, 513)])
def test_project_empty_and_grid_stride(b, n, d):
    check_forward_backward(make_inputs(b, n, d, torch.float32))


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["mean", "direction"])
def test_project_preserved_direction(kind):
    phi, x, y, v = make_inputs(3, 4, 65, torch.float32)
    if kind == "mean":
        v = torch.full_like(v, 0.5)
    state = v.unsqueeze(-1) * x[:, :1]
    torch.testing.assert_close(stream_mix_project(phi, state, y * 0, v), state, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("n,dtype", [(4, torch.float64), (4, torch.float32), (33, torch.float32)])
def test_project_cpu_fallback(n, dtype):
    inputs = make_inputs(2, n, 7, dtype, device="cpu")
    actual = stream_mix_project(*inputs)
    torch.testing.assert_close(actual.double(), dense_reference(*inputs), atol=2e-6, rtol=2e-5)
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(stream_mix_project, inputs, fast_mode=True)
        assert torch.autograd.gradgradcheck(stream_mix_project, inputs, fast_mode=True)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,n", [(torch.float64, 4), (torch.float32, 33)])
def test_project_cuda_fallback(dtype, n):
    inputs = make_inputs(2, n, 17, dtype)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        result = stream_mix_project(*inputs)
    assert not any(e.key == "flash_hypercons::_stream_mix_project" for e in prof.key_averages())
    torch.testing.assert_close(result.double(), dense_reference(*inputs), atol=2e-6, rtol=2e-5)


@pytest.mark.cuda
def test_project_registration_and_dispatch():
    inputs = make_inputs(2, 4, 17, torch.float32, layout="transposed")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        torch.autograd.grad(stream_mix_project(*inputs).sum(), inputs)
    names = {e.key for e in prof.key_averages()}
    assert {"flash_hypercons::_stream_mix_project", "flash_hypercons::_stream_mix_project_backward"} <= names
    torch.library.opcheck(torch.ops.flash_hypercons._stream_mix_project.default, inputs)
    p, x, _, v = (t.detach() for t in inputs)
    torch.library.opcheck(
        torch.ops.flash_hypercons._stream_mix_project_backward.default, (p, x, v, torch.randn_like(x))
    )


@pytest.mark.cuda
@pytest.mark.compile
def test_project_compile_forward_backward():
    from torch._dynamo.backends.common import aot_autograd
    from torch._functorch.aot_autograd import make_boxed_func

    captured = []

    def capture(gm, example_inputs):
        captured.append(gm.code)
        return gm.forward

    inputs = make_inputs(3, 4, 129, torch.bfloat16, mixed=True)
    check_forward_backward(inputs, torch.compile(stream_mix_project, backend=capture, fullgraph=True))
    assert any("_stream_mix_project" in code for code in captured)
    captured.clear()

    def capture_aot(gm, example_inputs):
        captured.append(gm.code)
        return make_boxed_func(gm.forward)

    check_forward_backward(
        inputs,
        torch.compile(
            stream_mix_project, backend=aot_autograd(fw_compiler=capture_aot, bw_compiler=capture_aot), fullgraph=True
        ),
    )
    assert any("_stream_mix_project_backward" in code for code in captured)
    check_forward_backward(inputs, torch.compile(stream_mix_project, fullgraph=True))


@pytest.mark.cuda
def test_project_higher_derivatives_and_transforms():
    inputs = make_inputs(2, 3, 5, torch.float32)
    grad = torch.randn_like(inputs[1], requires_grad=True)

    def twice(fn):
        first = torch.autograd.grad(fn(*inputs), inputs, grad, create_graph=True)
        return torch.autograd.grad(sum(t.square().sum() for t in first), (*inputs[:2], inputs[3], grad))

    for actual, reference in zip(twice(stream_mix_project), twice(dense_reference), strict=True):
        assert_close(actual, reference, tolerance=1e-4)
    tangents = tuple(torch.randn_like(t) for t in inputs)
    actual = torch.func.jvp(stream_mix_project, inputs, tangents)
    reference = torch.func.jvp(dense_reference, inputs, tangents)
    for a, r in zip(actual, reference, strict=True):
        assert_close(a, r, tolerance=2e-5)
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(t, tangent) for t, tangent in zip(inputs, tangents, strict=True)
        )
        out = torch.autograd.forward_ad.unpack_dual(stream_mix_project(*duals))
        assert_close(out.tangent, reference[1], tolerance=2e-5)
    stacked = tuple(torch.stack((t, t)) for t in inputs)
    assert_close(torch.vmap(stream_mix_project)(*stacked), torch.vmap(dense_reference)(*stacked), tolerance=2e-5)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "b,n,d,dtype",
    [
        (3, 4, 133, torch.float32),
        (128, 4, 768, torch.bfloat16),
        (512, 16, 768, torch.float32),
        (512, 32, 768, torch.float32),
        (128, 16, 768, torch.bfloat16),
        (128, 32, 768, torch.bfloat16),
    ],
)
def test_project_stream_graph_and_autocast(b, n, d, dtype):
    inputs = make_inputs(b, n, d, dtype, mixed=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        check_forward_backward(inputs)
        for _ in range(3):
            result = stream_mix_project(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = stream_mix_project(*inputs)
        grads = torch.autograd.grad(result.sum(), inputs)
    for step in range(2):
        if step:
            with torch.no_grad():
                for value in inputs:
                    value.mul_(0.75)
        graph.replay()
        ref_inputs = tuple(t.detach().double().requires_grad_() for t in inputs)
        expected = dense_reference(*ref_inputs)
        expected_grads = torch.autograd.grad(expected.sum(), ref_inputs)
        assert_close(result, expected, tolerance=TOLERANCES[dtype][0], dim=(-2, -1))
        for actual, reference, original in zip(grads, expected_grads, inputs, strict=True):
            assert_close(actual, reference, tolerance=TOLERANCES[original.dtype][1], dim=tuple(range(1, actual.ndim)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        check_forward_backward(inputs)


@pytest.mark.cuda
def test_project_noncurrent_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    inputs = make_inputs(2, 4, 131, torch.float32, device="cuda:1")
    with torch.cuda.device(0):
        check_forward_backward(inputs)
        assert torch.cuda.current_device() == 0


@pytest.mark.parametrize("bad", ["phi_shape", "v_shape", "y_shape", "zero_n", "integer", "sparse"])
def test_project_invalid_inputs(bad):
    inputs = list(make_inputs(2, 4, 7, torch.float32, device="cpu"))
    if bad == "phi_shape":
        inputs[0] = inputs[0][:, :3]
    elif bad == "v_shape":
        inputs[3] = inputs[3][:1]
    elif bad == "y_shape":
        inputs[2] = inputs[2][..., :3]
    elif bad == "zero_n":
        inputs = [t[:, :0] for t in inputs]
    elif bad == "integer":
        inputs[0] = inputs[0].int()
    elif bad == "sparse":
        inputs[0] = inputs[0].to_sparse()
    with pytest.raises(RuntimeError):
        stream_mix_project(*inputs)


@pytest.mark.cuda
def test_project_raw_validation_and_nonfinite_isolation():
    inputs = make_inputs(2, 4, 17, torch.float32)
    p, x, y, v = inputs
    with pytest.raises(RuntimeError):
        torch.ops.flash_hypercons._stream_mix_project(p, x, y, v[:1])
    with pytest.raises(RuntimeError):
        torch.ops.flash_hypercons._stream_mix_project_backward(p, x, v, y.half())
    expected = stream_mix_project(*inputs).detach()
    p = p.detach().clone()
    p[0, 1, 2] = float("nan")
    actual = stream_mix_project(p, x, y, v)
    assert actual[0, 1].isnan().all()
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [4, 16])
def test_project_triton_seed(dtype, n):
    path = os.environ.get("FLASH_STREAM_TRITON_SEED")
    if not path:
        pytest.skip("set FLASH_STREAM_TRITON_SEED to the captured v3 stream_mix_small_nb.py")
    spec = importlib.util.spec_from_file_location("stream_mix_seed", Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_forward_backward(make_inputs(3, n, 129, dtype), module.stream_mix_add_small_nb)


@pytest.mark.cuda
@pytest.mark.parametrize("decay", [0.0, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("residual_dtype", DTYPES)
def test_project_master_decay_and_residual_storage(decay, residual_dtype):
    inputs = list(make_inputs(3, 4, 133, residual_dtype, mixed=True))
    # Master casts transition coefficients to the autocast projection dtype,
    # while the normalized dynamic direction can retain FP32 parameter precision.
    inputs[0] = inputs[0].detach().to(torch.bfloat16).requires_grad_()
    inputs[2] = inputs[2].detach().to(torch.bfloat16).requires_grad_()
    check_forward_backward(tuple(inputs), decay=decay)


@pytest.mark.parametrize("decay", [-0.1, 1.1, float("nan")])
def test_project_invalid_decay(decay):
    with pytest.raises(RuntimeError):
        stream_mix_project(*make_inputs(2, 4, 7, torch.float32, device="cpu"), decay)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("decay", [0.0, 0.99])
def test_project_triton_master(dtype, decay):
    path = os.environ.get("FLASH_STREAM_TRITON_MASTER")
    if not path:
        pytest.skip("set FLASH_STREAM_TRITON_MASTER to the captured master stream_mix_small_nb.py")
    spec = importlib.util.spec_from_file_location("stream_mix_master", Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = list(make_inputs(3, 4, 129, dtype, mixed=True))
    inputs[2] = inputs[2].detach().to(torch.bfloat16).requires_grad_()
    check_forward_backward(tuple(inputs), module.stream_mix_add_small_nb, decay)


@pytest.mark.cuda
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("d", [131, 768, 2048])
@pytest.mark.parametrize("b", [128, 512])
def test_project_large_batch(n, dtype, d, b):
    check_forward_backward(make_inputs(b, n, d, dtype, mixed=True), decay=0.99)


@pytest.mark.cuda
@pytest.mark.parametrize("mean", [False, True])
def test_project_two_sided_caller_composition(mean):
    phi, x, y, v = make_inputs(3, 4, 65, torch.float32)
    if mean:
        v = torch.full((1, 4), 0.5, device="cuda", requires_grad=True)
    expanded = v.expand(x.shape[0], -1)
    # The master caller left-projects Phi and expands shared mean directions.
    q = torch.eye(4, device="cuda") - expanded.unsqueeze(-1) * expanded.unsqueeze(-2)
    result = stream_mix_project(q @ phi, x, y, expanded, 0.99)
    direction = v.double().expand(x.shape[0], -1)
    p = direction.unsqueeze(-1) * direction.unsqueeze(-2)
    q64 = torch.eye(4, device="cuda", dtype=torch.float64) - p
    expected = (0.99 * p + q64 @ phi.double() @ q64) @ x.double() + y.double()
    assert_close(result, expected, tolerance=2e-5, dim=(-2, -1))
    grad = torch.randn_like(x)
    inputs = (phi, x, y, v)
    for actual, reference in zip(
        torch.autograd.grad(result, inputs, grad),
        torch.autograd.grad(expected, inputs, grad.double()),
        strict=True,
    ):
        assert_close(actual, reference, tolerance=1e-4)


@pytest.mark.cuda
@pytest.mark.parametrize("phi_dtype", DTYPES)
@pytest.mark.parametrize("v_dtype", DTYPES)
def test_project_independent_coefficient_dtypes(phi_dtype, v_dtype):
    inputs = list(make_inputs(33, 4, 131, torch.bfloat16))
    for index, dtype in ((0, phi_dtype), (2, torch.float32), (3, v_dtype)):
        inputs[index] = inputs[index].detach().to(dtype).requires_grad_()
    check_forward_backward(tuple(inputs), decay=0.99)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [131, 768])
def test_project_source_is_not_rounded_before_addition(dtype, d):
    x = torch.ones(33, 4, d, dtype=dtype, device="cuda")
    phi = torch.eye(4, device="cuda").expand(33, 4, 4)
    v = torch.zeros(33, 4, device="cuda")
    y = torch.full(x.shape, -1 + 1e-4, device="cuda")
    torch.testing.assert_close(stream_mix_project(phi, x, y, v), (x.float() + y).to(dtype), atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("n", [4, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_project_state_parallel_to_direction(n, dtype):
    inputs = list(make_inputs(128, n, 2048, dtype, mixed=True))
    # dPhi is nearly zero here: cancellation after the feature reduction must
    # satisfy the same absolute-error behavior as the dense FP64 reference.
    scale = torch.randn(128, 1, 2048, device="cuda")
    inputs[1] = (inputs[3].detach().unsqueeze(-1) * scale).to(dtype).requires_grad_()
    check_forward_backward(tuple(inputs), decay=0.99)


@pytest.mark.cuda
def test_project_fp16_large_tensor_contraction():
    # Exercise all four WMMA tiles with FP16 X/G and FP32 parameter gradients.
    check_forward_backward(make_inputs(512, 32, 768, torch.float16, mixed=True), decay=0.99)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [4, 16])
@pytest.mark.parametrize("index", [1, 2, "grad"])
def test_project_contiguous_storage_offset(dtype, n, index):
    inputs = list(make_inputs(128, n, 768, dtype, mixed=True))
    value = inputs[1 if index == "grad" else index]
    shifted = torch.empty(value.numel() + 1, device=value.device, dtype=value.dtype)[1:].view_as(value)
    shifted.copy_(value.detach())
    assert shifted.is_contiguous() and shifted.storage_offset() == 1
    if index == "grad":
        check_forward_backward(tuple(inputs), decay=0.99, grad=shifted)
    else:
        inputs[index] = shifted.requires_grad_()
        check_forward_backward(tuple(inputs), decay=0.99)
