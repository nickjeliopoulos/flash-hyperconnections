import pytest
import torch

import flash_hypercons as fh

from .reference import check_error, reference
from .test_routing import op_names

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [4, 9, 32])
def test_opcheck(dtype, n):
    a = (torch.randn(3, n, n, device="cuda") * 0.03).to(dtype).mT.detach().requires_grad_()
    torch.library.opcheck(torch.ops.flash_hypercons._matrix_exp.default, (a,))
    torch.library.opcheck(torch.ops.flash_hypercons._frechet.default, (a.detach(), torch.randn_like(a)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [4, 16, 32])
def test_higher_derivatives_and_forward_ad(dtype, n):
    a = (torch.randn(2, n, n, device="cuda") * 0.03).to(dtype).requires_grad_()
    g, h = torch.randn_like(a) * 0.1, torch.randn_like(a) * 0.1
    y = fh.matrix_exp(a)
    (first,) = torch.autograd.grad(y, a, g, create_graph=True)
    (second,) = torch.autograd.grad(first, a, h)
    ref_a = a.detach().double().requires_grad_()
    (ref_first,) = torch.autograd.grad(reference(ref_a), ref_a, g.double(), create_graph=True)
    (ref_second,) = torch.autograd.grad(ref_first, ref_a, h.double())
    check_error(first, ref_first, backward=True)
    check_error(second, ref_second, backward=True)
    for grad_enabled in (False, True):
        with torch.set_grad_enabled(grad_enabled), torch.autograd.forward_ad.dual_level(), fh.use_backend():
            dual = torch.autograd.forward_ad.make_dual(a, g)
            output, names = op_names(lambda dual=dual: torch.linalg.matrix_exp(dual))
            primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
            assert "flash_hypercons::_matrix_exp" not in names
            check_error(primal, reference(a))
            check_error(tangent, reference(a, g), backward=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_func_fallbacks(dtype):
    a = (torch.randn(2, 4, 4, device="cuda") * 0.03).to(dtype)
    g = torch.randn_like(a) * 0.1
    with fh.use_backend():
        y, tangent = torch.func.jvp(torch.linalg.matrix_exp, (a,), (g,))
        grad = torch.func.grad(lambda x: torch.linalg.matrix_exp(x).sum())(a)
    check_error(y, reference(a))
    check_error(tangent, reference(a, g), backward=True)
    check_error(grad, reference(a.mT, torch.ones_like(a)), backward=True)
    batched = a.unsqueeze(0).expand(3, *a.shape)
    check_error(torch.vmap(fh.matrix_exp)(batched), reference(batched))


@pytest.mark.parametrize("tangent_arg", ["input", "grad", "both"])
@pytest.mark.parametrize("n", [4, 16, 32])
def test_forward_over_reverse(n, tangent_arg):
    a = torch.zeros(1, n, n, device="cuda", requires_grad=True)
    identity = torch.eye(n, device="cuda").unsqueeze(0)
    with torch.autograd.forward_ad.dual_level():
        x = torch.autograd.forward_ad.make_dual(a, identity) if tangent_arg in ("input", "both") else a
        g = torch.autograd.forward_ad.make_dual(identity, identity) if tangent_arg in ("grad", "both") else identity
        (gradient,) = torch.autograd.grad(fh.matrix_exp(x), x, g)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(gradient)
        assert tangent is not None
        torch.testing.assert_close(primal, identity)
        torch.testing.assert_close(tangent, (2 if tangent_arg == "both" else 1) * identity)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_nested_func_fallbacks(dtype):
    n = 4
    diagonal = torch.linspace(-0.2, 0.2, n, device="cuda", dtype=dtype)
    a = diagonal.diag().expand(3, n, n)
    identity = torch.eye(n, device="cuda", dtype=dtype).expand_as(a)
    expected = diagonal.double().exp().diag().expand_as(a)
    gradient = torch.func.grad(lambda x: fh.matrix_exp(x).diagonal(dim1=-2, dim2=-1).sum())
    for fn in (gradient, torch.vmap(gradient)):
        primal, tangent = torch.func.jvp(fn, (a,), (identity,))
        check_error(primal, expected, backward=True)
        check_error(tangent, expected, backward=True)


@pytest.mark.compile
@pytest.mark.parametrize("transform", ["grad", "jvp", "vmap"])
def test_compiled_transform_fallbacks(transform):
    a = torch.zeros(2, 4, 4, device="cuda")
    identity = torch.eye(4, device="cuda").expand_as(a)
    if transform == "grad":
        fn = torch.func.grad(lambda x: fh.matrix_exp(x).diagonal(dim1=-2, dim2=-1).sum())
    elif transform == "jvp":

        def fn(x):
            return torch.func.jvp(fh.matrix_exp, (x,), (identity,))[1]

    else:
        fn = torch.vmap(fh.matrix_exp)
    graphs = []

    def capture(gm, inputs):
        graphs.append(gm)
        return gm.forward

    actual = torch.compile(fn, backend=capture, fullgraph=True)(a)
    torch.testing.assert_close(actual, identity)
    assert not any("flash_hypercons" in str(node.target) for gm in graphs for node in gm.graph.nodes)


@pytest.mark.compile
@pytest.mark.parametrize("scoped", [False, True])
def test_compile(scoped):
    graphs = []

    def capture(gm, inputs):
        graphs.append(gm)
        return gm.forward

    fn = torch.linalg.matrix_exp if scoped else fh.matrix_exp
    a = (torch.randn(3, 9, 9, device="cuda") * 0.03).requires_grad_()
    captured = torch.compile(fn, backend=capture, fullgraph=True)
    compiled = torch.compile(fn, fullgraph=True)
    with fh.use_backend():
        captured(a)
        y = compiled(a)
    assert any(
        node.target == torch.ops.flash_hypercons._matrix_exp.default
        or node.target == torch.ops.flash_hypercons._matrix_exp
        for gm in graphs
        for node in gm.graph.nodes
    )
    grad, names = op_names(lambda: torch.autograd.grad(y.sum(), a)[0])
    assert "flash_hypercons::_frechet" in names
    check_error(y, reference(a))
    check_error(grad, reference(a.mT, torch.ones_like(a)), backward=True)
    if scoped:
        _, names = op_names(lambda: captured(a))
        assert "flash_hypercons::_matrix_exp" not in names


def test_autocast_and_parameter():
    a = torch.nn.Parameter(torch.randn(2, 8, 8, device="cuda") * 0.03)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = fh.matrix_exp(a)
    assert y.dtype == a.dtype
    check_error(y, reference(a))
    (grad,) = torch.autograd.grad(y.sum(), a)
    check_error(grad, reference(a.mT, torch.ones_like(a)), backward=True)
