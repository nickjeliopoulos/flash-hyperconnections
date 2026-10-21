import torch


def reference(a, direction=None):
    a = a.double()
    if direction is None:
        return torch.ops.aten.linalg_matrix_exp.default(a)
    n = a.shape[-1]
    zero = torch.zeros_like(a)
    block = torch.cat((torch.cat((a, direction.double()), -1), torch.cat((zero, a), -1)), -2)
    return torch.ops.aten.linalg_matrix_exp.default(block)[..., :n, n:]


def check_error(actual, expected, *, backward=False):
    assert actual.shape == expected.shape
    if not actual.numel():
        return
    assert actual.isfinite().all()
    tolerance = (0.005 if backward else 0.001) if actual.dtype == torch.float32 else (0.1 if backward else 0.05)
    delta = (actual.double() - expected.double()).abs().flatten(-2).amax(-1)
    scale = 1 + expected.double().abs().flatten(-2).amax(-1)
    torch.testing.assert_close(delta / scale, torch.zeros_like(delta), atol=tolerance, rtol=0)
