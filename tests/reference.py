import torch

from flash_hypercons.testing import assert_close


def reference(a, direction=None):
    a = a.double()
    if direction is None:
        return torch.ops.aten.linalg_matrix_exp.default(a)
    n = a.shape[-1]
    zero = torch.zeros_like(a)
    block = torch.cat((torch.cat((a, direction.double()), -1), torch.cat((zero, a), -1)), -2)
    return torch.ops.aten.linalg_matrix_exp.default(block)[..., :n, n:]


def diagonal_reference(a, direction=None):
    """Analytic exponential / Frechet derivative at a real diagonal matrix."""
    diagonal = a.double().diagonal(dim1=-2, dim2=-1)
    exponential = diagonal.exp()
    if direction is None:
        return exponential.diag_embed()
    difference = diagonal.unsqueeze(-1) - diagonal.unsqueeze(-2)
    repeated = difference == 0
    denominator = torch.where(repeated, torch.ones_like(difference), difference)
    divided_difference = exponential.unsqueeze(-2) * difference.expm1() / denominator
    divided_difference = torch.where(repeated, exponential.unsqueeze(-2), divided_difference)
    return divided_difference * direction.double()


def nilpotent_reference(a, direction=None):
    """Exact finite Taylor series, also for the nilpotent augmented matrix.

    The derivative needs terms through degree 2*N-1: differentiating only the
    degree N-1 forward polynomial would miss nonzero directional derivatives.
    """
    a = a.double()
    n = a.shape[-1]
    if direction is not None:
        zero = torch.zeros_like(a)
        a = torch.cat((torch.cat((a, direction.double()), -1), torch.cat((zero, a), -1)), -2)
    result = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype).expand_as(a).clone()
    term = result
    for degree in range(1, a.shape[-1]):
        term = term @ a / degree
        result = result + term
    return result if direction is None else result[..., :n, n:]


def check_error(actual, expected, *, backward=False):
    tolerance = (0.005 if backward else 0.001) if actual.dtype == torch.float32 else (0.1 if backward else 0.05)
    assert_close(actual, expected, tolerance=tolerance, dim=(-2, -1))
