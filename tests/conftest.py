import pytest
import torch


def pytest_collection_modifyitems(items):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        skip = pytest.mark.skip(reason="requires NVIDIA CUDA")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(9041)
