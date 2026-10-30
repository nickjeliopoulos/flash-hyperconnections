import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CUTLASS = ROOT / "third_party" / "cutlass" / "include"
if not (CUTLASS / "cute" / "tensor.hpp").is_file():
    raise RuntimeError("CUTLASS headers are missing. Run: git submodule update --init third_party/cutlass")

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0+PTX")

setup(
    ext_modules=[
        CUDAExtension(
            "flash_hypercons._C",
            # Deprecated expm_stream_mix.cu remains in the source archive, unbuilt.
            ["csrc/register.cpp", "csrc/matrix_exp.cu", "csrc/stream_mix.cu"],
            include_dirs=[str(CUTLASS)],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo", "--expt-relaxed-constexpr"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
