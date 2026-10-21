#include <Python.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace flash_hypercons {
at::Tensor matrix_exp_cuda(const at::Tensor& a);
at::Tensor frechet_cuda(const at::Tensor& a, const at::Tensor& direction);
}

TORCH_LIBRARY(flash_hypercons, m) {
  m.def("_matrix_exp(Tensor input) -> Tensor");
  m.def("_frechet(Tensor input, Tensor direction) -> Tensor");
}

TORCH_LIBRARY_IMPL(flash_hypercons, CUDA, m) {
  m.impl("_matrix_exp", &flash_hypercons::matrix_exp_cuda);
  m.impl("_frechet", &flash_hypercons::frechet_cuda);
}

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};

PyMODINIT_FUNC PyInit__C() {
  return PyModule_Create(&module);
}
