#include <Python.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>
#include <tuple>

namespace flash_hypercons {
at::Tensor matrix_exp_cuda(const at::Tensor& a);
at::Tensor frechet_cuda(const at::Tensor& a, const at::Tensor& direction);
at::Tensor stream_mix_project_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
std::tuple<at::Tensor, at::Tensor, at::Tensor> stream_mix_project_backward_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
}

TORCH_LIBRARY(flash_hypercons, m) {
  m.def("_matrix_exp(Tensor input) -> Tensor");
  m.def("_frechet(Tensor input, Tensor direction) -> Tensor");
  m.def("_stream_mix_project(Tensor phi, Tensor x, Tensor y, Tensor v, float projection_decay=1.0) -> Tensor");
  m.def("_stream_mix_project_backward(Tensor phi, Tensor x, Tensor v, Tensor grad, float projection_decay=1.0) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_hypercons, CUDA, m) {
  m.impl("_matrix_exp", &flash_hypercons::matrix_exp_cuda);
  m.impl("_frechet", &flash_hypercons::frechet_cuda);
  m.impl("_stream_mix_project", &flash_hypercons::stream_mix_project_cuda);
  m.impl("_stream_mix_project_backward", &flash_hypercons::stream_mix_project_backward_cuda);
}

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};

PyMODINIT_FUNC PyInit__C() {
  return PyModule_Create(&module);
}
