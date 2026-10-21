#include "MatrixExpSmall.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

namespace flash_hypercons {
using at::Tensor;
using at::kFloat;
using at::kHalf;
using at::kBFloat16;
namespace {

template <int N, bool Pair>
void launch_small_matrix_exp(
    const Tensor& x, const Tensor& g, const Tensor& out) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int dtype = out.scalar_type() == kFloat ? 0 : out.scalar_type() == kBFloat16 ? 1 : 2;
  const float* direction = Pair ? g.const_data_ptr<float>() : nullptr;
  const auto batch = x.size(0);
  if constexpr (N <= 8) {
    constexpr int threads = small_matrix_exp::tiny::threads;
    const auto blocks = (batch * N + threads - 1) / threads;
    small_matrix_exp::tiny::expm_kernel<N, Pair, -1><<<blocks, threads, 0, stream>>>(
        x.const_data_ptr<float>(), direction, out.mutable_data_ptr(), dtype, batch, 0);
  } else {
#define LAUNCH_SCALAR(T) small_matrix_exp::scalar::tiled_kernel<N, T, Pair, false><<<batch, T, 0, stream>>>(x.const_data_ptr<float>(), direction, out.mutable_data_ptr(), dtype, true, 0)
    if (batch <= 32) { LAUNCH_SCALAR(256); }
    else if (N == 32 || batch <= 128) { LAUNCH_SCALAR(128); }
    else { LAUNCH_SCALAR(64); }
#undef LAUNCH_SCALAR
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor prepare_matrix(const Tensor& a, int64_t padded_n) {
  if (a.size(-1) == padded_n) {
    // Contiguous FP32 views can still carry a lazy negative bit. The kernel
    // reads raw storage, so materialize that bit before passing its pointer.
    return a.to(kFloat).contiguous().resolve_neg();
  }
  auto result = at::zeros({a.size(0), padded_n, padded_n}, a.options().dtype(kFloat));
  result.narrow(-2, 0, a.size(-1)).narrow(-1, 0, a.size(-1)).copy_(a);
  return result;
}

} // namespace

// `direction` indicates whether we are in "forward" or "backward" mode
// We materialize A (forward) or products involving an augmented matrix A* (backward)
// Based on `direction`
Tensor small_matrix_exp_cuda(const Tensor& a, const std::optional<Tensor>& direction) {
  TORCH_CHECK(a.is_cuda() && a.layout() == at::kStrided && a.dim() == 3 &&
      a.size(-1) == a.size(-2) && a.size(-1) > 1 && a.size(-1) <= 32 &&
      (a.scalar_type() == kFloat || a.scalar_type() == kHalf || a.scalar_type() == kBFloat16),
      "expected a strided CUDA [B, N, N] tensor, 2 <= N <= 32, in float32, float16, or bfloat16");
  if (direction) {
    TORCH_CHECK(direction->sizes() == a.sizes() && direction->device() == a.device() &&
        direction->scalar_type() == a.scalar_type(), "matrix exponential direction must match the input");
  }
  const c10::cuda::CUDAGuard guard(a.device());
  const auto n = a.size(-1);
  if (a.numel() == 0) {
    return at::empty(a.sizes(), a.options());
  }
  const int padded_n = n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32;
  const auto x = prepare_matrix(a, padded_n);
  const auto g = direction ? prepare_matrix(*direction, padded_n) : Tensor();
  const auto out = at::empty(x.sizes(), a.options());
#define LAUNCH(N) if (direction) launch_small_matrix_exp<N, true>(x, g, out); else launch_small_matrix_exp<N, false>(x, g, out)
  switch (padded_n) {
    case 4: { LAUNCH(4); break; }
    case 8: { LAUNCH(8); break; }
    case 16: { LAUNCH(16); break; }
    case 32: { LAUNCH(32); break; }
  }
#undef LAUNCH
  return padded_n == n ? out : out.narrow(-2, 0, n).narrow(-1, 0, n).contiguous();
}

Tensor matrix_exp_cuda(const Tensor& a) {
  return small_matrix_exp_cuda(a, std::nullopt);
}

Tensor frechet_cuda(const Tensor& a, const Tensor& direction) {
  return small_matrix_exp_cuda(a, direction);
}

} // namespace flash_hypercons
