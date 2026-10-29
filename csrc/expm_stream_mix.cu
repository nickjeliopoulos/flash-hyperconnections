// N=4 forward fusion. Phi lives in shared memory; the impulse is formed in registers.
#include "MatrixExpSmall.cuh"
#include <ATen/Dispatch.h>
#include <algorithm>
#include <cstdint>

namespace flash_hypercons {
namespace {
namespace tiny = small_matrix_exp::tiny;

template <typename T, int V>
struct alignas(sizeof(T) * V) Vector { T values[V]; };

__device__ __forceinline__ float round_mix(float x, int dtype) {
  if (dtype == 1) return float(at::Half(x));
  if (dtype == 2) return float(at::BFloat16(x));
  return x;
}

// Exactly the same t18/scaling arithmetic as the standalone tiny exp operator.
__device__ __forceinline__ void transition(const float* a, float* p, int dtype, int row, unsigned mask) {
  tiny::Tile<4, false> tile;
  bool bad = false;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    tile.d[j] = a[row * 4 + j];
    bad |= !::isfinite(tile.d[j]);
  }
  if (__any_sync(mask, bad)) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) p[row * 5 + j] = NAN;
    return;
  }
  double norm = 0.;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    double col = ::fabs(double(tile.d[j]));
    col += __shfl_xor_sync(mask, col, 2, 4);
    col += __shfl_xor_sync(mask, col, 1, 4);
    norm = ::fmax(norm, col);
  }
  const int s = norm > tiny::theta18 ? int(::ceil(::log2(norm / tiny::theta18))) : 0;
  const float scale = ::ldexpf(1.f, -s);
  #pragma unroll
  for (int j = 0; j < 4; ++j) tile.d[j] *= scale;
  auto result = tiny::t18(tile, mask, row);
  for (int i = 0; i < s; ++i) result = tiny::product(result, result, mask);
  #pragma unroll
  for (int j = 0; j < 4; ++j) p[row * 5 + j] = round_mix(result.d[j], dtype);
}

template <typename T, int Group, int V>
__global__ __launch_bounds__(128) void fused_forward(const float* a, const T* x,
    const float* w, const float* z, const float* v, T* out,
    int64_t batch, int64_t d, int64_t v_stride, int dtype, float decay) {
  __shared__ float state[128 / Group][32];
  const int group = threadIdx.x / Group;
  const int lane = threadIdx.x % Group;
  float* p = state[group];
  float* direction = p + 20;
  float* weights = p + 24;
  float* c = p + 28;
  const int64_t b = int64_t(blockIdx.x) * (128 / Group) + group;
  if (b < batch && lane < 4) {
    direction[lane] = v[b * v_stride + lane];
    weights[lane] = round_mix(w[b * 4 + lane], dtype);
    transition(a + b * 16, p, dtype, lane, 0xfu);
  }
  __syncthreads();
  if (b < batch && lane < 4) {
    float pv = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) pv = fmaf(p[lane * 5 + j], direction[j], pv);
    c[lane] = decay * direction[lane] - pv;
  }
  __syncthreads();
  if (b >= batch) return;
  for (int64_t col = lane * V; col < d; col += Group * V) {
    float values[4][V], alpha[V] = {};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      auto pack = *reinterpret_cast<const Vector<T, V>*>(x + (b * 4 + j) * d + col);
      #pragma unroll
      for (int k = 0; k < V; ++k) {
        values[j][k] = float(pack.values[k]);
        alpha[k] = fmaf(direction[j], values[j][k], alpha[k]);
      }
    }
    auto source = *reinterpret_cast<const Vector<float, V>*>(z + b * d + col);
    #pragma unroll
    for (int k = 0; k < V; ++k) source.values[k] = round_mix(source.values[k], dtype);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float acc[V] = {};
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float coefficient = p[i * 5 + j];
        #pragma unroll
        for (int k = 0; k < V; ++k) acc[k] = fmaf(coefficient, values[j][k], acc[k]);
      }
      Vector<T, V> result;
      #pragma unroll
      for (int k = 0; k < V; ++k) {
        // Keep product rounding and the final addition separate (no contraction).
        const float y = round_mix(__fmul_rn(weights[i], source.values[k]), dtype);
        result.values[k] = T(fmaf(c[i], alpha[k], acc[k]) + y);
      }
      *reinterpret_cast<Vector<T, V>*>(out + (b * 4 + i) * d + col) = result;
    }
  }
}

bool supported(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

template <typename T, int Group>
void launch(const at::Tensor& a, const at::Tensor& x, const at::Tensor& w,
    const at::Tensor& z, const at::Tensor& v, at::Tensor& out,
    int64_t b, int64_t d, int64_t v_stride, int dtype, float decay) {
  const int blocks = (b + 128 / Group - 1) / (128 / Group);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool vector = d % 4 == 0 && uintptr_t(x.data_ptr()) % (4 * sizeof(T)) == 0
      && uintptr_t(z.data_ptr()) % 16 == 0;
  if (vector)
    fused_forward<T, Group, 4><<<blocks, 128, 0, stream>>>(a.data_ptr<float>(), x.data_ptr<T>(),
        w.data_ptr<float>(), z.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<T>(), b, d, v_stride, dtype, decay);
  else
    fused_forward<T, Group, 1><<<blocks, 128, 0, stream>>>(a.data_ptr<float>(), x.data_ptr<T>(),
        w.data_ptr<float>(), z.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<T>(), b, d, v_stride, dtype, decay);
}
} // namespace

at::Tensor expm_stream_mix_project_cuda(const at::Tensor& a, const at::Tensor& x,
    const at::Tensor& w, const at::Tensor& z, const at::Tensor& v,
    double decay, at::ScalarType mixing_dtype) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.size(1) == 4, "x must be CUDA [B,4,D]");
  TORCH_CHECK(decay >= 0 && decay <= 1, "projection_decay must be in [0,1]");
  TORCH_CHECK(supported(mixing_dtype), "mixing_dtype must be FP32, FP16 or BF16");
  for (const auto& t : {a, x, w, z, v}) {
    TORCH_CHECK(t.device() == x.device() && t.layout() == at::kStrided && supported(t.scalar_type()),
                "inputs must be strided FP32/FP16/BF16 tensors on one CUDA device");
  }
  const int64_t b = x.size(0), d = x.size(2);
  TORCH_CHECK(a.sizes() == at::IntArrayRef({b, 4, 4}), "a must be [B,4,4]");
  TORCH_CHECK(w.sizes() == at::IntArrayRef({b, 4}), "write_weights must be [B,4]");
  TORCH_CHECK(z.sizes() == at::IntArrayRef({b, d}), "writeback must be [B,D]");
  TORCH_CHECK(v.dim() == 2 && v.size(1) == 4 && (v.size(0) == 1 || v.size(0) == b), "v must be [1,4] or [B,4]");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  if (b == 0 || d == 0) return out;
  TORCH_CHECK(b <= 2147483647LL, "batch exceeds CUDA grid limit");
  auto aa = a.resolve_neg().to(at::kFloat).contiguous();
  auto xx = x.resolve_neg().contiguous();
  auto ww = w.resolve_neg().to(at::kFloat).contiguous();
  auto zz = z.resolve_neg().to(at::kFloat).contiguous();
  auto vv = v.resolve_neg().to(at::kFloat).contiguous();
  const int dtype = mixing_dtype == at::kFloat ? 0 : mixing_dtype == at::kHalf ? 1 : 2;
  const int64_t v_stride = v.size(0) == 1 ? 0 : 4;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "expm_stream_mix_project", [&] {
    // The measured large-B BF16/D768 workload benefits from two tokens per CTA.
    // Keep the general layout for other feature widths, dtypes and small batches.
    if (x.scalar_type() == at::kBFloat16 && b >= 8192 && d == 768)
      launch<scalar_t, 64>(aa, xx, ww, zz, vv, out, b, d, v_stride, dtype, decay);
    else
      launch<scalar_t, 128>(aa, xx, ww, zz, vv, out, b, d, v_stride, dtype, decay);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
} // namespace flash_hypercons
