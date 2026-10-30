// DEPRECATED: retained for reference; excluded from setup.py and operator registration.
// Exp/project/impulse fusion through N=32, with saved Phi for ordinary backward.
#include "MatrixExpSmall.cuh"
#include <ATen/Dispatch.h>
#include <algorithm>
#include <cstdint>
#include <tuple>

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
template <int P>
__device__ __forceinline__ void transition(const float* a, float* p, int dtype, int row, unsigned mask, int n) {
  tiny::Tile<P, false> tile;
  bool bad = false;
  #pragma unroll
  for (int j = 0; j < P; ++j) {
    tile.d[j] = row < n && j < n ? a[row * n + j] : 0.f;
    bad |= !::isfinite(tile.d[j]);
  }
  if (__any_sync(mask, bad)) {
    #pragma unroll
    for (int j = 0; j < P; ++j) p[row * (P + 1) + j] = NAN;
    return;
  }
  double norm = 0.;
  #pragma unroll
  for (int j = 0; j < P; ++j) {
    double col = ::fabs(double(tile.d[j]));
    #pragma unroll
    for (int offset = P / 2; offset; offset /= 2)
      col += __shfl_xor_sync(mask, col, offset, P);
    norm = ::fmax(norm, col);
  }
  const int s = norm > tiny::theta18 ? int(::ceil(::log2(norm / tiny::theta18))) : 0;
  const float scale = ::ldexpf(1.f, -s);
  #pragma unroll
  for (int j = 0; j < P; ++j) tile.d[j] *= scale;
  auto result = tiny::t18(tile, mask, row);
  for (int i = 0; i < s; ++i) result = tiny::product(result, result, mask);
  #pragma unroll
  for (int j = 0; j < P; ++j) p[row * (P + 1) + j] = round_mix(result.d[j], dtype);
}

template <typename T, int Group, int V, bool Save>
__global__ __launch_bounds__(128) void fused_forward(const float* a, const T* x,
    const float* w, const float* z, const float* v, T* out,
    int64_t batch, int64_t d, int64_t v_stride, int dtype, float decay, float* saved) {
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
    transition<4>(a + b * 16, p, dtype, lane, 0xfu, 4);
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
  if constexpr (Save) {
    if (lane < 16) saved[b * 16 + lane] = p[(lane / 4) * 5 + lane % 4];
  }
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

// Reuse the standalone cooperative polynomial, with zero padding loaded locally.
template <int P, int Threads>
__device__ __forceinline__ void cooperative_transition(const float* a, float* p, int dtype, int n) {
  using Ops = small_matrix_exp::scalar::Cooperative<P, Threads, false>;
  __shared__ float operands[2 * P * P];
  __shared__ int shared_s;
  Ops ops{operands};
  typename Ops::Tile tile;
  bool bad = false;
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) {
    const int i = Ops::index(e), row = i / P, col = i % P;
    tile.d[e] = row < n && col < n ? a[row * n + col] : 0.f;
    bad |= !::isfinite(tile.d[e]);
    operands[i] = tile.d[e];
  }
  if (__syncthreads_or(bad)) {
    for (int i = threadIdx.x; i < P * P; i += Threads) p[(i / P) * (P + 1) + i % P] = NAN;
    return;
  }
  if (threadIdx.x < P) {
    double norm = 0.;
    #pragma unroll
    for (int row = 0; row < P; ++row) norm += ::fabs(double(operands[row * P + threadIdx.x]));
    constexpr unsigned mask = 0xffffffffu >> (32 - P);
    #pragma unroll
    for (int offset = P / 2; offset; offset /= 2)
      norm = ::fmax(norm, __shfl_xor_sync(mask, norm, offset, P));
    if (threadIdx.x == 0)
      shared_s = norm > tiny::theta18 ? int(::ceil(::log2(norm / tiny::theta18))) : 0;
  }
  __syncthreads();
  const int s = shared_s;
  const float scale = ::ldexpf(1.f, -s);
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) tile.d[e] *= scale;
  auto result = ops.t18(tile);
  for (int i = 0; i < s; ++i) result = ops.product(result, result);
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) {
    const int i = Ops::index(e);
    p[(i / P) * (P + 1) + i % P] = round_mix(result.d[e], dtype);
  }
}

template <typename T, int P, int Threads, int V, int RowGroups, bool Save>
__global__ __launch_bounds__(Threads) void extended_forward(const float* a, const T* x,
    const float* w, const float* z, const float* v, T* out,
    int n, int64_t d, int64_t v_stride, int dtype, float decay, float* saved) {
  __shared__ float state[P * (P + 4)];
  float* p = state;
  float* direction = p + P * (P + 1);
  float* weights = direction + P;
  float* c = weights + P;
  const int64_t b = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane < P) {
    direction[lane] = lane < n ? v[b * v_stride + lane] : 0.f;
    weights[lane] = lane < n ? round_mix(w[b * n + lane], dtype) : 0.f;
  }
  if constexpr (P <= 8) {
    if (lane < P) transition<P>(a + b * n * n, p, dtype, lane, 0xffffffffu >> (32 - P), n);
  } else {
    cooperative_transition<P, Threads>(a + b * n * n, p, dtype, n);
  }
  __syncthreads();
  if (lane < P) {
    float pv = 0.f;
    #pragma unroll
    for (int j = 0; j < P; ++j) pv = fmaf(p[lane * (P + 1) + j], direction[j], pv);
    c[lane] = decay * direction[lane] - pv;
  }
  __syncthreads();
  if constexpr (Save) {
    for (int i = lane; i < n * n; i += Threads)
      saved[b * n * n + i] = p[(i / n) * (P + 1) + i % n];
  }
  constexpr int Columns = Threads / RowGroups;
  constexpr int Rows = P / RowGroups;
  const int row_base = (lane / Columns) * Rows;
  for (int64_t col = (lane % Columns) * V; col < d; col += Columns * V) {
    float values[P][V] = {}, alpha[V] = {};
    #pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < n) {
        auto pack = *reinterpret_cast<const Vector<T, V>*>(x + (b * n + j) * d + col);
        #pragma unroll
        for (int k = 0; k < V; ++k) {
          values[j][k] = float(pack.values[k]);
          alpha[k] = fmaf(direction[j], values[j][k], alpha[k]);
        }
      }
    }
    auto source = *reinterpret_cast<const Vector<float, V>*>(z + b * d + col);
    #pragma unroll
    for (int k = 0; k < V; ++k) source.values[k] = round_mix(source.values[k], dtype);
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
      const int row = row_base + i;
      if (row < n) {
        float acc[V] = {};
        #pragma unroll
        for (int j = 0; j < P; ++j) {
          const float coefficient = p[row * (P + 1) + j];
          #pragma unroll
          for (int k = 0; k < V; ++k) acc[k] = fmaf(coefficient, values[j][k], acc[k]);
        }
        Vector<T, V> result;
        #pragma unroll
        for (int k = 0; k < V; ++k) {
          const float y = round_mix(__fmul_rn(weights[row], source.values[k]), dtype);
          result.values[k] = T(fmaf(c[row], alpha[k], acc[k]) + y);
        }
        *reinterpret_cast<Vector<T, V>*>(out + (b * n + row) * d + col) = result;
      }
    }
  }
}

template <typename T, int P, bool Save, int Threads = 128, int V = (P <= 8 ? 4 : 2), int RowGroups = 1>
void launch_extended(const at::Tensor& a, const at::Tensor& x, const at::Tensor& w,
    const at::Tensor& z, const at::Tensor& v, at::Tensor& out,
    int64_t b, int n, int64_t d, int64_t v_stride, int dtype, float decay, float* saved) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool vector = d % V == 0 && uintptr_t(x.data_ptr()) % (V * sizeof(T)) == 0
      && uintptr_t(z.data_ptr()) % (V * sizeof(float)) == 0;
  if (vector)
    extended_forward<T, P, Threads, V, RowGroups, Save><<<b, Threads, 0, stream>>>(a.data_ptr<float>(), x.data_ptr<T>(),
        w.data_ptr<float>(), z.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<T>(), n, d, v_stride, dtype, decay, saved);
  else
    extended_forward<T, P, Threads, 1, RowGroups, Save><<<b, Threads, 0, stream>>>(a.data_ptr<float>(), x.data_ptr<T>(),
        w.data_ptr<float>(), z.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<T>(), n, d, v_stride, dtype, decay, saved);
}

bool supported(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

template <typename T, int Group, bool Save>
void launch(const at::Tensor& a, const at::Tensor& x, const at::Tensor& w,
    const at::Tensor& z, const at::Tensor& v, at::Tensor& out,
    int64_t b, int64_t d, int64_t v_stride, int dtype, float decay, float* saved) {
  const int blocks = (b + 128 / Group - 1) / (128 / Group);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool vector = d % 4 == 0 && uintptr_t(x.data_ptr()) % (4 * sizeof(T)) == 0
      && uintptr_t(z.data_ptr()) % 16 == 0;
  if (vector)
    fused_forward<T, Group, 4, Save><<<blocks, 128, 0, stream>>>(a.data_ptr<float>(), x.data_ptr<T>(),
        w.data_ptr<float>(), z.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<T>(), b, d, v_stride, dtype, decay, saved);
  else
    fused_forward<T, Group, 1, Save><<<blocks, 128, 0, stream>>>(a.data_ptr<float>(), x.data_ptr<T>(),
        w.data_ptr<float>(), z.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<T>(), b, d, v_stride, dtype, decay, saved);
}
} // namespace

template <bool Save>
std::tuple<at::Tensor, at::Tensor> expm_stream_mix_project_impl(const at::Tensor& a, const at::Tensor& x,
    const at::Tensor& w, const at::Tensor& z, const at::Tensor& v,
    double decay, at::ScalarType mixing_dtype) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.size(1) >= 2 && x.size(1) <= 32,
              "x must be CUDA [B,N,D] with 2 <= N <= 32");
  TORCH_CHECK(decay >= 0 && decay <= 1, "projection_decay must be in [0,1]");
  TORCH_CHECK(supported(mixing_dtype), "mixing_dtype must be FP32, FP16 or BF16");
  for (const auto& t : {a, x, w, z, v}) {
    TORCH_CHECK(t.device() == x.device() && t.layout() == at::kStrided && supported(t.scalar_type()),
                "inputs must be strided FP32/FP16/BF16 tensors on one CUDA device");
  }
  const int64_t b = x.size(0), d = x.size(2);
  const int n = x.size(1);
  TORCH_CHECK(a.sizes() == at::IntArrayRef({b, n, n}), "a must be [B,N,N]");
  TORCH_CHECK(w.sizes() == at::IntArrayRef({b, n}), "write_weights must be [B,N]");
  TORCH_CHECK(z.sizes() == at::IntArrayRef({b, d}), "layer_output must be [B,D]");
  TORCH_CHECK(v.dim() == 2 && v.size(1) == n && (v.size(0) == 1 || v.size(0) == b), "v must be [1,N] or [B,N]");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  auto phi = Save ? at::empty({b, n, n}, x.options().dtype(at::kFloat)) : at::Tensor();
  float* saved = Save ? phi.data_ptr<float>() : nullptr;
  if (b == 0 || (!Save && d == 0)) return {out, phi};
  TORCH_CHECK(b <= 2147483647LL, "batch exceeds CUDA grid limit");
  auto aa = a.resolve_neg().to(at::kFloat).contiguous();
  auto xx = x.resolve_neg().contiguous();
  auto ww = w.resolve_neg().to(at::kFloat).contiguous();
  auto zz = z.resolve_neg().to(at::kFloat).contiguous();
  auto vv = v.resolve_neg().to(at::kFloat).contiguous();
  const int dtype = mixing_dtype == at::kFloat ? 0 : mixing_dtype == at::kHalf ? 1 : 2;
  const int64_t v_stride = v.size(0) == 1 ? 0 : n;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "expm_stream_mix_project", [&] {
    // The measured large-B BF16/D768 workload benefits from two tokens per CTA.
    // Keep the general layout for other feature widths, dtypes and small batches.
    if (n != 4) {
      if (n <= 4) launch_extended<scalar_t, 4, Save>(aa, xx, ww, zz, vv, out, b, n, d, v_stride, dtype, decay, saved);
      else if (n <= 8) launch_extended<scalar_t, 8, Save>(aa, xx, ww, zz, vv, out, b, n, d, v_stride, dtype, decay, saved);
      else if (n <= 16) launch_extended<scalar_t, 16, Save>(aa, xx, ww, zz, vv, out, b, n, d, v_stride, dtype, decay, saved);
      else launch_extended<scalar_t, 32, Save>(aa, xx, ww, zz, vv, out, b, n, d, v_stride, dtype, decay, saved);
    } else if (x.scalar_type() == at::kBFloat16 && b >= 8192 && d == 768)
      launch<scalar_t, 64, Save>(aa, xx, ww, zz, vv, out, b, d, v_stride, dtype, decay, saved);
    else
      launch<scalar_t, 128, Save>(aa, xx, ww, zz, vv, out, b, d, v_stride, dtype, decay, saved);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, phi};
}

at::Tensor expm_stream_mix_project_cuda(const at::Tensor& a, const at::Tensor& x,
    const at::Tensor& w, const at::Tensor& z, const at::Tensor& v,
    double decay, at::ScalarType mixing_dtype) {
  return std::get<0>(expm_stream_mix_project_impl<false>(a, x, w, z, v, decay, mixing_dtype));
}

std::tuple<at::Tensor, at::Tensor> expm_stream_mix_project_training_cuda(
    const at::Tensor& a, const at::Tensor& x, const at::Tensor& w,
    const at::Tensor& z, const at::Tensor& v, double decay, at::ScalarType mixing_dtype) {
  return expm_stream_mix_project_impl<true>(a, x, w, z, v, decay, mixing_dtype);
}

at::Tensor stream_mix_cross_moment_cuda(const at::Tensor&, const at::Tensor&);
at::Tensor frechet_cuda(const at::Tensor&, const at::Tensor&);
std::tuple<at::Tensor, at::Tensor, at::Tensor> stream_mix_project_backward_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double);

namespace {
int storage_dtype(at::ScalarType type) {
  return type == at::kFloat ? 0 : type == at::kBFloat16 ? 1 : 2;
}

// Autograd rounds g to the source dtype, then each product before its reduction.
// Reductions accumulate in FP32 and round once more to the source dtype.
template <typename T, int P>
__global__ void impulse_backward(const T* grad, const float* w, const float* z,
    void* dw, void* dz, int n, int64_t d, int dtype, int wtype, int ztype) {
  __shared__ float partial[P * 4];
  const int64_t b = blockIdx.x;
  float sums[P] = {};
  for (int64_t col = threadIdx.x; col < d; col += 128) {
    const float source = round_mix(z[b * d + col], dtype);
    float gz = 0.f;
    #pragma unroll
    for (int i = 0; i < P; ++i) {
      if (i < n) {
        const float g = round_mix(float(grad[(b * n + i) * d + col]), dtype);
        sums[i] += round_mix(__fmul_rn(g, source), dtype);
        gz += round_mix(__fmul_rn(g, round_mix(w[b * n + i], dtype)), dtype);
      }
    }
    tiny::store_value(dz, ztype, b * d + col, round_mix(gz, dtype));
  }
  #pragma unroll
  for (int i = 0; i < P; ++i) {
    float value = sums[i];
    for (int offset = 16; offset; offset /= 2) value += __shfl_down_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & 31) == 0) partial[i * 4 + threadIdx.x / 32] = value;
  }
  __syncthreads();
  if (threadIdx.x < n) {
    float value = 0.f;
    for (int warp = 0; warp < 4; ++warp) value += partial[threadIdx.x * 4 + warp];
    tiny::store_value(dw, wtype, b * n + threadIdx.x, round_mix(value, dtype));
  }
}
// Reduce dPhi on chip and evaluate the paired exponential at (A^T, dPhi).
// Small matrices retain the packed subwarp ownership of standalone Frechet.
template <int N>
__global__ void finish_frechet_tiny(const float* a, const float* phi, const float* h,
    const float* v, void* da, void* dv, int64_t batch, int64_t v_stride,
    int dtype, int atype, int vtype, float decay) {
  const int row = threadIdx.x % N;
  const int group_start = (threadIdx.x & 31) - row;
  const unsigned mask = (0xffffffffu >> (32 - N)) << group_start;
  const int64_t b = (int64_t(blockIdx.x) * 128 + threadIdx.x) / N;
  if (b >= batch) return;
  const float direction = v[b * v_stride + row];
  float rho = 0.f, pv = 0.f;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    const float vj = __shfl_sync(mask, direction, j, N);
    rho = fmaf(h[(b * N + row) * N + j], vj, rho);
    pv = fmaf(phi[(b * N + row) * N + j], vj, pv);
  }
  const float c = decay * direction - pv;
  float pr = 0.f, hc = 0.f;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    pr = fmaf(phi[(b * N + i) * N + row], __shfl_sync(mask, rho, i, N), pr);
    hc = fmaf(h[(b * N + i) * N + row], __shfl_sync(mask, c, i, N), hc);
  }
  tiny::store_value(dv, vtype, b * N + row, (decay * rho - pr) + hc);
  tiny::Tile<N, true> tile;
  bool bad = false;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    tile.d[j] = a[(b * N + j) * N + row];
    tile.u[j] = round_mix(fmaf(-rho, __shfl_sync(mask, direction, j, N), h[(b * N + row) * N + j]), dtype);
    bad |= !::isfinite(tile.d[j]) || !::isfinite(tile.u[j]);
  }
  if (__any_sync(mask, bad)) {
    #pragma unroll
    for (int j = 0; j < N; ++j) tiny::store_value(da, atype, (b * N + row) * N + j, NAN);
    return;
  }
  double norm = 0.;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    double col = ::fabs(double(tile.d[j])) + ::fabs(double(tile.u[j]));
    #pragma unroll
    for (int offset = N / 2; offset; offset /= 2) col += __shfl_xor_sync(mask, col, offset, N);
    norm = ::fmax(norm, col);
  }
  const int s = norm > tiny::theta18 ? int(::ceil(::log2(norm / tiny::theta18))) : 0;
  const float scale = ::ldexpf(1.f, -s);
  #pragma unroll
  for (int j = 0; j < N; ++j) { tile.d[j] *= scale; tile.u[j] *= scale; }
  auto result = tiny::t18(tile, mask, row);
  for (int i = 0; i < s; ++i) result = tiny::product(result, result, mask);
  #pragma unroll
  for (int j = 0; j < N; ++j) tiny::store_value(da, atype, (b * N + row) * N + j, result.u[j]);
}

template <int N>
__global__ __launch_bounds__(128) void finish_frechet_large(const float* a, const float* phi,
    const float* h, const float* v, void* da, void* dv, int64_t v_stride,
    int dtype, int atype, int vtype, float decay) {
  using Ops = small_matrix_exp::scalar::Cooperative<N, 128, true>;
  __shared__ float direction[N], rho[N], c[N], operands[4 * N * N];
  __shared__ int shared_s;
  const int64_t b = blockIdx.x;
  if (threadIdx.x < N) direction[threadIdx.x] = v[b * v_stride + threadIdx.x];
  __syncthreads();
  if (threadIdx.x < N) {
    const int i = threadIdx.x;
    float r = 0.f, pv = 0.f;
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      r = fmaf(h[(b * N + i) * N + j], direction[j], r);
      pv = fmaf(phi[(b * N + i) * N + j], direction[j], pv);
    }
    rho[i] = r; c[i] = decay * direction[i] - pv;
  }
  __syncthreads();
  if (threadIdx.x < N) {
    const int j = threadIdx.x;
    float pr = 0.f, hc = 0.f;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      pr = fmaf(phi[(b * N + i) * N + j], rho[i], pr);
      hc = fmaf(h[(b * N + i) * N + j], c[i], hc);
    }
    tiny::store_value(dv, vtype, b * N + j, (decay * rho[j] - pr) + hc);
  }
  Ops ops{operands};
  typename Ops::Tile tile;
  bool bad = false;
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) {
    const int at = Ops::index(e), i = at / N, j = at % N;
    tile.d[e] = a[(b * N + j) * N + i];
    tile.u[e] = round_mix(fmaf(-rho[i], direction[j], h[b * N * N + at]), dtype);
    bad |= !::isfinite(tile.d[e]) || !::isfinite(tile.u[e]);
    operands[at] = tile.d[e]; operands[N * N + at] = tile.u[e];
  }
  if (__syncthreads_or(bad)) {
    #pragma unroll
    for (int e = 0; e < Ops::Elements; ++e) tiny::store_value(da, atype, b * N * N + Ops::index(e), NAN);
    return;
  }
  if (threadIdx.x < N) {
    double norm = 0.;
    #pragma unroll
    for (int row = 0; row < N; ++row)
      norm += ::fabs(double(operands[row * N + threadIdx.x])) + ::fabs(double(operands[N * N + row * N + threadIdx.x]));
    constexpr unsigned mask = 0xffffffffu >> (32 - N);
    #pragma unroll
    for (int offset = N / 2; offset; offset /= 2) norm = ::fmax(norm, __shfl_xor_sync(mask, norm, offset, N));
    if (threadIdx.x == 0) shared_s = norm > tiny::theta18 ? int(::ceil(::log2(norm / tiny::theta18))) : 0;
  }
  __syncthreads();
  const int s = shared_s;
  const float scale = ::ldexpf(1.f, -s);
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) { tile.d[e] *= scale; tile.u[e] *= scale; }
  auto result = ops.t18(tile);
  for (int i = 0; i < s; ++i) result = ops.product(result, result);
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) tiny::store_value(da, atype, b * N * N + Ops::index(e), result.u[e]);
}

template <typename T, int N, int V>
__global__ __launch_bounds__(128) void state_source_backward(const T* grad, const float* phi,
    const float* v, const float* w, const float* z, T* dx, void* dw, void* dz,
    int64_t d, int64_t v_stride, int dtype, int wtype, int ztype, float decay) {
  __shared__ float p[N * (N + 1)], direction[N], weights[N], c[N], partial[N * 4];
  const int64_t b = blockIdx.x;
  for (int at = threadIdx.x; at < N * N; at += 128) p[(at / N) * (N + 1) + at % N] = phi[b * N * N + at];
  if (threadIdx.x < N) {
    direction[threadIdx.x] = v[b * v_stride + threadIdx.x];
    weights[threadIdx.x] = round_mix(w[b * N + threadIdx.x], dtype);
  }
  __syncthreads();
  if (threadIdx.x < N) {
    float pv = 0.f;
    #pragma unroll
    for (int j = 0; j < N; ++j) pv = fmaf(p[threadIdx.x * (N + 1) + j], direction[j], pv);
    c[threadIdx.x] = decay * direction[threadIdx.x] - pv;
  }
  __syncthreads();
  float sums[N] = {};
  for (int64_t col = threadIdx.x * V; col < d; col += 128 * V) {
    float values[N][V], beta[V] = {}, gz[V] = {};
    auto source = *reinterpret_cast<const Vector<float, V>*>(z + b * d + col);
    #pragma unroll
    for (int k = 0; k < V; ++k) source.values[k] = round_mix(source.values[k], dtype);
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      auto pack = *reinterpret_cast<const Vector<T, V>*>(grad + (b * N + i) * d + col);
      #pragma unroll
      for (int k = 0; k < V; ++k) {
        const float g = float(pack.values[k]);
        values[i][k] = g;
        beta[k] = fmaf(c[i], g, beta[k]);
        const float rounded = round_mix(g, dtype);
        sums[i] += round_mix(__fmul_rn(rounded, source.values[k]), dtype);
        gz[k] += round_mix(__fmul_rn(rounded, weights[i]), dtype);
      }
    }
    #pragma unroll (N < 8 ? N : 1)
    for (int j = 0; j < N; ++j) {
      float acc[V] = {};
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        const float coefficient = p[i * (N + 1) + j];
        #pragma unroll
        for (int k = 0; k < V; ++k) acc[k] = fmaf(coefficient, values[i][k], acc[k]);
      }
      Vector<T, V> result;
      #pragma unroll
      for (int k = 0; k < V; ++k) result.values[k] = T(fmaf(direction[j], beta[k], acc[k]));
      *reinterpret_cast<Vector<T, V>*>(dx + (b * N + j) * d + col) = result;
    }
    #pragma unroll
    for (int k = 0; k < V; ++k) tiny::store_value(dz, ztype, b * d + col + k, round_mix(gz[k], dtype));
  }
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float value = sums[i];
    for (int offset = 16; offset; offset /= 2) value += __shfl_down_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & 31) == 0) partial[i * 4 + threadIdx.x / 32] = value;
  }
  __syncthreads();
  if (threadIdx.x < N) {
    float value = 0.f;
    for (int warp = 0; warp < 4; ++warp) value += partial[threadIdx.x * 4 + warp];
    tiny::store_value(dw, wtype, b * N + threadIdx.x, round_mix(value, dtype));
  }
}

template <typename T, int N>
void launch_state_source(const at::Tensor& g, const at::Tensor& phi, const at::Tensor& v,
    const at::Tensor& w, const at::Tensor& z, at::Tensor& dx, at::Tensor& dw,
    at::Tensor& dz, int64_t b, int64_t d, int64_t v_stride, int dtype, float decay) {
  constexpr int V = N <= 8 ? 4 : 2;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool vector = d % V == 0 && uintptr_t(g.data_ptr()) % (V * sizeof(T)) == 0 && uintptr_t(z.data_ptr()) % (V * sizeof(float)) == 0;
#define STATE_SOURCE(VEC) state_source_backward<T, N, VEC><<<b, 128, 0, stream>>>(g.data_ptr<T>(), phi.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), z.data_ptr<float>(), dx.data_ptr<T>(), dw.data_ptr(), dz.data_ptr(), d, v_stride, dtype, storage_dtype(dw.scalar_type()), storage_dtype(dz.scalar_type()), decay)
  if (vector) { STATE_SOURCE(V); } else { STATE_SOURCE(1); }
#undef STATE_SOURCE
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
expm_stream_mix_project_backward_cuda(const at::Tensor& a, const at::Tensor& x,
    const at::Tensor& w, const at::Tensor& z, const at::Tensor& v,
    const at::Tensor& phi, const at::Tensor& grad, double decay, at::ScalarType mixing_dtype) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.size(1) >= 2 && x.size(1) <= 32,
      "x must be CUDA [B,N,D] with 2 <= N <= 32");
  TORCH_CHECK(decay >= 0 && decay <= 1 && supported(mixing_dtype), "invalid decay or mixing dtype");
  for (const auto& t : {a, x, w, z, v, phi, grad})
    TORCH_CHECK(t.device() == x.device() && t.layout() == at::kStrided && supported(t.scalar_type()),
        "inputs must be strided FP32/FP16/BF16 tensors on one CUDA device");
  const int64_t b = x.size(0), d = x.size(2);
  const int n = x.size(1);
  TORCH_CHECK(a.sizes() == at::IntArrayRef({b, n, n}) && phi.sizes() == a.sizes() && phi.scalar_type() == at::kFloat,
      "a and saved FP32 phi must be [B,N,N]");
  TORCH_CHECK(w.sizes() == at::IntArrayRef({b, n}) && z.sizes() == at::IntArrayRef({b, d}), "invalid impulse shape");
  TORCH_CHECK(v.dim() == 2 && v.size(1) == n && (v.size(0) == 1 || v.size(0) == b), "v must be [1,N] or [B,N]");
  TORCH_CHECK(grad.sizes() == x.sizes() && grad.scalar_type() == x.scalar_type(), "grad must match x shape and dtype");
  const c10::cuda::CUDAGuard guard(x.device());
  if (b == 0 || d == 0) {
    auto da = at::zeros(a.sizes(), a.options());
    // Preserve the composition's nonfinite exponential adjoint even when D=0.
    if (b != 0)
      da = frechet_cuda(a.to(at::kFloat).mT(), da.to(at::kFloat)).to(a.scalar_type());
    return {da, at::zeros(x.sizes(), x.options()), at::zeros(w.sizes(), w.options()),
        at::zeros(z.sizes(), z.options()), at::zeros(v.sizes(), v.options())};
  }
  TORCH_CHECK(b <= 2147483647LL, "batch exceeds CUDA grid limit");
  auto g = grad.resolve_neg().contiguous();
  auto ww = w.resolve_neg().to(at::kFloat).contiguous(), zz = z.resolve_neg().to(at::kFloat).contiguous();
  auto dw = at::empty(w.sizes(), w.options()), dz = at::empty(z.sizes(), z.options());
  const int dtype = mixing_dtype == at::kFloat ? 0 : mixing_dtype == at::kHalf ? 1 : 2;
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Measured fused finish/state layouts win through N=16. N=32 retains the
  // separate stream/Frechet kernels below; combining them increased latency.
  if (b >= 32 && (n == 4 || n == 8 || n == 16)) {
    auto aa = a.resolve_neg().to(at::kFloat).contiguous();
    auto xc = x.resolve_neg().contiguous();
    auto pp = phi.resolve_neg().contiguous();
    auto vv = v.resolve_neg().to(at::kFloat).contiguous();
    auto h = stream_mix_cross_moment_cuda(xc, g);
    auto da = at::empty(a.sizes(), a.options()), dx = at::empty(x.sizes(), x.options());
    auto dv = at::empty({b, n}, v.options());
    const int64_t v_stride = v.size(0) == 1 ? 0 : n;
#define STATE(N) launch_state_source<scalar_t, N>(g, pp, vv, ww, zz, dx, dw, dz, b, d, v_stride, dtype, float(decay))
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "expm_state_source_backward", [&] {
      switch (n) {
        case 4: STATE(4); break;
        case 8: STATE(8); break;
        case 16: STATE(16); break;
      }
    });
#undef STATE
#define FINISH_TINY(N) finish_frechet_tiny<N><<<(b * N + 127) / 128, 128, 0, stream>>>(aa.data_ptr<float>(), pp.data_ptr<float>(), h.data_ptr<float>(), vv.data_ptr<float>(), da.data_ptr(), dv.data_ptr(), b, v_stride, dtype, storage_dtype(a.scalar_type()), storage_dtype(v.scalar_type()), float(decay))
#define FINISH_LARGE(N) finish_frechet_large<N><<<b, 128, 0, stream>>>(aa.data_ptr<float>(), pp.data_ptr<float>(), h.data_ptr<float>(), vv.data_ptr<float>(), da.data_ptr(), dv.data_ptr(), v_stride, dtype, storage_dtype(a.scalar_type()), storage_dtype(v.scalar_type()), float(decay))
    switch (n) {
      case 4: FINISH_TINY(4); break;
      case 8: FINISH_TINY(8); break;
      case 16: FINISH_LARGE(16); break;
    }
#undef FINISH_TINY
#undef FINISH_LARGE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {da, dx, dw, dz, dv.sum_to_size(v.sizes())};
  }
#define IMPULSE(P) impulse_backward<scalar_t, P><<<b, 128, 0, stream>>>(g.data_ptr<scalar_t>(), ww.data_ptr<float>(), zz.data_ptr<float>(), dw.data_ptr(), dz.data_ptr(), n, d, dtype, storage_dtype(w.scalar_type()), storage_dtype(z.scalar_type()))
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "expm_impulse_backward", [&] {
    if (n <= 4) { IMPULSE(4); }
    else if (n <= 8) { IMPULSE(8); }
    else if (n <= 16) { IMPULSE(16); }
    else { IMPULSE(32); }
  });
#undef IMPULSE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto [dp, dx, dv] = stream_mix_project_backward_cuda(phi, x, v.expand({b, n}), g, decay);
  auto da = frechet_cuda(a.to(at::kFloat).mT(), dp.to(mixing_dtype).to(at::kFloat)).to(a.scalar_type());
  return {da, dx, dw, dz, dv.sum_to_size(v.sizes())};
}
} // namespace flash_hypercons
