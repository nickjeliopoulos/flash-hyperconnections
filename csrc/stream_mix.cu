// V-projection stream mixing. FP32 arithmetic, including low-precision inputs.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <tuple>
#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cutlass/gemm/device/gemm_batched.h>

namespace flash_hypercons {
namespace {
constexpr int threads = 128;

__device__ float load_source(const void* y, int dtype, int64_t index) {
  if (dtype == 0) return static_cast<const float*>(y)[index];
  if (dtype == 1) return float(static_cast<const at::Half*>(y)[index]);
  return float(static_cast<const at::BFloat16*>(y)[index]);
}

template <typename T, int V>
struct alignas(sizeof(T) * V) StreamVector { T values[V]; };

template <int V>
__device__ void load_source_vector(const void* y, int dtype, int64_t index, float (&values)[V]) {
  if (dtype == 0) {
    auto pack = *reinterpret_cast<const StreamVector<float, V>*>(static_cast<const float*>(y) + index);
    #pragma unroll
    for (int k = 0; k < V; ++k) values[k] = pack.values[k];
  } else if (dtype == 1) {
    auto pack = *reinterpret_cast<const StreamVector<at::Half, V>*>(static_cast<const at::Half*>(y) + index);
    #pragma unroll
    for (int k = 0; k < V; ++k) values[k] = float(pack.values[k]);
  } else {
    auto pack = *reinterpret_cast<const StreamVector<at::BFloat16, V>*>(static_cast<const at::BFloat16*>(y) + index);
    #pragma unroll
    for (int k = 0; k < V; ++k) values[k] = float(pack.values[k]);
  }
}

// Four adjacent features per thread, with aligned vector loads/stores. This
// reduces CTA count and memory instructions for small stream counts.
template <typename T, int N, int V>
__global__ __launch_bounds__(128, 4) void project_forward_vector(const float* phi,
    const T* x, const void* y, const float* v, T* out, int64_t d,
    int64_t tiles, int64_t tasks, int y_dtype, float decay) {
  __shared__ float p[N * (N + 1)], direction[N], c[N];
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int64_t b = task / tiles;
    for (int k = threadIdx.x; k < N * N; k += blockDim.x)
      p[(k / N) * (N + 1) + k % N] = phi[b * N * N + k];
    if (threadIdx.x < N) direction[threadIdx.x] = v[b * N + threadIdx.x];
    __syncthreads();
    if (threadIdx.x < N) {
      float pv = 0;
      #pragma unroll
      for (int j = 0; j < N; ++j) pv = fmaf(p[threadIdx.x * (N + 1) + j], direction[j], pv);
      c[threadIdx.x] = decay * direction[threadIdx.x] - pv;
    }
    __syncthreads();
    const int64_t col = ((task % tiles) * blockDim.x + threadIdx.x) * V;
    if (col < d) {
      float values[N][V], alpha[V] = {};
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        auto pack = *reinterpret_cast<const StreamVector<T, V>*>(x + (b * N + j) * d + col);
        #pragma unroll
        for (int k = 0; k < V; ++k) {
          values[j][k] = float(pack.values[k]);
          alpha[k] = fmaf(direction[j], values[j][k], alpha[k]);
        }
      }
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        float acc[V] = {}, source[V];
        #pragma unroll
        for (int j = 0; j < N; ++j) {
          const float coefficient = p[i * (N + 1) + j];
          #pragma unroll
          for (int k = 0; k < V; ++k) acc[k] = fmaf(coefficient, values[j][k], acc[k]);
        }
        const int64_t index = (b * N + i) * d + col;
        load_source_vector<V>(y, y_dtype, index, source);
        StreamVector<T, V> result;
        #pragma unroll
        for (int k = 0; k < V; ++k) result.values[k] = T(fmaf(c[i], alpha[k], acc[k]) + source[k]);
        *reinterpret_cast<StreamVector<T, V>*>(out + index) = result;
      }
    }
    __syncthreads();
  }
}

// Candidate: reuse the complete x column across all output stream rows.
// Phi/v are shared once per block; alpha is computed once per D lane.
template <typename T, int N>
__global__ void project_forward_all_rows(const float* phi, const T* x, const void* y,
    const float* v, T* out, int64_t d, int64_t tiles, int64_t tasks, int y_dtype, float decay) {
  __shared__ float p[N * (N + 1)];
  __shared__ float direction[N];
  __shared__ float c[N];
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int64_t b = task / tiles;
    for (int k = threadIdx.x; k < N * N; k += blockDim.x) p[(k / N) * (N + 1) + k % N] = phi[b * N * N + k];
    if (threadIdx.x < N) direction[threadIdx.x] = v[b * N + threadIdx.x];
    __syncthreads();
    if (threadIdx.x < N) {
      float pv = 0;
      #pragma unroll
      for (int j = 0; j < N; ++j) pv = fmaf(p[threadIdx.x * (N + 1) + j], direction[j], pv);
      c[threadIdx.x] = decay * direction[threadIdx.x] - pv;
    }
    __syncthreads();
    const int64_t col = (task % tiles) * blockDim.x + threadIdx.x;
    if (col < d) {
      float values[N];
      float alpha = 0;
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        values[j] = float(x[(b * N + j) * d + col]);
        alpha = fmaf(direction[j], values[j], alpha);
      }
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        float acc = 0;
        #pragma unroll
        for (int j = 0; j < N; ++j) acc = fmaf(p[i * (N + 1) + j], values[j], acc);
        const int64_t index = (b * N + i) * d + col;
        out[index] = T(fmaf(c[i], alpha, acc) + load_source(y, y_dtype, index));
      }
    }
    __syncthreads();
  }
}

// Grid-striding keeps large batches independent of CUDA's grid-y limit.
template <typename T, int N>
__global__ void project_forward(const float* phi, const T* x, const void* y,
    const float* v, T* out, int n, int64_t d, int64_t tiles, int64_t tasks, int y_dtype, float decay) {
  const int count = N ? N : n;
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int64_t row = task / tiles;
    const int64_t b = row / count;
    const int i = row % count;
    const int64_t col = (task % tiles) * blockDim.x + threadIdx.x;
    if (col < d) {
      float acc = 0, alpha = 0, phi_v = 0;
      #pragma unroll
      for (int j = 0; j < count; ++j) {
        const float p = phi[(b * count + i) * count + j];
        const float value = float(x[(b * count + j) * d + col]);
        const float direction = v[b * count + j];
        acc = fmaf(p, value, acc);
        alpha = fmaf(direction, value, alpha);
        phi_v = fmaf(p, direction, phi_v);
      }
      acc = fmaf(decay * v[b * count + i] - phi_v, alpha, acc);
      out[row * d + col] = T(acc + load_source(y, y_dtype, row * d + col));
    }
  }
}

template <typename T>
__global__ void project_moments(const float* phi, const T* x, const T* g,
    const float* v, float* alpha, float* beta, int n, int64_t d,
    int64_t tiles, int64_t tasks, float decay) {
  __shared__ float c[32];
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int64_t b = task / tiles;
    if (threadIdx.x < n) {
      const int i = threadIdx.x;
      float pv = 0;
      for (int j = 0; j < n; ++j)
        pv = fmaf(phi[(b * n + i) * n + j], v[b * n + j], pv);
      c[i] = decay * v[b * n + i] - pv;
    }
    __syncthreads();
    const int64_t col = (task % tiles) * blockDim.x + threadIdx.x;
    if (col < d) {
      float a = 0, be = 0;
      for (int j = 0; j < n; ++j) {
        a = fmaf(v[b * n + j], float(x[(b * n + j) * d + col]), a);
        be = fmaf(c[j], float(g[(b * n + j) * d + col]), be);
      }
      alpha[b * d + col] = a;
      beta[b * d + col] = be;
    }
    __syncthreads();
  }
}

// Reuse each gradient column for the projection moments and every dx row.
template <typename T, int N, bool WriteMoments = true>
__global__ void project_moments_dx(const float* phi, const T* x, const T* g,
    const float* v, float* alpha, float* beta, T* dx, int64_t d,
    int64_t tiles, int64_t tasks, float decay) {
  __shared__ float p[N * (N + 1)];
  __shared__ float direction[N];
  __shared__ float c[N];
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int64_t b = task / tiles;
    for (int k = threadIdx.x; k < N * N; k += blockDim.x) p[(k % N) * (N + 1) + k / N] = phi[b * N * N + k];
    if (threadIdx.x < N) direction[threadIdx.x] = v[b * N + threadIdx.x];
    __syncthreads();
    if (threadIdx.x < N) {
      float pv = 0;
      #pragma unroll
      for (int j = 0; j < N; ++j) pv = fmaf(p[j * (N + 1) + threadIdx.x], direction[j], pv);
      c[threadIdx.x] = decay * direction[threadIdx.x] - pv;
    }
    __syncthreads();
    const int64_t col = (task % tiles) * blockDim.x + threadIdx.x;
    if (col < d) {
      float grads[N];
      float a = 0, be = 0;
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        const int64_t index = (b * N + i) * d + col;
        grads[i] = float(g[index]);
        if constexpr (WriteMoments) a = fmaf(direction[i], float(x[index]), a);
        be = fmaf(c[i], grads[i], be);
      }
      if constexpr (WriteMoments) {
        alpha[b * d + col] = a;
        beta[b * d + col] = be;
      }
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < N; ++i) acc = fmaf(p[j * (N + 1) + i], grads[i], acc);
        dx[(b * N + j) * d + col] = T(fmaf(direction[j], be, acc));
      }
    }
    __syncthreads();
  }
}

// Both operands are stored low-precision gradients/state. Tensor-core products
// therefore introduce no additional input rounding; accumulation stays FP32.
template <typename T, int N>
__global__ void project_cross_moment_tensor(const T* x, const T* g, float* h,
    int64_t d, int64_t batch) {
  for (int64_t b = blockIdx.x; b < batch; b += gridDim.x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    using MmaT = std::conditional_t<std::is_same_v<T, at::BFloat16>, __nv_bfloat16, __half>;
    constexpr int tiles = N / 16;
    const int warp = threadIdx.x / 32, i = (warp / tiles) * 16, j = (warp % tiles) * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, MmaT, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, MmaT, wmma::col_major> bt;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t k = 0; k < d; k += 16) {
      wmma::load_matrix_sync(a, reinterpret_cast<const MmaT*>(g + (b * N + i) * d + k), unsigned(d));
      wmma::load_matrix_sync(bt, reinterpret_cast<const MmaT*>(x + (b * N + j) * d + k), unsigned(d));
      wmma::mma_sync(acc, a, bt, acc);
    }
    wmma::store_matrix_sync(h + (b * N + i) * N + j, acc, N, wmma::mem_row_major);
#else
    // Correct fallback when a pre-SM80 PTX image is JIT-loaded on newer hardware.
    for (int index = threadIdx.x; index < N * N; index += blockDim.x) {
      const int i = index / N, j = index % N;
      float value = 0;
      for (int64_t k = 0; k < d; ++k)
        value = fmaf(float(g[(b * N + i) * d + k]), float(x[(b * N + j) * d + k]), value);
      h[b * N * N + index] = value;
    }
#endif
  }
}

// H = G X^T. A warp owns one output row and retains all its columns.
// Each X tile is loaded once for up to eight output rows.
template <typename T, int N>
__global__ void project_cross_moment(const T* x, const T* g, float* h,
    int64_t d, int64_t tasks) {
  constexpr int row_group = N < 8 ? N : 8;
  constexpr int groups = N / row_group;
  __shared__ float state[N * 256];
  const int lane = threadIdx.x & 31;
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int64_t b = task / groups;
    const int i = (task % groups) * row_group + threadIdx.x / 32;
    float sums[N] = {};
    for (int64_t base = 0; base < d; base += 256) {
      for (int k = threadIdx.x; k < N * 256; k += blockDim.x) {
        const int64_t col = base + k % 256;
        state[k] = col < d ? float(x[(b * N + k / 256) * d + col]) : 0.0f;
      }
      __syncthreads();
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int64_t col = base + k * 32 + lane;
        const float grad = col < d ? float(g[(b * N + i) * d + col]) : 0.0f;
        #pragma unroll
        for (int j = 0; j < N; ++j)
          sums[j] = fmaf(grad, state[j * 256 + k * 32 + lane], sums[j]);
      }
      __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float value = sums[j];
      for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if (lane == 0) h[(b * N + i) * N + j] = value;
    }
  }
}

// rho=H*v; dPhi=H-rho*v^T; dv=decay*rho-Phi^T*rho+H^T*c.
// Padded rows avoid bank conflicts while lanes reduce different matrix rows.
template <int N>
__global__ void project_finish_cross_moment(const float* phi, const float* v,
    const float* h, float* dp, float* dv, int64_t batch, float decay) {
  __shared__ float p[N * (N + 1)], cross[N * (N + 1)];
  __shared__ float direction[N], c[N], rho[N];
  for (int64_t b = blockIdx.x; b < batch; b += gridDim.x) {
    for (int k = threadIdx.x; k < N * N; k += blockDim.x) {
      const int at = (k / N) * (N + 1) + k % N;
      p[at] = phi[b * N * N + k];
      cross[at] = h[b * N * N + k];
    }
    if (threadIdx.x < N) direction[threadIdx.x] = v[b * N + threadIdx.x];
    __syncthreads();
    if (threadIdx.x < N) {
      const int i = threadIdx.x;
      float pv = 0, r = 0;
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        pv = fmaf(p[i * (N + 1) + j], direction[j], pv);
        r = fmaf(cross[i * (N + 1) + j], direction[j], r);
      }
      c[i] = decay * direction[i] - pv;
      rho[i] = r;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < N * N; k += blockDim.x) {
      const int i = k / N, j = k % N;
      dp[b * N * N + k] = fmaf(-rho[i], direction[j], cross[i * (N + 1) + j]);
    }
    if (threadIdx.x < N) {
      const int j = threadIdx.x;
      float pr = 0, hc = 0;
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        pr = fmaf(p[i * (N + 1) + j], rho[i], pr);
        hc = fmaf(cross[i * (N + 1) + j], c[i], hc);
      }
      dv[b * N + j] = (decay * rho[j] - pr) + hc;
    }
    __syncthreads();
  }
}

template <typename T, int N>
__global__ void project_dx(const float* phi, const T* g, const float* v,
    const float* beta, T* dx, int n, int64_t d, int64_t tiles, int64_t tasks) {
  const int count = N ? N : n;
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int64_t row = task / tiles, b = row / count;
    const int j = row % count;
    const int64_t col = (task % tiles) * blockDim.x + threadIdx.x;
    if (col < d) {
      float acc = 0;
      #pragma unroll
      for (int i = 0; i < count; ++i)
        acc = fmaf(phi[(b * count + i) * count + j], float(g[(b * count + i) * d + col]), acc);
      dx[row * d + col] = T(fmaf(v[row], beta[b * d + col], acc));
    }
  }
}

__device__ float block_sum(float value, float* shared) {
  for (int offset = 16; offset; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    value = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0f;
    for (int offset = 16; offset; offset /= 2)
      value += __shfl_down_sync(0xffffffff, value, offset);
  }
  __syncthreads();
  return value;
}

// One reduction block per Phi element or v element; no atomics or scratch
// proportional to B*N*N*D. rho/bp retain FP32 until the final dv contraction.
template <typename T>
__global__ void project_parameters(const T* x, const T* g, const float* v,
    const float* alpha, const float* beta, float* dp, float* rho, float* bp,
    int n, int64_t d, int64_t batch, int64_t tasks) {
  __shared__ float shared[32];
  const int64_t phi_tasks = batch * n * n;
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const bool matrix = task < phi_tasks;
    const int64_t index = matrix ? task : task - phi_tasks;
    const int64_t b = index / (matrix ? n * n : n);
    const int i = matrix ? (index / n) % n : index % n;
    const int j = matrix ? index % n : i;
    float sum = 0, second = 0;
    for (int64_t col = threadIdx.x; col < d; col += blockDim.x) {
      const float a = alpha[b * d + col];
      const float value = float(x[(b * n + j) * d + col]);
      const float grad = float(g[(b * n + i) * d + col]);
      if (matrix) sum = fmaf(grad, fmaf(-v[b * n + j], a, value), sum);
      else {
        sum = fmaf(grad, a, sum);
        second = fmaf(beta[b * d + col], value, second);
      }
    }
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) {
      if (matrix) dp[index] = sum;
      else rho[index] = sum;
    }
    if (!matrix) {
      second = block_sum(second, shared);
      if (threadIdx.x == 0) bp[index] = second;
    }
  }
}

// One block per output row, accumulating all Phi columns and the v moments.
// All warp partials are exchanged with one shared-memory synchronization.
template <typename T, int N>
__global__ void project_parameter_rows(const T* x, const T* g, const float* v,
    const float* alpha, const float* beta, float* dp, float* rho, float* bp,
    int64_t d, int64_t tasks) {
  __shared__ float partial[(N + 2) * (threads / 32)];
  for (int64_t row = blockIdx.x; row < tasks; row += gridDim.x) {
    const int64_t b = row / N;
    float sums[N + 2] = {};
    for (int64_t col = threadIdx.x; col < d; col += blockDim.x) {
      const float a = alpha[b * d + col];
      const float grad = float(g[row * d + col]);
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        const float value = float(x[(b * N + j) * d + col]);
        sums[j] = fmaf(grad, fmaf(-v[b * N + j], a, value), sums[j]);
      }
      sums[N] = fmaf(grad, a, sums[N]);
      sums[N + 1] = fmaf(beta[b * d + col], float(x[row * d + col]), sums[N + 1]);
    }
    #pragma unroll
    for (int j = 0; j < N + 2; ++j) {
      float value = sums[j];
      for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if ((threadIdx.x & 31) == 0) partial[j * (threads / 32) + threadIdx.x / 32] = value;
    }
    __syncthreads();
    if (threadIdx.x < N + 2) {
      const int j = threadIdx.x;
      float value = 0;
      #pragma unroll
      for (int warp = 0; warp < threads / 32; ++warp) value += partial[j * (threads / 32) + warp];
      if (j < N) dp[row * N + j] = value;
      else if (j == N) rho[row] = value;
      else bp[row] = value;
    }
    __syncthreads();
  }
}

__global__ void project_dv(const float* phi, const float* rho, const float* bp,
    float* dv, int n, int64_t rows, float decay) {
  for (int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows; row += int64_t(blockDim.x) * gridDim.x) {
    const int64_t b = row / n;
    const int j = row % n;
    float pr = 0;
    for (int i = 0; i < n; ++i) pr = fmaf(phi[(b * n + i) * n + j], rho[b * n + i], pr);
    dv[row] = (decay * rho[row] - pr) + bp[row];
  }
}

bool supported(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

void check_inputs(const at::Tensor& phi, const at::Tensor& x,
    const at::Tensor& y, const at::Tensor& v) {
  TORCH_CHECK(x.is_cuda() && x.layout() == at::kStrided && x.dim() == 3 && supported(x.scalar_type()),
      "x must be a strided CUDA [B,N,D] tensor in float32, float16, or bfloat16");
  TORCH_CHECK(x.size(1) >= 1 && x.size(1) <= 32, "CUDA stream mixing requires 1 <= N <= 32");
  for (const auto& t : {phi, y, v})
    TORCH_CHECK(t.device() == x.device() && t.layout() == at::kStrided && supported(t.scalar_type()),
        "all inputs must be strided tensors on the same CUDA device in float32, float16, or bfloat16");
  TORCH_CHECK(phi.dim() == 3 && phi.size(0) == x.size(0) && phi.size(1) == x.size(1) && phi.size(2) == x.size(1),
      "Phi must have shape [B,N,N]");
  TORCH_CHECK(y.sizes() == x.sizes(), "Y must match x shape");
  TORCH_CHECK(v.dim() == 2 && v.size(0) == x.size(0) && v.size(1) == x.size(1), "v must have shape [B,N]");
}

at::Tensor prepare(const at::Tensor& t) { return t.contiguous().resolve_neg(); }
unsigned blocks(int64_t tasks) { return unsigned(std::min<int64_t>(tasks, 65535)); }

// Small batched SIMT GEMM: exact FP32 inputs/products, without TF32 conversion.
template <int N>
bool launch_cross_moment_simt(const at::Tensor& x, const at::Tensor& g,
    const at::Tensor& h, cudaStream_t stream) {
  const int64_t b = x.size(0), d = x.size(2);
  if (b > 65535 || d > 2147483647) return false;
  using Gemm = cutlass::gemm::device::GemmBatched<
      float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor,
      float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,
      cutlass::arch::Sm50, cutlass::gemm::GemmShape<N, 32, 8>,
      cutlass::gemm::GemmShape<16, (N == 16 ? 32 : 16), 8>, cutlass::gemm::GemmShape<1, 1, 1>,
      cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2, 1, 1>;
  typename Gemm::Arguments args({N, N, int(d)}, {g.const_data_ptr<float>(), int(d)}, N * d,
      {x.const_data_ptr<float>(), int(d)}, N * d, {h.const_data_ptr<float>(), N}, N * N,
      {h.mutable_data_ptr<float>(), N}, N * N, {1.0f, 0.0f}, int(b));
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  TORCH_CHECK(gemm(args, nullptr, stream) == cutlass::Status::kSuccess,
      "stream mixing FP32 parameter contraction failed");
  return true;
}

template <typename T, int N>
void launch_cross_moment_only(const at::Tensor& x, const at::Tensor& g,
    const at::Tensor& h, bool tensor_eligible, cudaStream_t stream) {
  const int64_t b = x.size(0), d = x.size(2);
  constexpr int row_group = N < 8 ? N : 8;
  const int64_t tasks = b * (N / row_group);
  bool contracted = false;
  if constexpr (N == 16 || N == 32) {
    if constexpr (std::is_same_v<T, float>) {
      contracted = launch_cross_moment_simt<N>(x, g, h, stream);
    } else if constexpr (std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16>) {
      if (tensor_eligible) {
        project_cross_moment_tensor<T, N><<<blocks(b), (N / 16) * (N / 16) * 32, 0, stream>>>(
            x.const_data_ptr<T>(), g.const_data_ptr<T>(), h.mutable_data_ptr<float>(), d, b);
        contracted = true;
      }
    }
  }
  if (!contracted)
    project_cross_moment<T, N><<<blocks(tasks), row_group * 32, 0, stream>>>(
        x.const_data_ptr<T>(), g.const_data_ptr<T>(), h.mutable_data_ptr<float>(), d, tasks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int N>
void launch_cross_backward(const at::Tensor& p, const at::Tensor& x,
    const at::Tensor& g, const at::Tensor& v, const at::Tensor& h,
    const at::Tensor& dp, const at::Tensor& dx, const at::Tensor& dv, bool tensor_eligible, float decay) {
  const int64_t b = x.size(0), d = x.size(2), tiles = (d + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  project_moments_dx<T, N, false><<<blocks(b * tiles), threads, 0, stream>>>(
      p.const_data_ptr<float>(), nullptr, g.const_data_ptr<T>(), v.const_data_ptr<float>(),
      nullptr, nullptr, dx.mutable_data_ptr<T>(), d, tiles, b * tiles, decay);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_cross_moment_only<T, N>(x, g, h, tensor_eligible, stream);
  project_finish_cross_moment<N><<<blocks(b), threads, 0, stream>>>(p.const_data_ptr<float>(),
      v.const_data_ptr<float>(), h.const_data_ptr<float>(), dp.mutable_data_ptr<float>(),
      dv.mutable_data_ptr<float>(), b, decay);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int N>
void launch_rows(const at::Tensor& phi, const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& v, const at::Tensor& out, const at::Tensor& beta, bool backward, float decay) {
  const int n = x.size(1);
  const int64_t d = x.size(2), tiles = (d + threads - 1) / threads, tasks = x.size(0) * n * tiles;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if constexpr (N == 4 || N == 8) {
    if (!backward && x.size(0) >= 32 && d >= 256 && d % 4 == 0 &&
        reinterpret_cast<uintptr_t>(x.const_data_ptr()) % (4 * sizeof(T)) == 0 &&
        reinterpret_cast<uintptr_t>(y.const_data_ptr()) % (4 * y.element_size()) == 0) {
      const int64_t vector_tiles = (d + threads * 4 - 1) / (threads * 4);
      const int64_t vector_tasks = x.size(0) * vector_tiles;
      project_forward_vector<T, N, 4><<<blocks(vector_tasks), threads, 0, stream>>>(
          phi.const_data_ptr<float>(), x.const_data_ptr<T>(), y.const_data_ptr(),
          v.const_data_ptr<float>(), out.mutable_data_ptr<T>(), d, vector_tiles, vector_tasks,
          y.scalar_type() == at::kFloat ? 0 : y.scalar_type() == at::kHalf ? 1 : 2, decay);
      return;
    }
  }
  if constexpr (N >= 2) {
    if (!backward && x.size(0) >= 32) {
      const int64_t batch_tasks = x.size(0) * tiles;
      project_forward_all_rows<T, N><<<blocks(batch_tasks), threads, 0, stream>>>(
          phi.const_data_ptr<float>(), x.const_data_ptr<T>(), y.const_data_ptr(), v.const_data_ptr<float>(),
          out.mutable_data_ptr<T>(), d, tiles, batch_tasks,
          y.scalar_type() == at::kFloat ? 0 : y.scalar_type() == at::kHalf ? 1 : 2, decay);
      return;
    }
  }
  if (backward)
    project_dx<T, N><<<blocks(tasks), threads, 0, stream>>>(phi.const_data_ptr<float>(), x.const_data_ptr<T>(),
        v.const_data_ptr<float>(), beta.const_data_ptr<float>(), out.mutable_data_ptr<T>(), n, d, tiles, tasks);
  else
    project_forward<T, N><<<blocks(tasks), threads, 0, stream>>>(phi.const_data_ptr<float>(), x.const_data_ptr<T>(),
        y.const_data_ptr(), v.const_data_ptr<float>(), out.mutable_data_ptr<T>(), n, d, tiles, tasks,
        y.scalar_type() == at::kFloat ? 0 : y.scalar_type() == at::kHalf ? 1 : 2, decay);
}

void rows(const at::Tensor& phi, const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& v, const at::Tensor& out, const at::Tensor& beta, bool backward, float decay) {
#define ROWS(N) launch_rows<scalar_t, N>(phi, x, y, v, out, beta, backward, decay)
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_mix_rows", [&] {
    switch (x.size(1)) {
      case 1: ROWS(1); break;
      case 2: ROWS(2); break;
      case 4: ROWS(4); break;
      case 8: ROWS(8); break;
      case 16: ROWS(16); break;
      case 32: ROWS(32); break;
      default: ROWS(0); break;
    }
  });
#undef ROWS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

// Internal contraction shared by standalone and exp/mix backward. Inputs are
// validated and prepared by the callers; no public operator registration.
at::Tensor stream_mix_cross_moment_cuda(const at::Tensor& x, const at::Tensor& g) {
  const int64_t b = x.size(0), d = x.size(2);
  const int n = x.size(1);
  auto h = at::empty({b, n, n}, x.options().dtype(at::kFloat));
  const bool tensor_eligible = x.scalar_type() != at::kFloat && d % 16 == 0 && d <= 2147483647 &&
      uintptr_t(x.data_ptr()) % 32 == 0 && uintptr_t(g.data_ptr()) % 32 == 0 &&
      at::cuda::getCurrentDeviceProperties()->major >= 8;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "expm_stream_cross_moment", [&] {
    switch (n) {
      case 4: launch_cross_moment_only<scalar_t, 4>(x, g, h, tensor_eligible, stream); break;
      case 8: launch_cross_moment_only<scalar_t, 8>(x, g, h, tensor_eligible, stream); break;
      case 16: launch_cross_moment_only<scalar_t, 16>(x, g, h, tensor_eligible, stream); break;
      case 32: launch_cross_moment_only<scalar_t, 32>(x, g, h, tensor_eligible, stream); break;
      default: TORCH_CHECK(false, "cross moment requires N=4/8/16/32");
    }
  });
  return h;
}

at::Tensor stream_mix_project_cuda(const at::Tensor& phi, const at::Tensor& x,
    const at::Tensor& y, const at::Tensor& v, double decay) {
  TORCH_CHECK(decay >= 0 && decay <= 1, "projection_decay must lie in [0, 1]");
  check_inputs(phi, x, y, v);
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  if (x.numel() == 0) return out;
  rows(prepare(phi.to(at::kFloat)), prepare(x), prepare(y), prepare(v.to(at::kFloat)), out, at::Tensor(), false, float(decay));
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> stream_mix_project_backward_cuda(
    const at::Tensor& phi, const at::Tensor& x, const at::Tensor& v, const at::Tensor& grad, double decay) {
  TORCH_CHECK(decay >= 0 && decay <= 1, "projection_decay must lie in [0, 1]");
  TORCH_CHECK(grad.scalar_type() == x.scalar_type(), "gradient must match x dtype");
  check_inputs(phi, x, grad, v);
  const c10::cuda::CUDAGuard guard(x.device());
  auto dx = at::empty(x.sizes(), x.options());
  if (x.numel() == 0) return {at::zeros_like(phi), dx, at::zeros_like(v)};
  const auto p = prepare(phi.to(at::kFloat)), vc = prepare(v.to(at::kFloat));
  const auto xc = prepare(x), g = prepare(grad);
  const int n = x.size(1);
  const int64_t b = x.size(0), d = x.size(2), tiles = (d + threads - 1) / threads;
  const auto opts = x.options().dtype(at::kFloat);
  const bool tensor_eligible = (n == 16 || n == 32) && x.scalar_type() != at::kFloat &&
      d % 16 == 0 && d <= 2147483647 &&
      reinterpret_cast<uintptr_t>(xc.const_data_ptr()) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(g.const_data_ptr()) % 32 == 0 &&
      at::cuda::getCurrentDeviceProperties()->major >= 8;
  if (b >= 32 && (n == 4 || n == 8 || n == 16 || (n == 32 && (b >= 256 || tensor_eligible)))) {
    auto h = at::empty(phi.sizes(), opts), dp = at::empty(phi.sizes(), opts);
    auto dv = at::empty(v.sizes(), opts);
#define CROSS_BACKWARD(N) launch_cross_backward<scalar_t, N>(p, xc, g, vc, h, dp, dx, dv, tensor_eligible, float(decay))
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_mix_cross_backward", [&] {
      switch (n) {
        case 4: CROSS_BACKWARD(4); break;
        case 8: CROSS_BACKWARD(8); break;
        case 16: CROSS_BACKWARD(16); break;
        case 32: CROSS_BACKWARD(32); break;
      }
    });
#undef CROSS_BACKWARD
    return {dp.to(phi.scalar_type()), dx, dv.to(v.scalar_type())};
  }
  auto alpha = at::empty({b, d}, opts), beta = at::empty({b, d}, opts);
  auto dp = at::empty(phi.sizes(), opts), dv = at::empty(v.sizes(), opts);
  auto rho = at::empty(v.sizes(), opts), bp = at::empty(v.sizes(), opts);
  const auto stream = at::cuda::getCurrentCUDAStream();
  bool fused_moments = false;
#define MOMENTS_DX(N) project_moments_dx<scalar_t, N><<<blocks(b * tiles), threads, 0, stream>>>( \
    p.const_data_ptr<float>(), xc.const_data_ptr<scalar_t>(), g.const_data_ptr<scalar_t>(), \
    vc.const_data_ptr<float>(), alpha.mutable_data_ptr<float>(), beta.mutable_data_ptr<float>(), \
    dx.mutable_data_ptr<scalar_t>(), d, tiles, b * tiles, float(decay))
  if (b >= 32) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_mix_moments_dx", [&] {
      switch (n) {
        case 2: MOMENTS_DX(2); fused_moments = true; break;
        case 4: MOMENTS_DX(4); fused_moments = true; break;
        case 8: MOMENTS_DX(8); fused_moments = true; break;
        case 16: MOMENTS_DX(16); fused_moments = true; break;
        case 32: MOMENTS_DX(32); fused_moments = true; break;
      }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
#undef MOMENTS_DX
  if (!fused_moments) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_mix_moments", [&] {
      project_moments<scalar_t><<<blocks(b * tiles), threads, 0, stream>>>(p.const_data_ptr<float>(),
          xc.const_data_ptr<scalar_t>(), g.const_data_ptr<scalar_t>(), vc.const_data_ptr<float>(),
          alpha.mutable_data_ptr<float>(), beta.mutable_data_ptr<float>(), n, d, tiles, b * tiles, float(decay));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    rows(p, g, at::Tensor(), vc, dx, beta, true, float(decay));
  }
  const int64_t tasks = b * n * (n + 1);
#define PARAM_ROWS(N) project_parameter_rows<scalar_t, N><<<blocks(b * N), threads, 0, stream>>>( \
    xc.const_data_ptr<scalar_t>(), g.const_data_ptr<scalar_t>(), vc.const_data_ptr<float>(), \
    alpha.const_data_ptr<float>(), beta.const_data_ptr<float>(), dp.mutable_data_ptr<float>(), \
    rho.mutable_data_ptr<float>(), bp.mutable_data_ptr<float>(), d, b * N)
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_mix_parameters", [&] {
    if (b >= 32) {
      switch (n) {
        case 2: PARAM_ROWS(2); return;
        case 4: PARAM_ROWS(4); return;
        case 8: PARAM_ROWS(8); return;
        case 16: PARAM_ROWS(16); return;
        case 32: PARAM_ROWS(32); return;
      }
    }
    project_parameters<scalar_t><<<blocks(tasks), threads, 0, stream>>>(xc.const_data_ptr<scalar_t>(),
        g.const_data_ptr<scalar_t>(), vc.const_data_ptr<float>(), alpha.const_data_ptr<float>(), beta.const_data_ptr<float>(),
        dp.mutable_data_ptr<float>(), rho.mutable_data_ptr<float>(), bp.mutable_data_ptr<float>(), n, d, b, tasks);
  });
#undef PARAM_ROWS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  project_dv<<<blocks((b * n + threads - 1) / threads), threads, 0, stream>>>(p.const_data_ptr<float>(),
      rho.const_data_ptr<float>(), bp.const_data_ptr<float>(), dv.mutable_data_ptr<float>(), n, b * n, float(decay));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dp.to(phi.scalar_type()), dx, dv.to(v.scalar_type())};
}
} // namespace flash_hypercons
