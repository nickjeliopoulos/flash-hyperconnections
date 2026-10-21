// Adapted from the validated benchmarks/matrix_exp checkpoint.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/container/array.hpp>
#include <cute/tensor.hpp>
#include <cmath>

namespace flash_hypercons::small_matrix_exp::tiny {
constexpr int threads = 128;
constexpr double theta18 = 3.010066362817634;

template <int N, bool Pair>
struct Tile {
  cute::array<float, N> d;
  cute::array<float, Pair ? N : 1> u;
};

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> product(
    const Tile<N, Pair>& a, const Tile<N, Pair>& b, unsigned mask) {
  Tile<N, Pair> c;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    float d = 0.f;
    float u = 0.f;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      const float bd = __shfl_sync(mask, b.d[j], k, N);
      d = ::fmaf(a.d[k], bd, d);
      if constexpr (Pair) {
        const float b_upper = __shfl_sync(mask, b.u[j], k, N);
        // Separate products match the structured Triton algebra.
        u = ::fmaf(a.d[k], b_upper, u);
      }
    }
    c.d[j] = d;
    if constexpr (Pair) {
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < N; ++k) {
        const float bd = __shfl_sync(mask, b.d[j], k, N);
        v = ::fmaf(a.u[k], bd, v);
      }
      c.u[j] = u + v;
    }
  }
  return c;
}

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> add(
    const Tile<N, Pair>& a, const Tile<N, Pair>& b) {
  Tile<N, Pair> c;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    c.d[j] = a.d[j] + b.d[j];
    if constexpr (Pair) c.u[j] = a.u[j] + b.u[j];
  }
  return c;
}

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> combine(
    const Tile<N, Pair>& a, const Tile<N, Pair>& a2,
    const Tile<N, Pair>& a3, const Tile<N, Pair>& a6,
    float c0, float c1, float c2, float c3, float c6, int row) {
  Tile<N, Pair> b;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    b.d[j] = c0 * float(row == j) + c1*a.d[j] + c2*a2.d[j]
             + c3*a3.d[j] + c6*a6.d[j];
    if constexpr (Pair) {
      b.u[j] = c1*a.u[j] + c2*a2.u[j] + c3*a3.u[j] + c6*a6.u[j];
    }
  }
  return b;
}

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> t18(
    const Tile<N, Pair>& a, unsigned mask, int row) {
  auto a2 = product(a, a, mask);
  auto a3 = product(a2, a, mask);
  auto a6 = product(a3, a3, mask);
  auto b1 = combine(a, a2, a3, a6, 0.f,
      -0.10036558103014462f, -0.008029246482411570f,
      -0.000892138498045730f, 0.f, row);
  auto b2 = combine(a, a2, a3, a6, 0.f,
      0.39784974949645076f, 1.36783778460411719f,
      0.49828962252538268f, -0.00063789819459247233f, row);
  auto b3 = combine(a, a2, a3, a6, -10.96763960529620626f,
      1.68015813878906197f, 0.05717798464788655f,
      -0.00698210122488052f, 3.349750170860705e-05f, row);
  auto b4 = combine(a, a2, a3, a6, -0.09043168323908106f,
      -0.06764045190713819f, 0.06759613017740597f,
      0.02955525704293155f, -1.391802575160607e-05f, row);
  auto b5 = combine(a, a2, a3, a6, 0.f, 0.f,
      -0.09233646193671186f, -0.01693649390020817f,
      -1.400867981820361e-05f, row);
  auto a9 = add(product(b1, b5, mask), b4);
  return add(b2, product(add(b3, a9), a9, mask));
}

__device__ __forceinline__ void store_value(void* out, int dtype, int64_t i, float v) {
  if (dtype == 0) static_cast<float*>(out)[i] = v;
  else if (dtype == 1) static_cast<c10::BFloat16*>(out)[i] = c10::BFloat16(v);
  else static_cast<c10::Half*>(out)[i] = c10::Half(v);
}

// Mode 2 specializes the original S=2 baseline; -1 is adaptive; -2 is fixed runtime S.
template <int N, bool Pair, int Mode>
__global__ void expm_kernel(const float* x, const float* g, void* out,
                          int dtype, int64_t batch, int requested_s) {
  const int lane = threadIdx.x & 31;
  const int row = lane % N;
  const int group_start = lane - row;
  const unsigned mask = (0xffffffffu >> (32 - N)) << group_start;
  const int64_t b = (int64_t(blockIdx.x)*threads + threadIdx.x)/N;
  if (b >= batch) return;
  const int64_t base = b*N*N + row*N;
  Tile<N, Pair> a;
  bool bad = false;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    a.d[j] = x[base+j];
    bad |= !::isfinite(a.d[j]);
    if constexpr (Pair) {
      a.u[j] = g[base+j];
      bad |= !::isfinite(a.u[j]);
    }
  }
  if (__any_sync(mask, bad)) {
#pragma unroll
    for (int j = 0; j < N; ++j) store_value(out, dtype, base+j, NAN);
    return;
  }
  int s;
  if constexpr (Mode == -1) {
    // FP64 norm accumulation avoids overflow on finite FP32 entries.
    double norm = 0.;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      double col = ::fabs(double(a.d[j]));
      if constexpr (Pair) col += ::fabs(double(a.u[j]));
#pragma unroll
      for (int offset = N/2; offset; offset /= 2)
        col += __shfl_xor_sync(mask, col, offset, N);
      norm = ::fmax(norm, col);
    }
    s = norm > theta18 ? int(::ceil(::log2(norm/theta18))) : 0;
  } else if constexpr (Mode == 2) s = 2;
  else s = requested_s;
  const float scale = ::ldexpf(1.f, -s);
#pragma unroll
  for (int j = 0; j < N; ++j) {
    a.d[j] *= scale;
    if constexpr (Pair) a.u[j] *= scale;
  }
  auto result = t18(a, mask, row);
  if constexpr (Mode == 2) {
#pragma unroll
    for (int i = 0; i < 2; ++i) result = product(result, result, mask);
  } else {
    for (int i = 0; i < s; ++i) result = product(result, result, mask);
  }
#pragma unroll
  for (int j = 0; j < N; ++j)
    store_value(out, dtype, base+j, Pair ? result.u[j] : result.d[j]);
}

} // namespace flash_hypercons::small_matrix_exp::tiny

namespace flash_hypercons::small_matrix_exp::scalar {
// One CTA per matrix. Each lane owns consecutive row-major tile elements at
// a block-sized stride. Polynomial intermediates remain in registers; only
// operands of the current product are staged in shared memory.
template <int N, int Threads, bool Pair>
struct Cooperative {
  static constexpr int Elements = N*N/Threads;
  static constexpr int Columns = Elements >= 16 ? 4 : Elements >= 2 ? 2 : 1;
  static constexpr int Rows = Elements/Columns;
  using Ops = Cooperative<N,Threads,Pair>;
  __device__ __forceinline__ static int index(int e) {
    const int row = threadIdx.x/(N/Columns) + (e/Columns)*(Threads/(N/Columns));
    const int col = (threadIdx.x%(N/Columns))*Columns + e%Columns;
    return row*N+col;
  }
  struct Tile {
    cute::array<float, Elements> d;
    cute::array<float, Pair ? Elements : 1> u;
  };
  float* shared;

  __device__ __forceinline__ Tile product(const Tile& a, const Tile& b) const {
    const auto operands = cute::make_tensor(
        cute::make_smem_ptr(shared),
        cute::Layout<
            cute::Shape<cute::Int<N>, cute::Int<N>, cute::Int<Pair ? 4 : 2>>,
            cute::Stride<cute::Int<N>, cute::_1, cute::Int<N*N>>>{});
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      const int i = Ops::index(e);
      shared[i] = a.d[e];
      shared[N*N+i] = b.d[e];
      if constexpr (Pair) {
        shared[2*N*N+i] = a.u[e];
        shared[3*N*N+i] = b.u[e];
      }
    }
    __syncthreads();
    Tile result;
    float du[Elements], ud[Elements];
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      result.d[e] = 0.f;
      if constexpr (Pair) { du[e] = 0.f; ud[e] = 0.f; }
    }
    // Reuse each A value across Columns outputs and each B value across Rows.
#pragma unroll
    for (int k=0; k<N; ++k) {
      float bd[Columns], b_upper[Columns];
#pragma unroll
      for (int col=0; col<Columns; ++col) {
        const int j = index(col)%N;
        bd[col] = operands(k, j, cute::_1{});
        if constexpr (Pair) b_upper[col] = operands(k, j, cute::_3{});
      }
#pragma unroll
      for (int row=0; row<Rows; ++row) {
        const int i = index(row*Columns)/N;
        const float ad = operands(i, k, cute::_0{});
        float au = 0.f;
        if constexpr (Pair) au = operands(i, k, cute::_2{});
#pragma unroll
        for (int col=0; col<Columns; ++col) {
          const int e = row*Columns+col;
          result.d[e] = ::fmaf(ad,bd[col],result.d[e]);
          if constexpr (Pair) {
            du[e] = ::fmaf(ad,b_upper[col],du[e]);
            ud[e] = ::fmaf(au,bd[col],ud[e]);
          }
        }
      }
    }
    if constexpr (Pair) {
#pragma unroll
      for (int e=0; e<Elements; ++e) result.u[e] = du[e]+ud[e];
    }
    // Every reader must finish before the next product reuses shared storage.
    __syncthreads();
    return result;
  }

  __device__ __forceinline__ Tile add(const Tile& a, const Tile& b) const {
    Tile c;
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      c.d[e] = a.d[e] + b.d[e];
      if constexpr (Pair) c.u[e] = a.u[e] + b.u[e];
    }
    return c;
  }

  __device__ __forceinline__ Tile combine(const Tile& a, const Tile& a2,
      const Tile& a3, const Tile& a6, float c0, float c1, float c2,
      float c3, float c6) const {
    Tile b;
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      const int i = Ops::index(e);
      b.d[e] = c0*float(i/N == i%N) + c1*a.d[e] + c2*a2.d[e]
          + c3*a3.d[e] + c6*a6.d[e];
      if constexpr (Pair)
        b.u[e] = c1*a.u[e] + c2*a2.u[e] + c3*a3.u[e] + c6*a6.u[e];
    }
    return b;
  }

  __device__ __forceinline__ Tile t18(const Tile& a) const {
    auto a2 = product(a,a);
    auto a3 = product(a2,a);
    auto a6 = product(a3,a3);
    auto b1 = combine(a,a2,a3,a6,0.f,-0.10036558103014462f,
        -0.008029246482411570f,-0.000892138498045730f,0.f);
    auto b2 = combine(a,a2,a3,a6,0.f,0.39784974949645076f,
        1.36783778460411719f,0.49828962252538268f,-0.00063789819459247233f);
    auto b3 = combine(a,a2,a3,a6,-10.96763960529620626f,1.68015813878906197f,
        0.05717798464788655f,-0.00698210122488052f,3.349750170860705e-05f);
    auto b4 = combine(a,a2,a3,a6,-0.09043168323908106f,-0.06764045190713819f,
        0.06759613017740597f,0.02955525704293155f,-1.391802575160607e-05f);
    auto b5 = combine(a,a2,a3,a6,0.f,0.f,-0.09233646193671186f,
        -0.01693649390020817f,-1.400867981820361e-05f);
    auto a9 = add(product(b1,b5),b4);
    return add(b2,product(add(b3,a9),a9));
  }
};

__device__ __forceinline__ void store_value(void* out, int dtype, int64_t i, float v) {
  if (dtype == 0) static_cast<float*>(out)[i] = v;
  else if (dtype == 1) static_cast<c10::BFloat16*>(out)[i] = c10::BFloat16(v);
  else static_cast<c10::Half*>(out)[i] = c10::Half(v);
}

template <int N, int Threads, bool Pair, bool FixedTwo>
__device__ __forceinline__ void compute(
    const float* x, const float* g, void* out, int dtype, bool adaptive, int requested_s) {
  using Ops = Cooperative<N,Threads,Pair>;
  __shared__ float operands[(Pair ? 4 : 2)*N*N];
  __shared__ double column_norm[N];
  __shared__ int shared_s;
  Ops ops{operands};
  typename Ops::Tile a;
  const int64_t base = int64_t(blockIdx.x)*N*N;
  bool bad = false;
#pragma unroll
  for (int e=0; e<Ops::Elements; ++e) {
    const int i = Ops::index(e);
    a.d[e] = x[base+i];
    bad |= !::isfinite(a.d[e]);
    if constexpr (Pair) { a.u[e] = g[base+i]; bad |= !::isfinite(a.u[e]); }
  }
  if (__syncthreads_or(bad)) {
#pragma unroll
    for (int e=0; e<Ops::Elements; ++e)
      store_value(out,dtype,base+Ops::index(e),NAN);
    return;
  }
  int s = 2;
  if constexpr (!FixedTwo) {
    s = requested_s;
    if (adaptive) {
#pragma unroll
      for (int e=0; e<Ops::Elements; ++e) {
        const int i = Ops::index(e);
        operands[i] = a.d[e];
        if constexpr (Pair) operands[N*N+i] = a.u[e];
      }
      __syncthreads();
      if (threadIdx.x < N) {
        double v = 0.;
#pragma unroll
        for (int row=0; row<N; ++row) {
          v += ::fabs(double(operands[row*N+threadIdx.x]));
          if constexpr (Pair) v += ::fabs(double(operands[N*N+row*N+threadIdx.x]));
        }
        column_norm[threadIdx.x] = v;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        double norm = 0.;
        for (int col=0; col<N; ++col) norm = ::fmax(norm,column_norm[col]);
        shared_s = norm > 3.010066362817634 ? int(::ceil(::log2(norm/3.010066362817634))) : 0;
      }
      __syncthreads();
      s = shared_s;
    }
  }
  const float scale = ::ldexpf(1.f,-s);
#pragma unroll
  for (int e=0; e<Ops::Elements; ++e) {
    a.d[e] *= scale;
    if constexpr (Pair) a.u[e] *= scale;
  }
  auto result = ops.t18(a);
  if constexpr (FixedTwo) {
#pragma unroll
    for (int i=0; i<2; ++i) result = ops.product(result,result);
  } else {
    for (int i=0; i<s; ++i) result = ops.product(result,result);
  }
#pragma unroll
  for (int e=0; e<Ops::Elements; ++e)
    store_value(out,dtype,base+Ops::index(e),Pair ? result.u[e] : result.d[e]);
}

template <int N, int Threads, bool Pair, bool FixedTwo>
__global__ __launch_bounds__(Threads) void tiled_kernel(
    const float* x, const float* g, void* out, int dtype, bool adaptive, int requested_s) {
  compute<N, Threads, Pair, FixedTwo>(x, g, out, dtype, adaptive, requested_s);
}
} // namespace flash_hypercons::small_matrix_exp::scalar
