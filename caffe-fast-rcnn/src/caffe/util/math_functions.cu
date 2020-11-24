#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void logistic_activate_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = 0.5 * tanh(0.5 * a[index])+ 0.5;
  }
}
template <>
void caffe_gpu_logistic_activate<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  logistic_activate_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}
template <>
void caffe_gpu_logistic_activate<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  logistic_activate_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}
template <typename Dtype>
__global__ void hard_sigmoid_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fminf(1., fmaxf(0., a[index] * 0.2 + 0.5));
  }
}
template <>
void caffe_gpu_hard_sigmoid<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hard_sigmoid_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}
template <>
void caffe_gpu_hard_sigmoid<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hard_sigmoid_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void set_conv_kernel(const int n, const int m, const Dtype* x, Dtype* y) {
  //CUDA_KERNEL_LOOP(index1, n) {
  //  CUDA_KERNEL_LOOP(index2, m) {
  //    y[index2 + index1 * m] = x[index1];
  //    printf("%d %d %d\n",index1,index2,index2 + index1 * m);
  //  }
  //}
  CUDA_KERNEL_LOOP(index1, n*m) {
    y[index1] = x[(int)index1/m];
  }
}

template <typename Dtype>
void caffe_gpu_set_conv(const int N, const int M, const Dtype* X, Dtype* Y) {
  set_conv_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, X, Y);
}


template <>
void caffe_gpu_find_max(const int N, const double* X, double* max) {
    int maxIndex;
    CUBLAS_CHECK(cublasIdamax(Caffe::cublas_handle(), N, X, 1, &maxIndex));
    cudaMemcpy(max, X+maxIndex-1, sizeof(double), cudaMemcpyDeviceToHost);
}

template <>
void caffe_gpu_find_max(const int N, const float* X, float* max) {
    int maxIndex;
    CUBLAS_CHECK(cublasIsamax(Caffe::cublas_handle(), N, X, 1, &maxIndex));
    cudaMemcpy(max, X+maxIndex-1, sizeof(float), cudaMemcpyDeviceToHost);
    //CUBLAS_CHECK(cublasIsamax(Caffe::cublas_handle(), N, X, 1, index));
    //index += 1;
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template void caffe_gpu_set_conv<int>(const int N, const int M, const int* X, int* Y);
template void caffe_gpu_set_conv<float>(const int N, const int M, const float* X, float* Y);
template void caffe_gpu_set_conv<double>(const int N, const int M, const double* X, double* Y);


template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void fc_mul_kernel(const int n, const int m, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index*m] = a[index] * b[index];
  }
}

template <typename Dtype>
__global__ void fc_mul_kernel_tran(const int n, const int m, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index*m] = a[index] * b[index*m];
  }
}

template <>
void caffe_gpu_fc_mul<float>(const int N, const int M, const float* a, const float* b,
    float* y, const int tran) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if(tran == 0)
    fc_mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, a, b, y);
  else
    fc_mul_kernel_tran<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, a, b, y);
}

template <>
void caffe_gpu_fc_mul<double>(const int N, const int M, const double* a, const double* b,
    double* y, const int tran) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if(tran == 0)
    fc_mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, a, b, y);
  else
    fc_mul_kernel_tran<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, a, b, y);
}

template <typename Dtype>
__global__ void conv_add_kernel(const int n, const int m, const Dtype* x,
    Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
#if (CONV_ADD_PARALLEL_LEVEL ==1 )
  y[index + m*0] += x[index + m*0];
#elif (CONV_ADD_PARALLEL_LEVEL ==2 )
  y[index + m*0] += x[index + m*0];
  y[index + m*1] += x[index + m*1];
#elif (CONV_ADD_PARALLEL_LEVEL ==3 )
  y[index + m*0] += x[index + m*0];
  y[index + m*1] += x[index + m*1];
  y[index + m*2] += x[index + m*2];
  y[index + m*3] += x[index + m*3];
#endif
      //y[index + m*0] += x[index + m*0];
      //y[index + m*1] += x[index + m*1];
      //y[index + m*2] += x[index + m*2];
      //y[index + m*3] += x[index + m*3];
      //y[index + m*4] += x[index + m*4];
      //y[index + m*5] += x[index + m*5];
      //y[index + m*6] += x[index + m*6];
      //y[index + m*7] += x[index + m*7];
      //y[index + m*8] += x[index + m*8];
      //y[index + m*9] += x[index + m*9];
  }
}

template <>
void caffe_gpu_conv_add<float>(const int N, const int M, const float* x,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  conv_add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, x, y);
}

template <>
void caffe_gpu_conv_add<double>(const int N, const int M, const double* x,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  conv_add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, x, y);
}

template <typename Dtype>
__global__ void caffe_analyze_max_kernel(const int n, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
      y[index] = max(x[index], y[index]);
  }
}

template <>
void caffe_gpu_analyze_max<float>(const int N, const float* x, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_analyze_max_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}

template <>
void caffe_gpu_analyze_max<double>(const int N, const double* x, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_analyze_max_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}


template <>
void caffe_gpu_analyze_max_serial<float>(const int N, const float* x, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  //caffe_analyze_max_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
  //    N, x, y);

  for(int i = 0; i < N / ANALYZE_MODE_PARALLEL_FACTOR; i ++)
  {
    caffe_analyze_max_kernel<float><<<CAFFE_GET_BLOCKS(ANALYZE_MODE_PARALLEL_FACTOR), CAFFE_CUDA_NUM_THREADS>>>(
        ANALYZE_MODE_PARALLEL_FACTOR, x + i * ANALYZE_MODE_PARALLEL_FACTOR, y);
  }
  if(N % ANALYZE_MODE_PARALLEL_FACTOR != 0)
    caffe_analyze_max_kernel<float><<<CAFFE_GET_BLOCKS(N%ANALYZE_MODE_PARALLEL_FACTOR), CAFFE_CUDA_NUM_THREADS>>>(
        N % ANALYZE_MODE_PARALLEL_FACTOR, x+( N / ANALYZE_MODE_PARALLEL_FACTOR ) * ANALYZE_MODE_PARALLEL_FACTOR, y);
}

template <>
void caffe_gpu_analyze_max_serial<double>(const int N, const double* x, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  //caffe_analyze_max_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
  //    N, x, y);

  for(int i = 0; i < N / ANALYZE_MODE_PARALLEL_FACTOR; i ++)
  {
    caffe_analyze_max_kernel<double><<<CAFFE_GET_BLOCKS( ANALYZE_MODE_PARALLEL_FACTOR ), CAFFE_CUDA_NUM_THREADS>>>(
        ANALYZE_MODE_PARALLEL_FACTOR, x + i * ANALYZE_MODE_PARALLEL_FACTOR, y);
  }
  if(N%ANALYZE_MODE_PARALLEL_FACTOR != 0)
    caffe_analyze_max_kernel<double><<<CAFFE_GET_BLOCKS( N % ANALYZE_MODE_PARALLEL_FACTOR), CAFFE_CUDA_NUM_THREADS>>>(
        N % ANALYZE_MODE_PARALLEL_FACTOR, x + ( N / ANALYZE_MODE_PARALLEL_FACTOR ) * ANALYZE_MODE_PARALLEL_FACTOR, y);
}

template <typename Dtype>
__global__ void caffe_analyze_min_kernel(const int n, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
      y[index] = min(x[index], y[index]);
  }
}

template <>
void caffe_gpu_analyze_min<float>(const int N, const float* x, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_analyze_min_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}

template <>
void caffe_gpu_analyze_min<double>(const int N, const double* x, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_analyze_min_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}


template <>
void caffe_gpu_analyze_min_serial<float>(const int N, const float* x, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  //caffe_analyze_min_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
  //    N, x, y);
  for(int i = 0; i < N / ANALYZE_MODE_PARALLEL_FACTOR; i ++)
  {
    caffe_analyze_min_kernel<float><<<CAFFE_GET_BLOCKS(ANALYZE_MODE_PARALLEL_FACTOR), CAFFE_CUDA_NUM_THREADS>>>(
        ANALYZE_MODE_PARALLEL_FACTOR, x + i * ANALYZE_MODE_PARALLEL_FACTOR, y);
  }
  if(N%ANALYZE_MODE_PARALLEL_FACTOR!=0)
    caffe_analyze_min_kernel<float><<<CAFFE_GET_BLOCKS(N%ANALYZE_MODE_PARALLEL_FACTOR), CAFFE_CUDA_NUM_THREADS>>>(
        N % ANALYZE_MODE_PARALLEL_FACTOR, x + ( N / ANALYZE_MODE_PARALLEL_FACTOR ) * ANALYZE_MODE_PARALLEL_FACTOR, y);
}

template <>
void caffe_gpu_analyze_min_serial<double>(const int N, const double* x, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  //caffe_analyze_min_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
  //    N, x, y);
  for(int i = 0; i < N/ANALYZE_MODE_PARALLEL_FACTOR; i ++)
  {
    caffe_analyze_min_kernel<double><<<CAFFE_GET_BLOCKS(ANALYZE_MODE_PARALLEL_FACTOR), CAFFE_CUDA_NUM_THREADS>>>(
        ANALYZE_MODE_PARALLEL_FACTOR, x+i*ANALYZE_MODE_PARALLEL_FACTOR, y);
  }
  if(N%ANALYZE_MODE_PARALLEL_FACTOR!=0)
    caffe_analyze_min_kernel<double><<<CAFFE_GET_BLOCKS(N%ANALYZE_MODE_PARALLEL_FACTOR), CAFFE_CUDA_NUM_THREADS>>>(
        N%ANALYZE_MODE_PARALLEL_FACTOR, x+(N/ANALYZE_MODE_PARALLEL_FACTOR)*ANALYZE_MODE_PARALLEL_FACTOR, y);
}

template <typename Dtype>
__global__ void caffe_gpu_STE_kernel(const int n,const Dtype min, const Dtype max, const Dtype* x, Dtype* x_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if( x[index] >= max || x[index] <= min )
      x_diff[index] = 0;
  }
}

template <>
void caffe_gpu_STE<float>(const int N, const float min, const float max, const float* x, float* x_diff) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_STE_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, min, max, x, x_diff);
}

template <>
void caffe_gpu_STE<double>(const int N, const double min, const double max, const double* x, double* x_diff) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_STE_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, min, max, x, x_diff);
}

template <typename Dtype>
__global__ void caffe_gpu_STE_q_stat_kernel(const int n, const Dtype* x, Dtype* x_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if( x[index] > 0.5 )
      x_diff[index] = 0;
  }
}

template <>
void caffe_gpu_STE_q_stat<float>(const int N, const float* x, float* x_diff) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_STE_q_stat_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, x_diff);
}

template <>
void caffe_gpu_STE_q_stat<double>(const int N, const double* x, double* x_diff) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_STE_q_stat_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, x_diff);
}

template <typename Dtype>
__global__ void IVS_gpu_analyze_q_stat_thresh_kernel(Dtype* top_data, Dtype min, Dtype max, int n) {
  CUDA_KERNEL_LOOP(index, n) {
    if( top_data[index] >= max || top_data[index] <= min )
      top_data[index] = 1;
    else
      top_data[index] = 0;
  }
}

template <>
void IVS_gpu_analyze_set_thresh<double>(const int N, double* data, double min, double max) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  IVS_gpu_analyze_q_stat_thresh_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          data, min, max, N);
}

template <>
void IVS_gpu_analyze_set_thresh<float>(const int N, float* data, float min, float max) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  IVS_gpu_analyze_q_stat_thresh_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          data, min, max, N);
}

template <typename Dtype>
__global__ void IVS_gpu_analyze_q_stat_set_kernel(Dtype* IVS_top_q_stat_data, Dtype* top_data, int n, int m) {
  CUDA_KERNEL_LOOP(index, n) {
    IVS_top_q_stat_data[index] = top_data[index]/m;
  }
}

template <>
void IVS_gpu_analyze_q_stat<float>(float* IVS_top_q_stat_data, float* top_data, float min_add, float max_add, 
                                    float min_mult, float max_mult, int N, int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  IVS_gpu_analyze_q_stat_thresh_kernel<float><<<CAFFE_GET_BLOCKS(N*(int)(M/2)), CAFFE_CUDA_NUM_THREADS>>>(
          top_data, min_add, max_add, N*(int)(M/2));
  IVS_gpu_analyze_q_stat_thresh_kernel<float><<<CAFFE_GET_BLOCKS(N * ((int) (M/2) + M%2 )), CAFFE_CUDA_NUM_THREADS>>>(
          top_data + N * (int)(M/2), min_mult, max_mult, N * ((int) (M/2) + M%2 ) );
  for(int col_left = M; col_left != 1; col_left /= 2){
    caffe_gpu_add( col_left/2*N, top_data, top_data + col_left/2*N, top_data);
    if(col_left%2 == 1){
        caffe_gpu_add( N, top_data, top_data + N * (col_left-1), top_data);
    }
  }
  IVS_gpu_analyze_q_stat_set_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          IVS_top_q_stat_data, top_data, N, M);
}

template <>
void IVS_gpu_analyze_q_stat<double>(double* IVS_top_q_stat_data, double* top_data, double min_add, double max_add, 
                                    double min_mult, double max_mult, int N, int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  IVS_gpu_analyze_q_stat_thresh_kernel<double><<<CAFFE_GET_BLOCKS(N*(int)(M/2)), CAFFE_CUDA_NUM_THREADS>>>(
          top_data, min_add, max_add, N*(int)(M/2));
  IVS_gpu_analyze_q_stat_thresh_kernel<double><<<CAFFE_GET_BLOCKS(N * ((int) (M/2)+ M%2 )), CAFFE_CUDA_NUM_THREADS>>>(
          top_data + N * (int)(M/2), min_mult, max_mult, N * ((int) (M/2)+ M%2 ) );
  for(int col_left = M; col_left != 1; col_left /= 2){
    caffe_gpu_add( col_left/2*N, top_data, top_data + col_left/2*N, top_data);
    if(col_left%2 == 1){
        caffe_gpu_add( N, top_data, top_data + N * (col_left-1), top_data);
    }
  }
  IVS_gpu_analyze_q_stat_set_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          IVS_top_q_stat_data, top_data, N, M);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}


template <typename Dtype>
__global__ void conv_mul_kernel(const int n, const int m, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index%m];
  }
}

template <>
void caffe_gpu_conv_mul<float>(const int N, const int M, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  conv_mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, a, b, y);
}


template <>
void caffe_gpu_conv_mul<double>(const int N, const int M, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  conv_mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
