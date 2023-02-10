#ifndef MACROS_H
#define MACROS_H

#include <cusolverDn.h>
#include <helper_cuda.cuh>

template <typename T>
T* thrust::raw_pointer_cast(thrust::device_vector<T> vec) {
  return thrust::raw_pointer_cast(vec.data());
}

#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR: %s %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

inline void gpuAssert(cusolverStatus_t code, const char *file, int line,
                      bool abort = true) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr, "cuSOLVER ERROR: %s %s:%d\n", _cudaGetErrorEnum(code), file,
            line);
    if (abort)
      exit(code);
  }
}

inline void gpuAssert(cublasStatus_t code, const char *file, int line,
                      bool abort = true) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS error: %s %s:%d\n", _cudaGetErrorEnum(code), file,
            line);
    if (abort)
      exit(code);
  }
}

using index_t = unsigned;
using value_t = double;

#endif /* MACROS_H */
