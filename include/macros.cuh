#ifndef MACROS_H
#define MACROS_H

#include <cusolverDn.h>

#define CAST_THRUST(x) thrust::raw_pointer_cast(x)
#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

inline void gpuAssert(cusolverStatus_t code, const char *file, int line,
                      bool abort = true) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr, "cuSOLVER ERROR: %d %s %d\n", code, file,
            line);
    if (abort)
      exit(code);
  }
}

// inline void gpuAssert(cublasStatus_t code, const char *file, int line,
//                       bool abort = true) {
//   if (code != CUBLAS_STATUS_SUCCESS) {
//     fprintf(stderr, "GPUassert: %s %s %d\n", err, file,
//             line);
//     if (abort)
//       exit(code);
//   }
// }

using index_t = unsigned;
using value_t = double;

#endif /* MACROS_H */
