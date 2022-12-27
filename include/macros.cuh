#ifndef MACROS_H
#define MACROS_H

#define CAST_THRUST(x) thrust::raw_pointer_cast(x)
#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

using index_t = unsigned;
using value_t = float;

#endif /* MACROS_H */
