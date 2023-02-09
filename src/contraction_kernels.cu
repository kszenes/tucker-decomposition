#include "contraction_kernels.cuh"

__global__ void spt_TTMRankRBNnzKernelSM(
    value_t *Y_val, const index_t Y_stride, const index_t Y_nnz,
    const value_t *__restrict__ X_val, const index_t *__restrict__ X_inds_m,
    const index_t *__restrict__ fiberidx_val, const value_t *__restrict__ U_val,
    const index_t U_nrows, const index_t U_ncols, const index_t U_stride
) {
  extern __shared__ value_t mem_pool[];
  value_t *const Y_shr = (value_t *)mem_pool; // size U_ncols

  index_t num_loops_nnz = 1;
  index_t const nnz_per_loop = gridDim.x * blockDim.y;
  if (Y_nnz > nnz_per_loop) {
    num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
  }

  // Global indices of Y: Fiber = x and Inner fiber = r
  // Local indices: tidx and tidy
  const index_t tidx = threadIdx.x;
  const index_t tidy = threadIdx.y;
  index_t x;
  const index_t num_loops_r = U_ncols / blockDim.x;
  const index_t rest_loop = U_ncols - num_loops_r * blockDim.x;
  index_t r; // column idx of U

  for (index_t l = 0; l < num_loops_r;
       ++l) {                  // blockDim.x parallelised over cols(U)
    r = tidx + l * blockDim.x; // r: column idx of U
    for (index_t nl = 0; nl < num_loops_nnz; ++nl) { // Grid strided-pattern?
      x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

      Y_shr[tidy * blockDim.x + tidx] = 0;
      __syncthreads();

      if (x < Y_nnz) { // Why is this not above at line 348
        const index_t inz_begin = fiberidx_val[x];
        const index_t inz_end = fiberidx_val[x + 1];
        for (index_t i = inz_begin; i < inz_end; ++i) { // loop over a n-fiber
          const index_t row = X_inds_m[i];              // row of U
          // Loop over nnz in n-fiber of X and multiply with corresponding
          // U col elements and accumulate in single element of Y
          Y_shr[tidy * blockDim.x + tidx] +=
              X_val[i] * U_val[row * U_stride + r]; // Original
              // printf("X_val[%d] = %f\tU_val[%d, %d] = %f\n",
              // i, X_val[i], row, r, U_val[row * U_stride + r]);
        }
        __syncthreads();

        Y_val[x * Y_stride + r] = Y_shr[tidy * blockDim.x + tidx];
        __syncthreads();
      }
    }
  }

  if (rest_loop > 0 && tidx < rest_loop) {
    r = tidx + num_loops_r * blockDim.x;

    for (index_t nl = 0; nl < num_loops_nnz; ++nl) {
      x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

      Y_shr[tidy * blockDim.x + tidx] = 0;
      __syncthreads();

      if (x < Y_nnz) {
        const index_t inz_begin = fiberidx_val[x];
        const index_t inz_end = fiberidx_val[x + 1];
        for (index_t i = inz_begin; i < inz_end; ++i) {
          const index_t row = X_inds_m[i];
          Y_shr[tidy * blockDim.x + tidx] +=
              X_val[i] * U_val[row * U_stride + r]; // Original
          // Y_shr[tidy*blockDim.x + tidx] += X_val[i] * U_val[r*U_stride +
          // row];
        }
        __syncthreads();

        Y_val[x * Y_stride + r] = Y_shr[tidy * blockDim.x + tidx];
        __syncthreads();
      }
    }
  }
}

// TODO: verify unsure of result!!
// __global__ void ttm_semisparse_kernel(
//     const index_t *__restrict__ fiber_ptr,
//     const index_t *__restrict__ X_indices_m, const index_t nrows,
//     const index_t ncols, const index_t out_num_chunks,
//     const index_t out_chunk_size, const index_t X_chunk_size,
//     value_t *__restrict__ out_values, const value_t *__restrict__ X_values,
//     const value_t *__restrict__ U_values
// ) {

//   for (int cur_chunk = threadIdx.x + blockIdx.x * blockDim.x;
//        cur_chunk < X_chunk_size; cur_chunk += blockDim.x * gridDim.x) {
//     for (int nnz_element = threadIdx.y + blockIdx.y * blockDim.y;
//          nnz_element < out_num_chunks; nnz_element += gridDim.y) {
//       int fiber_begin = fiber_ptr[nnz_element];
//       int fiber_end = fiber_ptr[nnz_element + 1];
//       for (int fiber_idx = fiber_begin; fiber_idx < fiber_end; ++fiber_idx) {
//         int row = X_indices_m[fiber_idx];
//         int chunk_start = fiber_idx * X_chunk_size;
//         for (int col = 0; col < ncols; ++col) {
//           out_values
//               [nnz_element * out_chunk_size + col * X_chunk_size + cur_chunk] +=
//               X_values[chunk_start + cur_chunk] * U_values[row * ncols + col];
//         }
//       }
//     }
//   }
// }

__global__ void ttm_semisparse_kernel(
    const index_t *__restrict__ fiber_ptr,
    const index_t *__restrict__ X_indices_m, const index_t nrows,
    const index_t ncols, const index_t out_num_chunks,
    const index_t out_chunk_size, const index_t X_chunk_size,
    value_t *__restrict__ out_values, const value_t *__restrict__ X_values,
    const value_t *__restrict__ U_values
) {
    size_t i = blockIdx.x;             // i := mode-n fiber
    size_t inz_begin = fiber_ptr[i];    // inz_begin/end := global indices for monde-n fiber of X
    size_t inz_end = fiber_ptr[i + 1];
    size_t r = threadIdx.x;
    for(size_t k = threadIdx.y; k < X_chunk_size; k += blockDim.y) {
        value_t accumulate = 0;
        for(size_t j = inz_begin; j < inz_end; ++j) { // loop over fiber i
            size_t c = X_indices_m[j]; // get mode-n index of X: c ∈ [1, size(mode-n)]
            // printf("c = %d; j = %d\n", (unsigned) c, (unsigned) j);
            if(c < nrows && r < ncols) {
                accumulate += X_values[j * X_chunk_size + k] * U_values[c * ncols + r];
                // accumulate += X_values[j * X_chunk_size + k];
                // printf("U_values: %d; X_values: %d\n", (int)(c * ncols + r), (int)(j * X_chunk_size + k));
            }
        }
        out_values[i * out_chunk_size + r * X_chunk_size + k] += accumulate;
        // out_values[0] += accumulate;
    }
}

// __global__ void ttm_last_kernel(
//     const index_t *__restrict__ X_indices_m, const index_t nrows,
//     const index_t ncols, const index_t out_chunk_size,
//     const index_t X_chunk_size, value_t *__restrict__ out_values,
//     const value_t *__restrict__ X_values, const value_t *__restrict__ U_values
// ) {
//   for (int nnz_element = threadIdx.x + blockIdx.x * blockDim.x;
//        nnz_element < out_chunk_size; nnz_element += blockDim.x) {
//     int row = X_indices_m[nnz_element];
//     int chunk_start = fiber_idx * X_chunk_size;
//     for (int col = 0; col < ncols; ++col) {
//       out_values
//           [nnz_element * out_chunk_size + cur_chunk * X_chunk_size + col] +=
//           X_values[chunk_start + cur_chunk] * U_values[row * ncols + col];
//     }
//   }
// }