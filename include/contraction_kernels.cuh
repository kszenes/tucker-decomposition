#ifndef CONTRACTION_KERNELS_H
#define CONTRACTION_KERNELS_H

#include "includes.cuh"
#include "macros.cuh"

__global__ void spt_TTMRankRBNnzKernelSM(
    value_t *Y_val, const index_t Y_stride, const index_t Y_nnz,
    const value_t *__restrict__ X_val, const index_t *__restrict__ X_inds_m,
    const index_t *__restrict__ fiberidx_val, const value_t *__restrict__ U_val,
    const index_t U_nrows, const index_t U_ncols, const index_t U_stride
);

__global__ void ttm_semisparse_kernel(
    const index_t *__restrict__ fiber_ptr,
    const index_t *__restrict__ X_indices_m, const index_t nrows,
    const index_t ncols, const index_t out_num_chunks,
    const index_t out_chunk_size, const index_t X_chunk_size,
    value_t *__restrict__ out_values, const value_t *__restrict__ X_values,
    const value_t *__restrict__ U_values
);

#endif /* CONTRACTION_KERNELS_H */
