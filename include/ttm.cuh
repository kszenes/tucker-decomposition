#ifndef TTM_H
#define TTM_H

#include "DenseMatrix.cuh"
#include "SparseTensor2.cuh"
#include "SparseTensor3.cuh"
#include "includes.cuh"
#include "macros.cuh"
#include "svd.cuh"
#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"

__global__ void
spt_TTMRankRBNnzKernelSM(value_t *Y_val, index_t Y_stride, index_t Y_nnz,
                         const index_t *__restrict__ X_inds_m,
                         const index_t *__restrict__ fiberidx_val,
                         const value_t *__restrict__ U_val, index_t U_nrows,
                         index_t U_ncols, index_t U_stride);

void tensor_times_matrix(SparseTensor2 &Y, SparseTensor3 &X, DenseMatrix &U,
                         index_t mode, bool skip_sort);
void tensor_times_matrix(SparseTensor2 &X, DenseMatrix &U);
void call_contraction_kernel(
    const CSFTensor3 &X, const DenseMatrix &U,
    thrust::device_vector<value_t> &out_values, const size_t out_mode
);

#endif /* TTM_H */
