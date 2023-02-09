#ifndef TTM_H
#define TTM_H

#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"
#include "contraction_kernels.cuh"
#include "includes.cuh"
#include "macros.cuh"
#include "svd.cuh"

thrust::device_vector<value_t> ttm_chain(
    const CSFTensor3 &tensor, std::vector<DenseMatrix> &factor_matrices
);

thrust::device_vector<value_t> contract_first_mode(
    const CSFTensor3 &tensor, const DenseMatrix &matrix
);

thrust::device_vector<value_t> contract_mode(
    const CSFTensor3 &tensor, const DenseMatrix &matrix,
    const thrust::device_vector<value_t>& in_values,
    const index_t contracted_mode, const size_t subchunk_size
);

thrust::device_vector<value_t> contract_last_mode(
    const CSFTensor3 &tensor, const std::vector<DenseMatrix> &matrices,
    const thrust::device_vector<value_t> &in_values,
    const size_t subchunk_size
);

#endif /* TTM_H */
