#ifndef HOST_CONTRACTIONS_H
#define HOST_CONTRACTIONS_H

#include <numeric>
#include <functional>

#include "CSFTensor3.cuh"
#include "COOTensor3.cuh"
#include "DenseMatrix.cuh"

#include "includes.cuh"
#include "macros.cuh"

thrust::host_vector<value_t> host_first_contraction(
  const COOTensor3& X, const DenseMatrix& U, const index_t contractedMode
);

thrust::host_vector<value_t> contract(
  const thrust::host_vector<value_t>& in,
  const thrust::host_vector<value_t>& Uvals,
  const std::vector<index_t>& out_shape,
  const std::vector<index_t>& in_shape,
  const index_t Ucols,
  const index_t contractedMode
);

thrust::host_vector<value_t> host_second_contraction(
  const CSFTensor3& csf,
  const thrust::host_vector<value_t>& denseTensor,
  const std::vector<DenseMatrix>& Us
);

thrust::host_vector<value_t> host_third_contraction(
  const CSFTensor3& csf,
  const thrust::host_vector<value_t>& denseTensor,
  const std::vector<DenseMatrix>& Us
);

#endif /* HOST_CONTRACTIONS_H */
