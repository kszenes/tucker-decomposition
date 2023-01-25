#ifndef SVD_H
#define SVD_H
#include "DenseMatrix.cuh"
#include "includes.cuh"
#include "macros.cuh"
#include "CSFTensor3.cuh"

#include <mkl_lapacke.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

thrust::host_vector<value_t> call_svd(
    thrust::host_vector<value_t> &matrix, const size_t nrows,
    const size_t ncols, const bool only_U = true
);

void svd(
  const CSFTensor3& csf,
  thrust::device_vector<value_t>& sspTensor,
  DenseMatrix& U_to_update,
  const index_t subchunkSize,
  const bool on_gpu
);

#endif /* SVD_H */