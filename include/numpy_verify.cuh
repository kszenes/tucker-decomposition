#ifndef NUMPY_VERIFY_H
#define NUMPY_VERIFY_H

#include "includes.cuh"
#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"

void print_verification_script(
  const COOTensor3 &X,
  const std::vector<CSFTensor3>& CSFTensors,
  const std::vector<DenseMatrix>& factor_matrices,
  const thrust::device_vector<value_t>& coreTensor
);

#endif /* NUMPY_VERIFY_H */
