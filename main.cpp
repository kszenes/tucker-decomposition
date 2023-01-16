#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"
#include "timer.h"
#include "ttm.cuh"
#include "tucker.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mkl_lapacke.h>
#include <string>

int main() {
  // TODO: Bug with sparse_tiny.tns
  COOTensor3 X(
    "/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_100_50.tns",
    true
  );

  // std::vector<index_t> sizes = {30, 25, 20, 15, 10, 5, 2};
  // for (const auto e : sizes) {
  //   std::vector<index_t> matrixSizes{e, e, e};
  //   fmt::print("MatrixSizes: {}\n", matrixSizes);
  //   tucker_decomp(X, matrixSizes);
  // }

  const index_t rank = 16;
  std::vector<index_t> matrixSizes{rank, rank, rank};
  tucker_decomp(X, matrixSizes);

}
