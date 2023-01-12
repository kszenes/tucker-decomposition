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
  COOTensor3 X("/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_5_5_5.tns", true);
 // X.print();

  // fmt::print("Indices:\n[");
  // fmt::print("{},", X.h_modes[0]);
  // fmt::print("{},", X.h_modes[1]);
  // fmt::print("{}]\n\n", X.h_modes[2]);
  // fmt::print("Values: {}\n\n", X.h_values);

  // CSFTensor3 csf(X, 0);
  // csf.print();

  // index_t colsU = 3;
  // DenseMatrix U(csf.shape.back(), colsU, "random_seed");
  // U.print();
  // fmt::print("U = {}\n", U.h_values);

  // auto out_tensor = thrust::host_vector<value_t>(contract_first_mode(csf, U));
  // fmt::print("out = {}\n", out_tensor);

  std::vector<index_t> matrixSizes{5, 5, 5};



  fmt::print("MatrixSizes: {}\n", matrixSizes);
  tucker_decomp(X, matrixSizes);

  // csf.buildCSFTensor3(X.d_modes, X.d_values);
  // std::cout << "Done\n";

  // DenseMatrix U(5, 2, "ones");
  // U.print();

  // SparseTensor2 Y(X.shape, 2, U.ncols);
  // std::cout << "Y.CHUNK_SIZE := " << Y.chunk_size << '\n';
  // tensor_times_matrix(Y, X, U, 2, true);
  // std::cout << "Y.CHUNK_SIZE := " << Y.chunk_size << '\n';

  // tensor_times_matrix(Y, U);
}