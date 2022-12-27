#include "DenseMatrix.cuh"
#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "ttm.cuh"
#include "tucker.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mkl_lapacke.h>
#include <string>

int main() {
  COOTensor3 X("example_tensors/sparse_5_5_5.tns", true);
  X.print();

  tucker_decomp(X, {2, 2, 2});

  // csf.buildCSFTensor3(X.d_modes, X.d_values);
  // std::cout << "Done\n";

  // DenseMatrix U(5, 2, "ones");
  // U.print();

  // SparseTensor2 Y(X.mode_sizes, 2, U.ncols);
  // std::cout << "Y.CHUNK_SIZE := " << Y.chunk_size << '\n';
  // tensor_times_matrix(Y, X, U, 2, true);
  // std::cout << "Y.CHUNK_SIZE := " << Y.chunk_size << '\n';

  // tensor_times_matrix(Y, U);


}