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
#include "parse_args.hpp"



int main(int argc, char* argv[]) {
  const auto [filename, matrixSizes] = parse_args(argc, argv);

  COOTensor3 X(
    filename,
    true
  );

  tucker_decomp(X, matrixSizes);
}
