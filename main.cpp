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

  CPUTimer coo_timer;
  coo_timer.start();
  COOTensor3 X(
    filename,
    true
  );
  auto time = coo_timer.seconds();

  const double tol = 1e-5;
  const int maxiter = 100;
  tucker_decomp(X, matrixSizes, tol, maxiter);
  fmt::print("COO/IO:    {} [s]\n", time);
}
