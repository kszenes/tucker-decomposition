#include "parse_args.hpp"
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

int main(int argc, char* argv[]) {
  const auto params  = parse_args(argc, argv);

  CPUTimer coo_timer;
  coo_timer.start();
  COOTensor3 X(
    params.filename,
    true
  );
  auto time = coo_timer.seconds();

  tucker_decomp(X, params);
  fmt::print("COO/IO:    {} [s]\n", time);
}
