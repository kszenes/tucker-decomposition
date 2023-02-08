#include "gtest/gtest.h"

#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"
#include "ttm.cuh"

#include "decompress.hpp"
#include "host_contractions.hpp"


TEST(TensorContraction, SecondContraction) {
  // TODO: Use Random Tensor
  std::string filename{"/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_5_5_5.tns"};
  const index_t colsU = 3;

  COOTensor3 X(filename, true);

  // Test along all modes
  for (index_t sortedMode = 0; sortedMode < X.nmodes; ++sortedMode) {
    CSFTensor3 csf(X, sortedMode);
    index_t contractedMode = csf.mode_permutation.back();
    DenseMatrix U(csf.shape.back(), colsU, "random");
    auto host_reference_out{host_first_contraction(X, U, contractedMode)};

    auto sparse_out = thrust::host_vector<value_t>(contract_first_mode(csf, U));

    thrust::host_vector<value_t> gpu_out(host_reference_out.size());

    auto mode0{decompress_mode(csf.fptr[0], csf.fidx[0])};
    auto mode1 = thrust::host_vector<index_t>(csf.fidx[1]);

    std::vector<index_t> out_shape;
    for (index_t i = 0; i < X.nmodes; ++i) {
      if (i != csf.mode_permutation.back()) {
        out_shape.push_back(X.shape[i]);
      } else {
        out_shape.push_back(U.ncols);
      }
    }
    std::vector<index_t> strides{out_shape[1] * out_shape[2], out_shape[2], 1};

    // Densify tensor
    assert(mode0.size() == mode1.size() && "Mode size mismatch after expansion");
    for (index_t i = 0; i < mode0.size(); ++i) {
      auto first_mode = mode0[i];
      auto second_mode = mode1[i];
      auto index = first_mode * strides[csf.mode_permutation[0]] +
                  second_mode * strides[csf.mode_permutation[1]];
      for (index_t j = 0; j < U.ncols; ++j) {
        gpu_out[index + j * strides[csf.mode_permutation[2]]] = sparse_out[i * U.ncols + j];
      }
    };

    for (index_t i = 0; i < gpu_out.size(); ++i) {
      ASSERT_NEAR(gpu_out[i], host_reference_out[i], 1e-6) << 
        fmt::format("Failed for mode {} at index {}\n", contractedMode, i);
    }
  }
}