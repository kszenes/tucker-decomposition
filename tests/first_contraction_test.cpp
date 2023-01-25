#include "gtest/gtest.h"

#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"
#include "ttm.cuh"

#include "decompress.hpp"
#include "host_contractions.hpp"

TEST(TensorContraction, FirstContraction) {
  // TODO: Use Random Tensor
  std::string filename{"/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_5_5_5.tns"};
  const std::vector<index_t> colsU{5, 5, 5};

  COOTensor3 X(filename, true);
  // X.print();

  std::vector<DenseMatrix> matrices(3);

  for (index_t i = 0; i < X.nmodes; ++i) {
    matrices[i] = DenseMatrix(X.shape[i], colsU[i], "random_seed");
  }

  // // Test along all modes
  for (index_t sortedMode = 0; sortedMode < X.nmodes; ++sortedMode) {
    CSFTensor3 csf(X, sortedMode);
    // csf.print();
    index_t contractedMode = csf.cyclic_permutation.back();
    const auto& firstMatrix = matrices[contractedMode];
    auto host_reference_out{host_first_contraction(X, firstMatrix, contractedMode)};

    auto sparse_out = thrust::host_vector<value_t>(contract_first_mode(csf, firstMatrix));

    thrust::host_vector<value_t> gpu_out(host_reference_out.size());

    auto mode0{decompress_mode(csf.fptr[0], csf.fidx[0])};
    auto mode1 = thrust::host_vector<index_t>(csf.fidx[1]);

    std::vector<index_t> out_shape;
    for (index_t i = 0; i < X.nmodes; ++i) {
      if (i != csf.cyclic_permutation.back()) {
        out_shape.push_back(X.shape[i]);
      } else {
        out_shape.push_back(firstMatrix.ncols);
      }
    }
    std::vector<index_t> strides{out_shape[1] * out_shape[2], out_shape[2], 1};


    // Densify tensor
    assert(mode0.size() == mode1.size() && "Mode size mismatch after expansion");
    for (index_t i = 0; i < mode0.size(); ++i) {
      auto first_mode = mode0[i];
      auto second_mode = mode1[i];
      auto index = first_mode * strides[csf.cyclic_permutation[0]] +
                  second_mode * strides[csf.cyclic_permutation[1]];
      for (index_t j = 0; j < firstMatrix.ncols; ++j) {
        gpu_out[index + j * strides[csf.cyclic_permutation[2]]] = sparse_out[i * firstMatrix.ncols + j];
      }
    };

    for (index_t i = 0; i < gpu_out.size(); ++i) {
      ASSERT_NEAR(gpu_out[i], host_reference_out[i], 1e-6) << 
        fmt::format("Failed for mode {} at index {}\n", contractedMode, i) <<
        fmt::format("Expected: {}\nGot: {}\n", host_reference_out, gpu_out);
    }
    // fmt::print("gpu_out = {}\n", gpu_out);
    // fmt::print("firstMatrix = {}\n", firstMatrix.h_values);
    // === Second contraction ===
    const auto& secondMatrix = matrices[csf.cyclic_permutation[1]];
    // fmt::print("secondMatrix = {}\n", secondMatrix.h_values);

    auto host_reference_second = host_second_contraction(csf, host_reference_out, matrices);
    // fmt::print("Second contraction host: size={}\n{}\n", host_reference_second.size(), host_reference_second);

    auto penultimateTensor = thrust::host_vector<value_t>(
      contract_second_mode(csf, secondMatrix, sparse_out, firstMatrix.ncols)
    );
    // fmt::print("Second contraction device: size={}\n{}\n", penultimateTensor.size(), penultimateTensor);

    out_shape[csf.cyclic_permutation[0]] = csf.shape[0];
    out_shape[csf.cyclic_permutation[1]] = secondMatrix.ncols;
    out_shape[csf.cyclic_permutation[2]] = firstMatrix.ncols;
    strides[0] = out_shape[1] * out_shape[2];
    strides[1] = out_shape[2];
    strides[2] = 1;
    // fmt::print("2nd Contraction strides = {}\n", strides);

    // Densify tensor
    auto chunkSize = secondMatrix.ncols * firstMatrix.ncols;
    auto subchunkSize = firstMatrix.ncols;
    thrust::host_vector<value_t> second_contraction_dense(host_reference_second.size());
    auto mode0_2nd_contraction = thrust::host_vector<index_t>(csf.fidx[0]);
    for (index_t i = 0; i < mode0_2nd_contraction.size(); ++i) {
      auto first_mode = mode0_2nd_contraction[i];
      for (index_t second_mode = 0; second_mode < secondMatrix.ncols; ++second_mode) {
        for (index_t third_mode = 0; third_mode < firstMatrix.ncols; ++third_mode) {
          
          index_t index = first_mode * strides[csf.cyclic_permutation[0]] +
                          second_mode * strides[csf.cyclic_permutation[1]] +
                          third_mode * strides[csf.cyclic_permutation[2]];

          second_contraction_dense[index] = penultimateTensor[i * chunkSize +
                                                    second_mode * subchunkSize +
                                                    third_mode];
        }
      }
    };
    // fmt::print("second_contraction_dense =\n{}\n", second_contraction_dense);
    // fmt::print("host_reference_second =\n{}\n", host_reference_second);
    for (index_t i = 0; i < second_contraction_dense.size(); ++i) {
      ASSERT_NEAR(second_contraction_dense[i], host_reference_second[i], 1e-6) << 
        fmt::format("Failed for 2nd contraction mode {} at index {}\n", csf.cyclic_permutation[1], i);
    }

    // === Third Contraction ===
    const auto& thirdMatrix = matrices[csf.cyclic_permutation[0]];
    // fmt::print("thirdMatrix: size = {}; vals = {}\n", thirdMatrix.h_values.size(), thirdMatrix.h_values);
    // fmt::print("penultimateTensor: size = {}; vals = {}\n", penultimateTensor.size(), penultimateTensor);
    auto coreTensor = thrust::host_vector<value_t>(
      contract_last_mode(csf, matrices, penultimateTensor, firstMatrix.ncols * secondMatrix.ncols)
    );
    // fmt::print("coreTensor (gpu) = {}\n", coreTensor);
    
    auto host_reference_third = host_third_contraction(csf, host_reference_second, matrices);
    // fmt::print("Expected: {}\nGot: {}\n", host_reference_third, coreTensor);
    // for (index_t i = 0; i < coreTensor.size(); ++i) {
    //   ASSERT_NEAR(coreTensor[i], host_reference_third[i], 1e-6) << 
    //     fmt::format("Failed for 3rd contraction mode {} at index {}\n", csf.cyclic_permutation[0], i);
    // }
  }
}