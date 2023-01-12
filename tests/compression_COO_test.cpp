#include "gtest/gtest.h"

#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"
#include "decompress.hpp"

TEST(TestCompression, CSFCompression) {
  // TODO: Make random
  COOTensor3 X("/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_5_5_5.tns", true);

  for (index_t test_mode = 0; test_mode < X.nmodes; ++test_mode) {
    CSFTensor3 csf(X, test_mode);
    X.sort_mode(test_mode);

    std::vector<thrust::host_vector<index_t>> decompressed_modes;
    std::vector<thrust::host_vector<index_t>> reference_modes;
    for (index_t mode = 0; mode < X.nmodes; ++mode) {

      for (index_t i = 0; i < csf.fidx.size() - 1; ++i) {
        reference_modes.emplace_back(X.d_modes[i]);

        thrust::host_vector<index_t> cur_mode(csf.fidx[i]);
        // Recursively decompress
        for (index_t j = i; j < csf.fidx.size() - 1; ++j) {
          thrust::host_vector<index_t> cur_ptr(csf.fptr[j]);
          cur_mode = decompress_mode(cur_ptr, cur_mode);
        }
        decompressed_modes.emplace_back(cur_mode);
      }
      decompressed_modes.emplace_back(csf.fidx.back());
      reference_modes.emplace_back(X.d_modes[csf.fidx.size() - 1]);

      // test modes
      for (index_t i = 0; i < csf.nmodes; ++i) {
        auto true_mode = csf.cyclic_permutation[i];
        ASSERT_TRUE(decompressed_modes[i] == reference_modes[true_mode]) <<
          "Failed for CSF-" << mode << ". Mismatch at mode: " << i << '\n' <<
          fmt::format("Expected: {};\nGot: {}\n", reference_modes[true_mode], decompressed_modes[i]);
      }
    }
  }

}
