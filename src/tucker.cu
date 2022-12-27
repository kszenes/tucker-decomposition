#include "ttm.cuh"
#include "tucker.cuh"

void tucker_decomp(const COOTensor3 &X, const std::vector<index_t> &ranks) {
  assert(X.nmodes == ranks.size() &&
         "Number of U sizes does not match X modes");
  // Sorting Xs
  std::vector<CSFTensor3> CSFTensors;
  CSFTensors.reserve(X.nmodes);
  std::vector<DenseMatrix> factor_matrices;
  factor_matrices.reserve(X.nmodes);
  for (unsigned mode = 0; mode < X.nmodes; ++mode) {
    CSFTensors.emplace_back(X, mode);
    CSFTensors[mode].print();
    factor_matrices.emplace_back(X.mode_sizes[mode], ranks[mode]);
    factor_matrices[mode].print();
    thrust::device_vector<value_t> out_values;
    call_contraction_kernel(CSFTensors[0], factor_matrices[1], out_values, 1);
  }
}