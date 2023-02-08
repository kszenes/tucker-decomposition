#include "host_contractions.hpp"

thrust::host_vector<value_t> contract(
  const thrust::host_vector<value_t>& in,
  const thrust::host_vector<value_t>& Uvals,
  const std::vector<index_t>& out_shape,
  const std::vector<index_t>& in_shape,
  const index_t Ucols,
  const index_t contractedMode
) {
  auto outSize = std::accumulate(
    out_shape.begin(), out_shape.end(), 1, std::multiplies<index_t>{});
  fmt::print("outSize = {}\n", outSize);
  fmt::print("out_shape = {}\n", out_shape);
  fmt::print("in_shape = {}\n", in_shape);

  thrust::host_vector<value_t> out(outSize);
  // Perform contraction
  if (contractedMode == 0) {
    for (index_t k = 0; k < in_shape[0]; ++k) {
      for (index_t mode1 = 0; mode1 < in_shape[1]; ++mode1) {
        for (index_t mode2 = 0; mode2 < in_shape[2]; ++mode2) {
          for (index_t r = 0; r < Ucols; ++r) {
            out[
              r * out_shape[1] * out_shape[2] +
              mode1 * out_shape[2] + mode2
            ] += in[k * in_shape[1] * in_shape[2] +
                   mode1 * in_shape[2] + mode2] * Uvals[k * Ucols + r];
          }
        }
      }
    }
  } else if (contractedMode == 1) {
    for (index_t mode0 = 0; mode0 < in_shape[0]; ++mode0) {
      for (index_t k = 0; k < in_shape[1]; ++k) {
        for (index_t mode2 = 0; mode2 < in_shape[2]; ++mode2) {
          for (index_t r = 0; r < Ucols; ++r) {
            out[
              mode0 * out_shape[1] * out_shape[2] +
              r * out_shape[2] + mode2
            ] += in[mode0 * in_shape[1] * in_shape[2] +
                   k * in_shape[2] + mode2] * Uvals[k * Ucols + r];
          }
        }
      }
    }
  } else if (contractedMode == 2) {
    for (index_t mode0 = 0; mode0 < in_shape[0]; ++mode0) {
      for (index_t mode1 = 0; mode1 < in_shape[1]; ++mode1) {
        for (index_t k = 0; k < in_shape[2]; ++k) {
          for (index_t r = 0; r < Ucols; ++r) {
            out[
              mode0 * out_shape[1] * out_shape[2] +
              mode1 * out_shape[2] + r
            ] += in[mode0 * in_shape[1] * in_shape[2] +
                   mode1 * in_shape[2] + k] * Uvals[k * Ucols + r];
          }
        }
      }
    }
  }
  return out;
}

thrust::host_vector<value_t> host_first_contraction(
  const COOTensor3& X, const DenseMatrix& U, const index_t contractedMode
) {
  assert(X.shape[mode] == U.nrows && "COOTensor and Dense Matrix size mismatch");

  auto denseSize = std::accumulate(
    X.shape.begin(), X.shape.end(), 1, std::multiplies<index_t>{});

  std::vector<index_t> out_shape;
  for (index_t i = 0; i < X.nmodes; ++i) {
    if (i != contractedMode) {
      out_shape.push_back(X.shape[i]);
    } else {
      out_shape.push_back(U.ncols);
    }
  }

  thrust::host_vector<value_t> densifiedTensor(denseSize);

  // Densify sparse input
  index_t i = 0;
  for (index_t i = 0; i < X.nnz; ++i) {
        auto mode0 = X.h_modes[0][i];
        auto mode1 = X.h_modes[1][i];
        auto mode2 = X.h_modes[2][i];
        auto value = X.h_values[i];
        densifiedTensor[
          mode0 * X.shape[1] * X.shape[2] +
          mode1 * X.shape[2] +
          mode2
        ] = value;
  }

  return contract(
    densifiedTensor, U.h_values,
    out_shape, X.shape, U.ncols, contractedMode);

}


thrust::host_vector<value_t> host_second_contraction(
  const CSFTensor3& csf,
  const thrust::host_vector<value_t>& denseTensor,
  const std::vector<DenseMatrix>& Us
) {
  auto contractedMode = csf.mode_permutation[1];

  std::vector<index_t> in_shape(3);
  in_shape[csf.mode_permutation[0]] = csf.shape[0];
  in_shape[csf.mode_permutation[1]] = csf.shape[1];
  in_shape[csf.mode_permutation[2]] = Us[csf.mode_permutation[2]].ncols;

  std::vector<index_t> out_shape(3);
  out_shape[csf.mode_permutation[0]] = csf.shape[0];
  out_shape[csf.mode_permutation[1]] = Us[csf.mode_permutation[1]].ncols;
  out_shape[csf.mode_permutation[2]] = Us[csf.mode_permutation[2]].ncols;

  return contract(
    denseTensor, Us[csf.mode_permutation[1]].h_values,
    out_shape, in_shape, Us[csf.mode_permutation[1]].ncols, contractedMode
  );

}

thrust::host_vector<value_t> host_third_contraction(
  const CSFTensor3& csf,
  const thrust::host_vector<value_t>& denseTensor,
  const std::vector<DenseMatrix>& Us
) {
  auto contractedMode = csf.mode_permutation[0];

  std::vector<index_t> in_shape(3);
  in_shape[csf.mode_permutation[0]] = csf.shape[0];
  in_shape[csf.mode_permutation[1]] = Us[csf.mode_permutation[1]].ncols;
  in_shape[csf.mode_permutation[2]] = Us[csf.mode_permutation[2]].ncols;

  std::vector<index_t> out_shape(3);
  out_shape[csf.mode_permutation[0]] = Us[csf.mode_permutation[0]].ncols;
  out_shape[csf.mode_permutation[1]] = Us[csf.mode_permutation[1]].ncols;
  out_shape[csf.mode_permutation[2]] = Us[csf.mode_permutation[2]].ncols;

  return contract(
    denseTensor, Us[csf.mode_permutation[0]].h_values,
    out_shape, in_shape, Us[csf.mode_permutation[0]].ncols, contractedMode
  );

}