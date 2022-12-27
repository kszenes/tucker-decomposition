#include "CSFTensor3.cuh"

template <typename T>
void copy_thrust(
    thrust::device_vector<T> &dest, const thrust::device_vector<T> &src
) {
  dest.resize(src.size());
  thrust::copy(src.begin(), src.end(), dest.begin());
}

CSFTensor3::CSFTensor3(const COOTensor3& coo_tensor, const int& mode) {
  // Set up parameters and vector sizes
  nmodes = coo_tensor.d_modes.size();
  nnz = coo_tensor.d_values.size();
  shape.resize(nmodes);
  cyclic_permutation.resize(nmodes);
  for (size_t i = 0; i < nmodes; ++i) {
    cyclic_permutation[i] = (i + mode) % nmodes;
    shape[i] = coo_tensor.mode_sizes[cyclic_permutation[i]];
  }

  fptr.resize(nmodes - 1);
  fidx.resize(nmodes);
  // Copy tmp coo_modes for csf construction
  std::vector<thrust::device_vector<index_t>> coo_modes(nmodes);
  for (size_t i = 0; i < nmodes; ++i) {
    copy_thrust(coo_modes[i], coo_tensor.d_modes[cyclic_permutation[i]]);
  }

  // Copy values
  copy_thrust(d_values, coo_tensor.d_values);

  buildCSFTensor3(coo_modes);
}

void CSFTensor3::buildCSFTensor3(
    std::vector<thrust::device_vector<index_t>> &coo_modes
) {
  auto nmodes = coo_modes.size();

  auto zip_it_3d = thrust::make_zip_iterator(thrust::make_tuple(
      coo_modes[0].begin(), coo_modes[1].begin(), coo_modes[2].begin()
  ));

  // Sort along 0th mode
  thrust::sort_by_key(
      zip_it_3d, zip_it_3d + nnz, d_values.begin(),
      [] __host__ __device__(const IndexTuple3D &a, const IndexTuple3D &b) {
        return (thrust::get<0>(a) < thrust::get<0>(b)) ||
               (thrust::get<0>(a) == thrust::get<0>(b) &&
                thrust::get<1>(a) < thrust::get<1>(b)) ||
               (thrust::get<0>(a) == thrust::get<0>(b) &&
                thrust::get<1>(a) == thrust::get<1>(b) &&
                thrust::get<2>(a) < thrust::get<2>(b));
      }
  );
  // Set first fiber idx
  copy_thrust(fidx[2], coo_modes[2]);

  auto zip2d_in = thrust::make_zip_iterator(
      thrust::make_tuple(coo_modes[0].begin(), coo_modes[1].begin())
  );

  // fmt::print("First compression:\n{}\n{}\n", coo_modes[0], coo_modes[1]);
  // First compression
  fptr[1].resize(coo_modes[0].size());
  thrust::sequence(fptr[1].begin(), fptr[1].end(), 0);
  index_t largest_index = *(fptr[1].end() - 1) + 1; 
  auto ret = thrust::unique_by_key(
      zip2d_in, zip2d_in + nnz, fptr[1].begin(),
      [] __device__(const IndexTuple2D &a, const IndexTuple2D &b) {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b);
      }
  );
  auto num_fibers = thrust::distance(fptr[1].begin(), ret.second);
  *ret.second = largest_index;
  fptr[1].resize(num_fibers + 1);
  coo_modes[0].resize(num_fibers);
  coo_modes[1].resize(num_fibers);
  // TODO: resizing coo_modes is not strictly necessary
  //       -> can modify size of fidx[1] directly
  copy_thrust(fidx[1], coo_modes[1]);

  // fmt::print("Second compression:\n{}\n\n", coo_modes[0]);
  // Second compression
  fptr[0].resize(coo_modes[0].size());
  thrust::sequence(fptr[0].begin(), fptr[0].end(), 0);
  largest_index = *(fptr[0].end() - 1) + 1;
  ret = thrust::unique_by_key(
      coo_modes[0].begin(), coo_modes[0].end(), fptr[0].begin()
  );
  num_fibers = thrust::distance(fptr[0].begin(), ret.second);
  *ret.second = largest_index;
  coo_modes[0].resize(num_fibers);
  fptr[0].resize(num_fibers + 1);
  copy_thrust(fidx[0], coo_modes[0]);
}

void CSFTensor3::print() const {
  fmt::print("\nCSF Tensor:\n  Modes: {}\n  Non Zeros: {}\n", nmodes, nnz);
  fmt::print("  shape = {}\n", shape);
  fmt::print("  permutation = {}\n\n  == Indices ==\n", cyclic_permutation);
  for (size_t i = 0; i < fidx.size(); ++i) {
    fmt::print("  fidx[{}] = {}\n", i, fidx[i]);
  }
  fmt::print("\n  == Pointers ==\n");
  for (size_t i = 0; i < fptr.size(); ++i) {
    fmt::print("  fptr[{}] = {}\n", i, fptr[i]);
  }

}
