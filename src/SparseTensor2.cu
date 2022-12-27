#include "SparseTensor2.cuh"

SparseTensor2::SparseTensor2(const std::vector<index_t> &other_sizes,
                             const index_t contracted_mode,
                             const index_t new_chunk_size) {
  assert((mode_sizes.size() == 0) && "Tensor already initialised");
  auto tmp_prod = 1;
  for (unsigned i = 0; i < other_sizes.size() - 1; ++i) {
    mode_sizes.emplace_back(other_sizes[i]);
    tmp_prod *= other_sizes[i];
  }
  nmodes = mode_sizes.size();
  nnz = tmp_prod * chunk_size;
  chunk_size = new_chunk_size;
}

SparseTensor2::SparseTensor2(const SparseTensor2 &other)
    : nmodes(other.nmodes), nnz(other.nnz), chunk_size(other.chunk_size),
      mode_sizes(other.mode_sizes), d_modes(other.d_modes),
      h_modes(other.h_modes), d_values(other.d_values),
      h_values(other.h_values) {
  std::cout << "Copying tensor\n";
  d_zip_it = thrust::make_zip_iterator(
      thrust::make_tuple(d_modes[0].begin(), d_modes[1].begin()));
}

void SparseTensor2::to_device() {
  d_values = h_values;
  if (d_modes.size() != nmodes)
    d_modes.resize(nmodes);
  for (unsigned i = 0; i < nmodes; ++i)
    d_modes[i] = h_modes[i];
}

void SparseTensor2::to_host() {
  h_values = d_values;
  if (h_modes.size() != nmodes)
    h_modes.resize(nmodes);
  for (unsigned i = 0; i < nmodes; ++i)
    h_modes[i] = d_modes[i];
}

void SparseTensor2::print() {
  to_host();
  std::string output = "Tensor modes: ";
  for (const auto &e : mode_sizes)
    output += std::to_string(e) + " x ";
  output += ":\n\n";
  for (unsigned i = 0; i < nnz; ++i) {
    for (const auto &e : h_modes)
      output += std::to_string(e[i]) + '\t';
    output += "[ ";
    for (unsigned chunk = 0; chunk < chunk_size; ++chunk)
      output += std::to_string(h_values[i * chunk_size + chunk]) + ' ';
    output += "]\n";
  }
  std::cout << output << '\n';
}

void SparseTensor2::sort_mode(const index_t mode) {
  assert((mode < nmodes) && "Mode out of bounds in sort");
  if (mode == 0) {
    thrust::sort_by_key(
        d_zip_it, d_zip_it + nnz, d_values.begin(),
        [] __host__ __device__(const IndexTuple &a, const IndexTuple &b) {
          return (thrust::get<0>(a) < thrust::get<0>(b)) ||
                 (thrust::get<0>(a) == thrust::get<0>(b) &&
                  thrust::get<1>(a) < thrust::get<1>(b));
        });
  } else if (mode == 1) {
    thrust::sort_by_key(
        d_zip_it, d_zip_it + nnz, d_values.begin(),
        [] __host__ __device__(const IndexTuple &a, const IndexTuple &b) {
          return (thrust::get<1>(a) < thrust::get<1>(b)) ||
                 (thrust::get<1>(a) == thrust::get<1>(b) &&
                  thrust::get<0>(a) < thrust::get<0>(b));
        });
  }
}