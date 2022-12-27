#include "COOTensor3.cuh"

COOTensor3::COOTensor3(const std::string &filename,
                             const bool starts_at_zero)
    : chunk_size(1) {
  std::cout << "Reading sparse tensor from: " << filename << '\n';
  std::ifstream ifile(filename);
  unsigned num_nz = 0, num_modes = 0;
  index_t index(0);
  value_t value(0.0);

  if (!ifile.is_open()) {
    std::cerr << "Failed to open " << filename << '\n';
    exit(1);
  }

  ifile >> num_nz >> num_modes;
  nnz = num_nz;
  nmodes = num_modes;
  std::cout << "Mode " << nmodes << " Tensor with " << nnz
            << " non-zero elements\n";

  h_modes.resize(nmodes);
  for (auto &e : h_modes)
    e.resize(nnz);

  mode_sizes.resize(nmodes);
  mode_permutation.resize(nmodes);
  for (unsigned i = 0; i < nmodes; ++i) {
    if (!(ifile >> index)) {
      std::cerr << "Error occured during reading tensor\n";
      exit(1);
    };
    mode_sizes[i] = index;
    mode_permutation[i] = i;
  }

  std::cout << "With modes: ";
  for (const auto &e : mode_sizes)
    std::cout << e << ' ';
  std::cout << '\n';

  h_values.resize(nnz);
  for (unsigned i = 0; i < nnz; ++i) {
    for (unsigned mode = 0; mode < nmodes; ++mode) {
      if (!(ifile >> index)) {
        std::cerr << "Error occured during reading tensor\n";
        exit(1);
      };
      index -= starts_at_zero ? 1 : 0;
      h_modes[mode][i] = index;
    }
    if (!(ifile >> value)) {
      std::cerr << "Error occured during reading of tensor\n";
      exit(1);
    }
    h_values[i] = value;
  }

  // Copy to device
  to_device();
  // Create zip iterator
  d_zip_it = thrust::make_zip_iterator(thrust::make_tuple(
      d_modes[0].begin(), d_modes[1].begin(), d_modes[2].begin()));

  ifile.close();
}

COOTensor3::COOTensor3(const std::vector<index_t> &other_sizes,
                             const index_t contracted_mode,
                             const index_t new_chunk_size) {
  assert((mode_sizes.size() == 0) && "Tensor already initialised");
  auto tmp_prod = 1;
  for (unsigned i = 0; i < other_sizes.size(); ++i) {
    if (i == contracted_mode) {
      continue;
    }
    mode_sizes.emplace_back(other_sizes[i]);
    tmp_prod *= other_sizes[i];
  }
  nmodes = mode_sizes.size();
  nnz = tmp_prod * chunk_size;
  chunk_size = new_chunk_size;
}

COOTensor3::COOTensor3(const COOTensor3 &other, const std::vector<index_t>& permutation)
    : nmodes(other.nmodes), nnz(other.nnz), chunk_size(other.chunk_size),
      d_values(other.d_values),
      h_values(other.h_values) {
  mode_sizes.resize(nmodes);
  d_modes.resize(nmodes);
  h_modes.resize(nmodes);
  mode_permutation.resize(nmodes);
  for (unsigned i = 0; i < nmodes; ++i) {
    mode_sizes[permutation[i]] = other.mode_sizes[i];
    d_modes[permutation[i]] = other.d_modes[i];
    h_modes[permutation[i]] = other.h_modes[i];
    mode_permutation[permutation[i]] = other.mode_permutation[i];
  }
  std::cout << "Copying and permuting tensor\n";
  d_zip_it = thrust::make_zip_iterator(thrust::make_tuple(
      d_modes[0].begin(), d_modes[1].begin(), d_modes[2].begin()));
}

void COOTensor3::to_device() {
  d_values = h_values;
  if (d_modes.size() != nmodes)
    d_modes.resize(nmodes);
  for (unsigned i = 0; i < nmodes; ++i)
    d_modes[i] = h_modes[i];
}

void COOTensor3::to_host() {
  h_values = d_values;
  if (h_modes.size() != nmodes)
    h_modes.resize(nmodes);
  for (unsigned i = 0; i < nmodes; ++i)
    h_modes[i] = d_modes[i];
}

void COOTensor3::print() {
  to_host();
  std::string output = "Sparse Tensor: \n";
  output += "nnz = " + std::to_string(nnz) + "; shape = (";
  for (const auto &e : mode_sizes)
    output += std::to_string(e) + ", ";
  output += ")\n";
  output += "permutation: [";
  for (const auto &e : mode_permutation)
    output += std::to_string(e) + ", ";
  output += "]\n";
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

void COOTensor3::sort_mode(const index_t mode) {
  assert((mode < nmodes) && "Mode out of bounds in sort");
  if (mode == 0) {
    thrust::sort_by_key(
        d_zip_it, d_zip_it + nnz, d_values.begin(),
        [] __host__ __device__(const IndexTuple &a, const IndexTuple &b) {
          return (thrust::get<0>(a) < thrust::get<0>(b)) ||
                 (thrust::get<0>(a) == thrust::get<0>(b) &&
                  thrust::get<1>(a) < thrust::get<1>(b)) ||
                 (thrust::get<0>(a) == thrust::get<0>(b) &&
                  thrust::get<1>(a) == thrust::get<1>(b) &&
                  thrust::get<2>(a) < thrust::get<2>(b));
        });
  } else if (mode == 1) {
    thrust::sort_by_key(
        d_zip_it, d_zip_it + nnz, d_values.begin(),
        [] __host__ __device__(const IndexTuple &a, const IndexTuple &b) {
          return (thrust::get<1>(a) < thrust::get<1>(b)) ||
                 (thrust::get<1>(a) == thrust::get<1>(b) &&
                  thrust::get<0>(a) < thrust::get<0>(b)) ||
                 (thrust::get<1>(a) == thrust::get<1>(b) &&
                  thrust::get<0>(a) == thrust::get<0>(b) &&
                  thrust::get<2>(a) < thrust::get<2>(b));
        });
  } else if (mode == 2) {
    thrust::sort_by_key(
        d_zip_it, d_zip_it + nnz, d_values.begin(),
        [] __host__ __device__(const IndexTuple &a, const IndexTuple &b) {
          return (thrust::get<2>(a) < thrust::get<2>(b)) ||
                 (thrust::get<2>(a) == thrust::get<2>(b) &&
                  thrust::get<1>(a) < thrust::get<1>(b)) ||
                 (thrust::get<2>(a) == thrust::get<2>(b) &&
                  thrust::get<1>(a) == thrust::get<1>(b) &&
                  thrust::get<0>(a) < thrust::get<0>(b));
        });
  }
}