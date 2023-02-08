#include "COOTensor3.cuh"

COOTensor3::COOTensor3(const std::string &filename, const bool starts_at_zero)
    : chunk_size(1) {
  GPUTimer timer;
  timer.start();
  fmt::print("Reading sparse tensor from: {}\n", filename);
  std::ifstream ifile(filename);
  unsigned num_nz = 0, num_modes = 0;
  index_t index(0);
  value_t value(0.0);

  if (!ifile.is_open()) {
    fmt::print(stderr, "Failed to open {}\n", filename);
    exit(1);
  }

  ifile >> num_nz >> num_modes;
  nnz = num_nz;
  nmodes = num_modes;
  fmt::print("Mode {} Tensor with {} non_zero elements\n", nmodes, nnz);

  h_modes.resize(nmodes);
  for (auto &e : h_modes)
    e.resize(nnz);

  shape.resize(nmodes);
  mode_permutation.resize(nmodes);
  for (unsigned i = 0; i < nmodes; ++i) {
    if (!(ifile >> index)) {
      fmt::print(stderr, "Error occured during reading of tensor");
      exit(1);
    };
    shape[i] = index;
    mode_permutation[i] = i;
  }

  fmt::print("With modes: {}\n", shape);

  h_values.resize(nnz);
  for (unsigned i = 0; i < nnz; ++i) {
    for (unsigned mode = 0; mode < nmodes; ++mode) {
      if (!(ifile >> index)) {
        fmt::print(stderr, "Error occured during reading of tensor");
        exit(1);
      };
      index -= starts_at_zero ? 1 : 0;
      h_modes[mode][i] = index;
    }
    if (!(ifile >> value)) {
      fmt::print(stderr, "Error occured during reading of tensor");
      exit(1);
    }
    h_values[i] = value;
  }

  // Copy to device
  to_device();

  ifile.close();
  auto time = timer.seconds();
  fmt::print("Loaded COO tensor in {} [s]\n", time);
}

COOTensor3::COOTensor3(
    const std::vector<index_t> &other_sizes, const index_t contracted_mode,
    const index_t new_chunk_size
) {
  assert((shape.size() == 0) && "Tensor already initialised");
  auto tmp_prod = 1;
  for (unsigned i = 0; i < other_sizes.size(); ++i) {
    if (i == contracted_mode) {
      continue;
    }
    shape.emplace_back(other_sizes[i]);
    tmp_prod *= other_sizes[i];
  }
  nmodes = shape.size();
  nnz = tmp_prod * chunk_size;
  chunk_size = new_chunk_size;
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
  std::string output = "COO Tensor: \n";
  output += "nnz = " + std::to_string(nnz) + "; shape = (";
  for (const auto &e : shape)
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