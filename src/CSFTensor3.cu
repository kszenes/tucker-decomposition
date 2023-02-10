#include "CSFTensor3.cuh"

template <typename T>
void copy_thrust(
    thrust::device_vector<T> &dest, const thrust::device_vector<T> &src
) {
  dest.resize(src.size());
  thrust::copy(src.begin(), src.end(), dest.begin());
}

template <typename T, size_t... Indices>
auto gen_tuple_impl(std::vector<T>& v, std::index_sequence<Indices...> ) {
    return thrust::make_tuple(std::begin(v[Indices])...);
}

template <size_t N, typename T>
auto gen_begin_tuple(std::vector<T>& v) {
    assert(std::size(v) >= N);
    return gen_tuple_impl(v, std::make_index_sequence<N>{});
}

CSFTensor3::CSFTensor3(const COOTensor3 &coo_tensor, const unsigned &mode)
:  nmodes(coo_tensor.d_modes.size()), nnz(coo_tensor.d_values.size()),
   shape(coo_tensor.shape)
{
  GPUTimer timer;
  timer.start();

  mode_permutation.resize(nmodes);
  std::iota(mode_permutation.begin(), mode_permutation.end(), 0);
  // === Optimal ordering based on extents ===
  fmt::print("Using OPTIMAL CSF mode ordering!\n");
  std::sort(
    mode_permutation.begin(),
    mode_permutation.end(),
    [&](const auto a, const auto b){
      if (a == mode) {
        return true;
      } else if (b == mode) {
        return false;
      } else {
        return coo_tensor.shape[a] <= coo_tensor.shape[b];
      }
      });

  //  === Matching ParTI ordering ===
  // fmt::print("Using PARTI CSF mode ordering!\n");
  // for(size_t i = 0; i < nmodes; ++i) {
  //     if(i < mode) {
  //         mode_permutation[nmodes - i - 1] = i;
  //     } else if(i != mode) {
  //         mode_permutation[nmodes - i] = i;
  //     }
  // }
  // mode_permutation[0] = mode;

  std::transform(
    mode_permutation.begin(),
    mode_permutation.end(),
    shape.begin(),
    [&](const auto& i){ return coo_tensor.shape[i]; }
  );
  fmt::print("Permutation = {}\n", mode_permutation);
  
  fptr.resize(nmodes);
  fidx.resize(nmodes);
  // Copy tmp coo_modes for csf construction
  std::vector<thrust::device_vector<index_t>> coo_modes(nmodes);
  for (size_t i = 0; i < nmodes; ++i) {
    copy_thrust(coo_modes[i], coo_tensor.d_modes[mode_permutation[i]]);
  }

  // Copy values
  copy_thrust(d_values, coo_tensor.d_values);

  buildCSFTensor3(coo_modes);
  auto time = timer.seconds();
  fmt::print("Constructed a CSF tensor in {} [s]\n", time);
}

void CSFTensor3::buildCSFTensor3(
    std::vector<thrust::device_vector<index_t>> &coo_modes
) {
  auto nmodes = coo_modes.size();

  if (nmodes == 4) {
    auto zip_it_4d = thrust::make_zip_iterator(gen_begin_tuple<4>(coo_modes));

    // Sort along 0th mode
    thrust::sort_by_key(
        zip_it_4d, zip_it_4d + nnz, std::begin(d_values)
    );

  } else if (nmodes == 3) {
    auto zip_it_3d = thrust::make_zip_iterator(gen_begin_tuple<3>(coo_modes));

    // Sort along 0th mode
    thrust::sort_by_key(
        zip_it_3d, zip_it_3d + nnz, std::begin(d_values)
    );

  } else {
    std::runtime_error("Tucker only supports order 3 and 4 tensors.");
  }

  // Set first fiber idx
  copy_thrust(fidx.back(), coo_modes.back());

  if (nmodes == 4) {
    auto zip3d_in = thrust::make_zip_iterator(gen_begin_tuple<3>(coo_modes));

    // fmt::print("First compression:\n{}\n{}\n", coo_modes[0], coo_modes[1]);
    // First compression
    fptr[3].resize(coo_modes[0].size());
    thrust::sequence(fptr[3].begin(), fptr[3].end(), 0);
    index_t largest_index = fptr[3].back() + 1;
    auto [first, second] = thrust::unique_by_key(
        zip3d_in, zip3d_in + coo_modes[0].size(), fptr[3].begin()
    );
    auto num_fibers = thrust::distance(fptr[3].begin(), second);
    *second = largest_index;
    fptr[3].resize(num_fibers + 1);
    coo_modes[0].resize(num_fibers);
    coo_modes[1].resize(num_fibers);
    coo_modes[2].resize(num_fibers);
    copy_thrust(fidx[2], coo_modes[2]);
  }

  auto zip2d_in = thrust::make_zip_iterator(gen_begin_tuple<2>(coo_modes));

  // fmt::print("First compression:\n{}\n{}\n", coo_modes[0], coo_modes[1]);
  // First compression
  fptr[2].resize(coo_modes[0].size());
  thrust::sequence(fptr[2].begin(), fptr[2].end(), 0);
  index_t largest_index = fptr[2].back() + 2;
  auto [first, second] = thrust::unique_by_key(
      zip2d_in, zip2d_in + coo_modes[0].size(), fptr[2].begin()
  );
  auto num_fibers = thrust::distance(fptr[2].begin(), second);
  *second = largest_index;
  fptr[2].resize(num_fibers + 1);
  coo_modes[0].resize(num_fibers);
  coo_modes[1].resize(num_fibers);
  // TODO: resizing coo_modes is not strictly necessary
  //       -> can modify size of fidx[1] directly
  copy_thrust(fidx[1], coo_modes[1]);

  // fmt::print("Second compression:\n{}\n\n", coo_modes[0]);
  // Second compression
  fptr[1].resize(coo_modes[0].size());
  thrust::sequence(fptr[1].begin(), fptr[1].end(), 0);
  largest_index = fptr[1].back() + 1;
  auto ret = thrust::unique_by_key(
      coo_modes[0].begin(), coo_modes[0].end(), fptr[1].begin()
  );
  num_fibers = thrust::distance(fptr[1].begin(), ret.second);
  *ret.second = largest_index;
  coo_modes[0].resize(num_fibers);
  fptr[1].resize(num_fibers + 1);
  copy_thrust(fidx[0], coo_modes[0]);

  thrust::host_vector<index_t> h_fptr(2, 0);
  h_fptr.back() = fidx.front().size();
  fptr.front() = h_fptr;
}

void CSFTensor3::print() const {
  fmt::print("\nCSF Tensor:\n  Modes: {}\n  Non Zeros: {}\n", nmodes, nnz);
  fmt::print("  shape = {}\n", shape);
  fmt::print(
      "  permutation = {}\n\n  == Values ==\n  values = {}\n",
      mode_permutation, d_values
  );
  fmt::print("\n  == Indices ==\n");
  for (size_t i = 0; i < fidx.size(); ++i) {
    fmt::print("  fidx[{}] = {}\n", i, fidx[i]);
  }
  fmt::print("\n  == Pointers ==\n");
  for (size_t i = 0; i < fptr.size(); ++i) {
    fmt::print("  fptr[{}] = {}\n", i, fptr[i]);
  }
  fmt::print("\n");
}

value_t CSFTensor3::sqnorm() const {
  return thrust::transform_reduce(
    d_values.begin(), d_values.end(),
    thrust::square<value_t>(), 0.0, thrust::plus<value_t>());
}
