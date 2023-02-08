
#include "ttm.cuh"

thrust::device_vector<value_t>
contract_first_mode(const CSFTensor3 &tensor, const DenseMatrix &matrix) {
  size_t out_num_chunks = tensor.fidx[1].size();
  size_t out_chunk_size = matrix.ncols;
  thrust::device_vector<value_t> out_values(out_num_chunks * out_chunk_size);
  out_values.resize(out_num_chunks * out_chunk_size);

  const index_t max_nblocks = 32768;
  const index_t max_nthreads_per_block = 256;
  index_t max_nthreadsy = 32;

  index_t nthreadsx = 1;
  index_t nthreadsy = 1;
  index_t all_nblocks = 0;
  index_t nblocks = 0;
  index_t shmen_size = 0;

  if (matrix.ncols <= max_nthreadsy)
    nthreadsx = matrix.ncols;
  else
    nthreadsx = max_nthreadsy;
  nthreadsy = max_nthreads_per_block / nthreadsx;

  if (out_num_chunks < nthreadsy) {
    nthreadsy = out_num_chunks;
    nblocks = 1;
  } else {
    all_nblocks = (out_num_chunks + nthreadsy - 1) / nthreadsy;
    if (all_nblocks < max_nblocks) {
      nblocks = all_nblocks;
    } else {
      nblocks = max_nblocks;
    }
  }
  shmen_size = nthreadsx * nthreadsy * sizeof(value_t);
  assert(shmen_size >= nthreadsx * nthreadsy * sizeof(value_t));
  dim3 dimBlock(nthreadsx, nthreadsy);
  // std::cout << "SHMEM size: " << (shmen_size / sizeof(value_t)) << " ("
  //           << shmen_size << " bytes)\n";
  // std::cout << "all_nblocks: " << all_nblocks << "; nthreadsx: " << nthreadsx
  //           << "; nthreadsy: " << nthreadsy << '\n';

  // std::cout << "U_rows: " << matrix.nrows << ": U_cols: " << matrix.ncols
  //           << '\n';
  // std::cout << "out_num_chunks = " << out_num_chunks
  //           << "; out_chunk_size: " << out_chunk_size << '\n';
  GPUTimer timer;
  timer.start();

  spt_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, shmen_size>>>(
      thrust::raw_pointer_cast(out_values.data()), out_chunk_size, out_num_chunks,
      thrust::raw_pointer_cast(tensor.d_values.data()), thrust::raw_pointer_cast(tensor.fidx[2].data()),
      thrust::raw_pointer_cast(tensor.fptr[1].data()), thrust::raw_pointer_cast(matrix.d_values.data()),
      matrix.nrows, matrix.ncols, matrix.ncols
  );
  auto time = timer.seconds();
  fmt::print(
      "First_contraction<<<{}, ({}, {}), {}>>>:\n\t  executed in {} [s]\n", nblocks,
      dimBlock.x, dimBlock.y, shmen_size, time
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  return out_values;
}

thrust::device_vector<value_t> contract_second_mode(
    const CSFTensor3 &tensor, const DenseMatrix &matrix,
    const thrust::device_vector<value_t> &in_values, const size_t subchunk_size
) {
  size_t out_num_chunks = tensor.fidx[0].size();
  size_t out_chunk_size = matrix.ncols * subchunk_size;
  // fmt::print("\nnum_chunks = {}; chunk_size = {}; matrix.ncols = {}; subchunk_size = {}\n", out_num_chunks, out_chunk_size, matrix.ncols, subchunk_size);
  thrust::device_vector<value_t> out_values(out_num_chunks * out_chunk_size);
  auto threads = dim3(matrix.ncols, 1024 / matrix.ncols);
  auto grid = dim3(
    out_num_chunks
    // (subchunk_size + threads.x - 1) / threads.x,
    // (out_num_chunks + threads.y - 1) / threads.y
  );

  GPUTimer timer;
  timer.start();
  // fmt::print("U = {}\n", matrix.d_values);
      
  ttm_semisparse_kernel<<<grid, threads>>>(
      thrust::raw_pointer_cast(tensor.fptr[0].data()), thrust::raw_pointer_cast(tensor.fidx[1].data()),
      matrix.nrows, matrix.ncols, out_num_chunks, out_chunk_size, subchunk_size,
      thrust::raw_pointer_cast(out_values.data()), thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(matrix.d_values.data())
  );
  auto time = timer.seconds();
  fmt::print(
      "Second_contraction<<<({}, {}),({}, {})>>>\n\t  executed in {} [s]\n", grid.x,
      grid.y, threads.x, threads.y, time
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  return out_values;
}

thrust::device_vector<value_t> contract_last_mode(
    const CSFTensor3 &tensor, const std::vector<DenseMatrix> &matrices,
    const thrust::device_vector<value_t> &in_values, const size_t subchunk_size
) {
  auto mode = tensor.cyclic_permutation.front();
  const auto& matrix = matrices[mode];

  size_t out_num_chunks = 1;
  size_t out_chunk_size = matrix.ncols * subchunk_size;
  // fmt::print("\nnum_chunks = {}; chunk_size = {}; matrix.ncols = {}; subchunk_size = {}\n", out_num_chunks, out_chunk_size, matrix.ncols, subchunk_size);
  thrust::device_vector<value_t> out_values(out_num_chunks * out_chunk_size);
  auto threads = dim3(matrix.ncols, 1024 / matrix.ncols);
  auto grid = dim3(
    out_num_chunks
    // (subchunk_size + threads.x - 1) / threads.x,
    // (out_num_chunks + threads.y - 1) / threads.y
  );

  GPUTimer timer;
  timer.start();

  // Only one fiber left
  thrust::host_vector<index_t> h_fptr(2, 0);
  h_fptr.back() = tensor.fidx[0].size();
  // fmt::print("\nh_fptr = {}\n", h_fptr);
  auto d_fptr = thrust::device_vector<index_t>(h_fptr);
  // thrust::sequence(d_fptr.begin(), d_fptr.end());
      
  ttm_semisparse_kernel<<<grid, threads>>>(
      thrust::raw_pointer_cast(d_fptr.data()), thrust::raw_pointer_cast(tensor.fidx[0].data()),
      matrix.nrows, matrix.ncols, out_num_chunks, out_chunk_size, subchunk_size,
      thrust::raw_pointer_cast(out_values.data()), thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(matrix.d_values.data())
  );
  auto time = timer.seconds();
  fmt::print(
      "Last_contraction<<<({},({}, {})>>>:\n  executed in {} [s]\n", grid.x,
      threads.x, threads.y, time
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  return out_values;
}

thrust::host_vector<value_t> ttm_chain(
    const CSFTensor3 &tensor, std::vector<DenseMatrix> &factor_matrices
) {
  thrust::host_vector<index_t> last_mode(tensor.fidx[0]);

  // Contract First Mode
  auto out_first = contract_first_mode(
      tensor, factor_matrices[tensor.cyclic_permutation.back()]
  );
  // fmt::print("out_first = {}\n", out_first);
  auto subchunk_size = factor_matrices[tensor.cyclic_permutation.back()].ncols;
  // Contract Second Mode
  auto out_second = contract_second_mode(
      tensor,
      factor_matrices[tensor.cyclic_permutation[1]],
      out_first, subchunk_size
  );

  subchunk_size *= factor_matrices[tensor.cyclic_permutation[1]].ncols;
  auto &U_update = factor_matrices[tensor.cyclic_permutation.front()];

  thrust::host_vector<value_t> ssp_matrix(out_second);
  return ssp_matrix;
}