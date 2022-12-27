#include "ttm.cuh"

__global__ void spt_TTMRankRBNnzKernelSM(
    value_t *Y_val, index_t Y_stride, index_t Y_nnz,
    const value_t *__restrict__ X_val, const index_t *__restrict__ X_inds_m,
    const index_t *__restrict__ fiberidx_val, const value_t *__restrict__ U_val,
    index_t U_nrows, index_t U_ncols, index_t U_stride
) {
  extern __shared__ value_t mem_pool[];
  value_t *const Y_shr = (value_t *)mem_pool; // size U_ncols

  index_t num_loops_nnz = 1;
  index_t const nnz_per_loop = gridDim.x * blockDim.y;
  if (Y_nnz > nnz_per_loop) {
    num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
  }

  // Global indices of Y: Fiber = x and Inner fiber = r
  // Local indices: tidx and tidy
  const index_t tidx = threadIdx.x;
  const index_t tidy = threadIdx.y;
  index_t x;
  const index_t num_loops_r = U_ncols / blockDim.x;
  const index_t rest_loop = U_ncols - num_loops_r * blockDim.x;
  index_t r; // column idx of U

  for (index_t l = 0; l < num_loops_r;
       ++l) {                  // blockDim.x parallelised over cols(U)
    r = tidx + l * blockDim.x; // r: column idx of U
    for (index_t nl = 0; nl < num_loops_nnz; ++nl) { // Grid strided-pattern?
      x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

      Y_shr[tidy * blockDim.x + tidx] = 0;
      __syncthreads();

      if (x < Y_nnz) { // Why is this not above at line 348
        const index_t inz_begin = fiberidx_val[x];
        const index_t inz_end = fiberidx_val[x + 1];
        for (index_t i = inz_begin; i < inz_end; ++i) { // loop over a n-fiber
          const index_t row = X_inds_m[i];              // row of U
          // Loop over nnz in n-fiber of X and multiply with corresponding
          // U col elements and accumulate in single element of Y
          Y_shr[tidy * blockDim.x + tidx] +=
              X_val[i] * U_val[row * U_stride + r]; // Original
          // Y_shr[tidy*blockDim.x + tidx] += X_val[i] * U_val[r*U_stride +
          // row];
        }
        __syncthreads();

        Y_val[x * Y_stride + r] = Y_shr[tidy * blockDim.x + tidx];
        __syncthreads();
      }
    }
  }

  if (rest_loop > 0 && tidx < rest_loop) {
    r = tidx + num_loops_r * blockDim.x;

    for (index_t nl = 0; nl < num_loops_nnz; ++nl) {
      x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

      Y_shr[tidy * blockDim.x + tidx] = 0;
      __syncthreads();

      if (x < Y_nnz) {
        const index_t inz_begin = fiberidx_val[x];
        const index_t inz_end = fiberidx_val[x + 1];
        for (index_t i = inz_begin; i < inz_end; ++i) {
          const index_t row = X_inds_m[i];
          Y_shr[tidy * blockDim.x + tidx] +=
              X_val[i] * U_val[row * U_stride + r]; // Original
          // Y_shr[tidy*blockDim.x + tidx] += X_val[i] * U_val[r*U_stride +
          // row];
        }
        __syncthreads();

        Y_val[x * Y_stride + r] = Y_shr[tidy * blockDim.x + tidx];
        __syncthreads();
      }
    }
  }
}

// TODO: does not do grid-strided pattern for large sizes
__global__ void ttm_semisparse_kernel(
    const index_t *__restrict__ fiber_ptr,
    const index_t *__restrict__ X_indices_m, index_t nrows, index_t ncols,
    index_t Y_chunk_size, index_t X_chunk_size, value_t *__restrict__ Y_values,
    const value_t *__restrict__ X_values, const value_t *__restrict__ U_values
) {

  int tix = threadIdx.x + blockIdx.x * blockDim.x; // over chunk x
  int tiy = threadIdx.y + blockIdx.y * blockDim.y; // over nnz
  // TODO: check stepping out of bounds
  int fiber_begin = fiber_ptr[tiy];
  int fiber_end = fiber_ptr[tiy + 1];
  for (int fiber_idx = fiber_begin; fiber_idx < fiber_end; ++fiber_idx) {
    int row = X_indices_m[fiber_idx];
    int chunk_start = fiber_idx * X_chunk_size;
    for (int col = 0; col < ncols; ++col) {
      Y_values[tiy * Y_chunk_size + tix * X_chunk_size + col] +=
          X_values[chunk_start + tix] * U_values[row * ncols + col];
    }
  }
}

void construct_fiberidx_and_outTensor(
    SparseTensor2 &dest, thrust::device_vector<index_t> &fiber_ptr,
    SparseTensor3 &ref, index_t mode
) {
  thrust::sequence(fiber_ptr.begin(), fiber_ptr.end(), 0);
  if (mode == 0) {
    auto ret = thrust::unique_by_key(
        ref.d_zip_it, ref.d_zip_it + ref.nnz, fiber_ptr.begin(),
        [] __device__(
            const thrust::tuple<index_t, index_t, index_t> &a,
            const thrust::tuple<index_t, index_t, index_t> &b
        ) {
          return thrust::get<1>(a) == thrust::get<1>(b) &&
                 thrust::get<2>(a) == thrust::get<2>(b);
        }
    );
    // for (const auto& e : fiber_ptr) std::cout << e << ' ';
    dest.nnz = *ret.second;
    dest.d_values = thrust::device_vector<value_t>(dest.nnz * dest.chunk_size);
    // dest.d_modes.push_back();
    dest.d_modes.push_back(ref.d_modes[1]);
    dest.d_modes.push_back(ref.d_modes[2]);
    dest.d_modes[0].resize(*ret.second);
    dest.d_modes[1].resize(*ret.second);
    auto original_size = fiber_ptr.size();
    fiber_ptr.resize(*ret.second + 1);
    fiber_ptr[fiber_ptr.size() - 1] = original_size;
  } else if (mode == 1) {
    auto ret = thrust::unique_by_key(
        ref.d_zip_it, ref.d_zip_it + ref.nnz, fiber_ptr.begin(),
        [] __device__(
            const thrust::tuple<index_t, index_t, index_t> &a,
            const thrust::tuple<index_t, index_t, index_t> &b
        ) {
          return thrust::get<0>(a) == thrust::get<0>(b) &&
                 thrust::get<2>(a) == thrust::get<2>(b);
        }
    );
    dest.nnz = *ret.second;
    dest.d_values = thrust::device_vector<value_t>(dest.nnz * dest.chunk_size);
    dest.d_modes.push_back(ref.d_modes[0]);
    // dest.d_modes.push_back();
    dest.d_modes.push_back(ref.d_modes[2]);
    dest.d_modes[0].resize(*ret.second);
    dest.d_modes[1].resize(*ret.second);
    auto original_size = fiber_ptr.size();
    fiber_ptr.resize(*ret.second + 1);
    fiber_ptr[fiber_ptr.size() - 1] = original_size;
  } else if (mode == 2) {
    auto ret = thrust::unique_by_key(
        ref.d_zip_it, ref.d_zip_it + ref.nnz, fiber_ptr.begin(),
        [] __device__(
            const thrust::tuple<index_t, index_t, index_t> &a,
            const thrust::tuple<index_t, index_t, index_t> &b
        ) {
          return thrust::get<0>(a) == thrust::get<0>(b) &&
                 thrust::get<1>(a) == thrust::get<1>(b);
        }
    );
    dest.nnz = *ret.second;
    dest.d_values = thrust::device_vector<value_t>(dest.nnz * dest.chunk_size);
    dest.d_modes.push_back(ref.d_modes[0]);
    dest.d_modes.push_back(ref.d_modes[1]);
    // dest.d_modes.push_back();
    dest.d_modes[0].resize(*ret.second);
    dest.d_modes[1].resize(*ret.second);
    auto original_size = fiber_ptr.size();
    fiber_ptr.resize(*ret.second + 1);
    fiber_ptr[fiber_ptr.size() - 1] = original_size;
  }
  dest.d_zip_it = thrust::make_zip_iterator(
      thrust::make_tuple(dest.d_modes[0].begin(), dest.d_modes[1].begin())
  );
}

void construct_fiberidx_and_outTensor(
    thrust::device_vector<index_t> &dest,
    thrust::device_vector<index_t> &fiber_ptr, SparseTensor2 &ref
) {
  thrust::sequence(fiber_ptr.begin(), fiber_ptr.end(), 0);
  auto ret = thrust::unique_by_key(
      ref.d_zip_it, ref.d_zip_it + ref.nnz, fiber_ptr.begin(),
      [] __device__(
          const thrust::tuple<index_t, index_t> &a,
          const thrust::tuple<index_t, index_t> &b
      ) { return thrust::get<0>(a) == thrust::get<0>(b); }
  );
  // for (const auto &e : fiber_ptr)
  //   std::cout << e << ' ';
  dest = ref.d_modes[0];
  dest.resize(*ret.second);
  auto original_size = fiber_ptr.size();
  fiber_ptr.resize(*ret.second + 1);
  fiber_ptr[fiber_ptr.size() - 1] = original_size;
}

void call_contraction_kernel(
    const SparseTensor3 &X, const DenseMatrix &U, SparseTensor2 &Y,
    const thrust::device_vector<index_t> fiber_ptr
) {
  const index_t max_nblocks = 32768;
  const index_t max_nthreads_per_block = 256;
  index_t max_nthreadsy = 32;

  index_t nthreadsx = 1;
  index_t nthreadsy = 1;
  index_t all_nblocks = 0;
  index_t nblocks = 0;
  index_t shmen_size = 0;

  if (U.ncols <= max_nthreadsy)
    nthreadsx = U.ncols;
  else
    nthreadsx = max_nthreadsy;
  nthreadsy = max_nthreads_per_block / nthreadsx;

  if (Y.nnz < nthreadsy) {
    nthreadsy = Y.nnz;
    nblocks = 1;
  } else {
    all_nblocks = (Y.nnz + nthreadsy - 1) / nthreadsy;
    if (all_nblocks < max_nblocks) {
      nblocks = all_nblocks;
    } else {
      nblocks = max_nblocks;
    }
  }
  shmen_size = nthreadsx * nthreadsy * sizeof(value_t);
  assert(shmen_size >= nthreadsx * nthreadsy * sizeof(value_t));
  dim3 dimBlock(nthreadsx, nthreadsy);
  std::cout << "SHMEM size: " << (shmen_size / sizeof(value_t)) << " ("
            << shmen_size << " bytes)\n";
  std::cout << "all_nblocks: " << all_nblocks << "; nthreadsx: " << nthreadsx
            << "; nthreadsy: " << nthreadsy << '\n';

  std::cout << "X_nnz: " << X.nnz << '\n';
  std::cout << "U_rows: " << U.nrows << ": U_cols: " << U.ncols << '\n';
  std::cout << "Y.nnz = " << Y.nnz << "; Y_stride: " << Y.chunk_size << '\n';

  spt_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, shmen_size>>>(
      CAST_THRUST(Y.d_values.data()), Y.chunk_size, Y.nnz,
      CAST_THRUST(X.d_values.data()), CAST_THRUST(X.d_modes[0].data()),
      CAST_THRUST(fiber_ptr.data()), CAST_THRUST(U.d_values.data()), U.nrows,
      U.ncols, U.ncols
  );
}

// void call_contraction_kernel(const thrust::device_vector<index_t> &fiber_ptr,
//                              const thrust::device_vector<index_t>
//                              &X_indices_m, const
//                              thrust::device_vector<value_t> &X_values, const
//                              thrust::device_vector<value_t> &Y_values, const
//                              DenseMatrix &U, const index_t Y_subchunk_size,
//                              const index_t chunk_size) {
//   index_t kernel_blockDim_y = std::min(Y_subchunk_size, 1024 /
//   Y_num_subchunks); assert(kernel_blockDim_y > 0); std::fprintf(
//       stderr,
//       "[CUDA TTM Kernel] Launch ttm_cuda_kernel<<<%zu, (%zu, %zu), 0>>()\n",
//       Y_num_chunks, Y_num_subchunks, kernel_blockDim_y);
//   ttm_cuda_kernel<<<Y.num_chunks, dim3(Y_num_subchunks,
//   kernel_blockDim_y)>>>(
//       CAST_THRUST(fiber_ptr.data()), CAST_THRUST(X_indices_m.data()),
//       U.nrows, U.ncols, Y.chunk_size, subchunk_size, X.chunk_size, U.ncols,
//       CAST_THRUST(Y_values.data()), CAST_THRUST(X_values.data()),
//       CAST_THRUST(U.d_values));
// }

void tensor_times_matrix(
    SparseTensor2 &Y, SparseTensor3 &X, DenseMatrix &U, index_t contracted_mode,
    bool skip_sort
) {
  if (skip_sort) {
    std::cout << "Skipping sort\n";
  } else {
    X.sort_mode(1);
  }
  thrust::device_vector<index_t> fiber_ptr(X.nnz);
  std::cout << "Contracting mode: " << contracted_mode << '\n';

  construct_fiberidx_and_outTensor(Y, fiber_ptr, X, contracted_mode);
  for (const auto &e : fiber_ptr)
    std::cout << e << ' ';
  std::cout << '\n';

  call_contraction_kernel(X, U, Y, fiber_ptr);
  Y.print();
}

void tensor_times_matrix(SparseTensor2 &X, DenseMatrix &U) {

  thrust::device_vector<index_t> remaining_mode(X.nnz);
  thrust::device_vector<index_t> fiber_ptr(X.nnz);
  std::cout << "Contracting mode: 1\n";

  thrust::device_vector<index_t> contracted_mode(X.d_modes[1]);
  std::cout << "Contracted mode: ";
  for (const auto &e : contracted_mode)
    std::cout << e << ' ';
  std::cout << '\n';

  construct_fiberidx_and_outTensor(remaining_mode, fiber_ptr, X);
  std::cout << "Fiber idx: ";
  for (const auto &e : fiber_ptr)
    std::cout << e << ' ';
  std::cout << '\n';

  auto Y_subchunk_size = X.chunk_size;
  auto Y_num_subchunks = X.d_modes[0].size();
  auto Y_chunk_size = X.chunk_size * U.ncols;
  auto Y_num_chunks = fiber_ptr.size() - 1;

  thrust::device_vector<value_t> Y_values(Y_num_chunks * Y_chunk_size);

  std::cout << "Y_num_subchunks = " << Y_num_subchunks << '\n';
  std::cout << "Y_subchunk_size = " << Y_subchunk_size << '\n';
  std::cout << "Y_chunk_size = " << Y_chunk_size << '\n';
  std::cout << "Y_num_chunks = " << Y_num_chunks << '\n';
  std::cout << "Y_values.size() = " << Y_values.size() << '\n';

  // index_t kernel_blockDim_y = std::min(Y_subchunk_size, 1024 /
  // Y_num_subchunks);
  std::fprintf(
      stderr,
      "[CUDA TTM Kernel] Launch ttm_semisparse_kernel<<<(%zu, %zu), >>()\n",
      X.chunk_size, Y_num_chunks
  );
  ttm_semisparse_kernel<<<dim3(X.chunk_size, Y_num_chunks), 1>>>(
      CAST_THRUST(fiber_ptr.data()), CAST_THRUST(contracted_mode.data()),
      U.nrows, U.ncols, Y_chunk_size, X.chunk_size,
      CAST_THRUST(Y_values.data()), CAST_THRUST(X.d_values.data()),
      CAST_THRUST(U.d_values.data())
  );

  int j = 0;
  for (const auto &e : remaining_mode) {
    std::cout << "(" << e << ",:,:) = [";
    for (unsigned i = 0; i < Y_chunk_size; ++i) {
      std::cout << Y_values[j * Y_chunk_size + i] << ' ';
    }
    std::cout << "]\n";
    ++j;
  }

  thrust::host_vector<index_t> h_remaining_mode(remaining_mode);
  DenseMatrix Unew(5, Y_chunk_size, "zeros");
  unsigned row = 0;
  for (const auto &e : h_remaining_mode) {
    for (unsigned i = 0; i < Y_chunk_size; ++i) {
      Unew(e, i) = Y_values[row * Y_chunk_size + i];
    }
    ++row;
  }
  Unew.print();
  for (const auto &e : Unew.h_values) {
    std::cout << e << ' ';
  }
  std::cout << '\n';

  svd(Unew, 2);
}

void call_contraction_kernel(
    const CSFTensor3 &X, const DenseMatrix &U,
    thrust::device_vector<value_t> &out_values, const size_t out_mode
) {
  size_t out_num_chunks = X.fidx[out_mode].size();
  size_t out_chunk_size = U.ncols;
  out_values.resize(out_num_chunks * out_chunk_size);

  const index_t max_nblocks = 32768;
  const index_t max_nthreads_per_block = 256;
  index_t max_nthreadsy = 32;

  index_t nthreadsx = 1;
  index_t nthreadsy = 1;
  index_t all_nblocks = 0;
  index_t nblocks = 0;
  index_t shmen_size = 0;

  if (U.ncols <= max_nthreadsy)
    nthreadsx = U.ncols;
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
  std::cout << "SHMEM size: " << (shmen_size / sizeof(value_t)) << " ("
            << shmen_size << " bytes)\n";
  std::cout << "all_nblocks: " << all_nblocks << "; nthreadsx: " << nthreadsx
            << "; nthreadsy: " << nthreadsy << '\n';

  std::cout << "U_rows: " << U.nrows << ": U_cols: " << U.ncols << '\n';
  std::cout << "out_num_chunks = " << out_num_chunks
            << "; out_chunk_size: " << out_chunk_size << '\n';

  spt_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, shmen_size>>>(
      CAST_THRUST(out_values.data()), out_chunk_size, out_num_chunks,
      CAST_THRUST(X.d_values.data()), CAST_THRUST(X.fidx[out_mode + 1].data()),
      CAST_THRUST(X.fptr[out_mode].data()), CAST_THRUST(U.d_values.data()),
      U.nrows, U.ncols, U.ncols
  );
}
