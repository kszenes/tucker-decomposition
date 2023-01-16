#include "svd.cuh"

thrust::host_vector<value_t> call_svd(
    thrust::host_vector<value_t> &matrix, const size_t nrows,
    const size_t ncols, const bool only_U
) {
  // fmt::print("nrows = {}; ncols = {}\n", nrows, ncols);
  auto svd_min = std::min(nrows, ncols);
  std::vector<value_t> S(svd_min);
  int info = 0;
  value_t work_query = 0.0;
  int lwork = -1;
  thrust::host_vector<value_t> U(nrows * nrows);
  GPUTimer timer;
  if (only_U) {
    info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 'A', 'N', nrows, ncols, matrix.data(), ncols,
        S.data(), U.data(), nrows, nullptr, 1, &work_query, lwork
    );
    if (info != 0) {
      std::runtime_error("SVD failed!");
    }
    lwork = static_cast<int>(work_query);
    std::vector<value_t> work(lwork);
    timer.start();
    info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 'A', 'N', nrows, ncols, matrix.data(), ncols,
        S.data(), U.data(), nrows, nullptr, 1, work.data(), lwork
    );
    auto time = timer.seconds();
    fmt::print("SVD: executed in {} [s] ", time);
    if (info != 0) {
      std::runtime_error("SVD failed!");
    }
    // fmt::print("U = {}\nS = {}\n", U, S);
  } else {
    thrust::host_vector<value_t> VT(ncols * ncols);
    info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 'A', 'A', nrows, ncols, matrix.data(), ncols,
        S.data(), U.data(), nrows, VT.data(), ncols, &work_query, lwork
    );
    if (info != 0) {
      std::runtime_error("SVD failed!");
    }
    lwork = static_cast<int>(work_query);
    std::vector<value_t> work(lwork);
    timer.start();
    info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 'A', 'A', nrows, ncols, matrix.data(), ncols,
        S.data(), U.data(), nrows, VT.data(), ncols, work.data(), lwork
    );
    auto time = timer.seconds();
    fmt::print("SVD;\n\t executed in {} [s]", time);
    if (info != 0) {
      std::runtime_error("SVD failed!");
    }
    // fmt::print("U = {}\nS = {}\nVT = {}\n", U, S, VT);
  }
  return U;
}

void svd(
  const CSFTensor3& csf,
  thrust::device_vector<value_t>& sspTensor,
  DenseMatrix& U_to_update,
  const index_t subchunk_size
  const bool on_gpu
) {
  thrust::host_vector<value_t> h_sspTensor(sspTensor); 
  thrust::host_vector<index_t> last_mode(csf.fidx[0]);

  thrust::host_vector<value_t> Usvd =
      call_svd(h_sspTensor, last_mode.size(), subchunk_size, true);

  // TODO: Fill with zeros needed since values non zero at the beginning
  // FIX:  Initialize appropriate rows to zero
  thrust::fill(U_to_update.h_values.begin(), U_to_update.h_values.end(), 0);

  
  for (unsigned row = 0; row < last_mode.size(); row++) {
    for (unsigned col = 0; col < U_to_update.ncols; col++) {
      U_to_update.h_values[last_mode[row] * U_to_update.ncols + col] = Usvd[row * last_mode.size() + col];
    }
  }
  U_to_update.to_device();
  // fmt::print("U_to_update = {}\n", U_to_update.d_values);

}
