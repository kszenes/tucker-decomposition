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
    fmt::print("SVD: executed in {} [s]\n", time);
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
    fmt::print("SVD;\n\t executed in {} [s]\n", time);
    if (info != 0) {
      std::runtime_error("SVD failed!");
    }
    // fmt::print("U = {}\nS = {}\nVT = {}\n", U, S, VT);
  }
  return U;
}

thrust::device_vector<value_t> call_svd(
    const cusolverDnHandle_t& cusolverH,
    thrust::device_vector<value_t> &sspTensor,
    const size_t svd_rows,
    const size_t svd_cols
) {
    int lwork = 0;
    double *d_work = nullptr;
    double *d_rwork = new double[std::min(svd_rows, svd_cols) - 1];
    int *devInfo = nullptr;
    CUDA_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, svd_cols, svd_rows, &lwork));
    // NOTE: in order to compute U using row major, compute V instead!
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));
    // TODO: One of the copies is superfluous if jobvt set to 'S'
    thrust::device_vector<double> Usvd(svd_rows * svd_rows); // Actually V
    thrust::device_vector<double> Vsvd(svd_cols * svd_cols); // Actually U
    thrust::device_vector<double> S(std::min(svd_rows, svd_cols));
    thrust::device_vector<double> ssp_copy(sspTensor);

    signed char jobu = 'N';
    signed char jobvt = 'A';
    GPUTimer timer;
    timer.start();
    fmt::print("m = {}\n", svd_cols);
    fmt::print("n = {}\n", svd_rows);
    fmt::print("lda = {}\n", svd_cols);
    fmt::print("ldu = {}\n", svd_cols);
    fmt::print("ldvT = {}\n", svd_rows);
    // CUDA_CHECK(cusolverDnDgesvd(
    //   cusolverH, jobu, jobvt, svd_cols, svd_rows,
    //   CAST_THRUST(ssp_copy.data()), svd_cols,
    //   CAST_THRUST(S.data()),
    //   CAST_THRUST(Vsvd.data()), svd_cols, 
    //   CAST_THRUST(Usvd.data()), svd_rows,
    //   d_work, lwork, d_rwork, devInfo));
    auto code = cusolverDnDgesvd(
      cusolverH, jobu, jobvt, svd_cols, svd_rows,
      CAST_THRUST(ssp_copy.data()), svd_cols,
      CAST_THRUST(S.data()),
      CAST_THRUST(Vsvd.data()), svd_cols, 
      CAST_THRUST(Usvd.data()), svd_rows,
      d_work, lwork, d_rwork, devInfo);
    fmt::print("devInfo = {}\n", *devInfo);
    // if (code != CUSOLVER_STATUS_SUCCESS) {
    //   fmt::print("cuSOLVER error #{} with devInfo: {}\n", code, *devInfo);
    //   exit(1);
    // }

    auto time = timer.seconds();
    fmt::print("cuSOLVER SVD exectued in {} [s]\n", time);

    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    delete[] d_rwork;
    return Usvd;
}
thrust::device_vector<value_t> call_svdj(
    const cusolverDnHandle_t& cusolverH,
    thrust::device_vector<value_t> &sspTensor,
    const size_t svd_rows,
    const size_t svd_cols
) {
    int lwork = 0;
    double *d_work = nullptr;
    int *devInfo = nullptr;
    gesvdjInfo_t gesvdj_params = NULL;

    /* configuration of gesvdj  */
    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 0;                                      /* econ = 1 for economy size */
    /* numerical results of gesvdj  */
    double residual = 0;
    int executed_sweeps = 0;

    thrust::device_vector<double> Usvd(svd_rows * svd_rows); // Actually V
    thrust::device_vector<double> Vsvd(svd_cols * svd_cols); // Actually U
    thrust::device_vector<double> S(std::min(svd_rows, svd_cols));
    thrust::device_vector<double> ssp_copy(sspTensor);

    /* step 2: configuration of gesvdj */
    CUDA_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    /* default value of tolerance is machine zero */
    CUDA_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));

    /* default value of max. sweeps is 100 */
    CUDA_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

    fmt::print("svd_rows = {}, svd_cols = {}\n", svd_rows, svd_cols);
    std::printf("tol = %E, default value is machine zero \n", tol);
    std::printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    std::printf("econ = %d \n", econ);

    /* step 4: query working space of SVD */
    CUDA_CHECK(cusolverDnDgesvdj_bufferSize(
        cusolverH, jobz, econ, svd_cols, svd_rows,        
        CAST_THRUST(ssp_copy.data()), svd_cols,             
        CAST_THRUST(S.data()),
        CAST_THRUST(Vsvd.data()), svd_cols,
        CAST_THRUST(Usvd.data()), svd_rows,
        &lwork, gesvdj_params));
    // NOTE: in order to compute U using row major, compute V instead!
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
    // TODO: One of the copies is superfluous if jobvt set to 'S'

    GPUTimer timer;
    timer.start();
    CUDA_CHECK(cusolverDnDgesvdj(
        cusolverH, jobz, econ, svd_cols, svd_rows,        
        CAST_THRUST(ssp_copy.data()), svd_cols,             
        CAST_THRUST(S.data()),
        CAST_THRUST(Vsvd.data()), svd_cols,
        CAST_THRUST(Usvd.data()), svd_rows,
        d_work, lwork, devInfo, gesvdj_params));
    auto time = timer.seconds();
    fmt::print("cuSOLVER SVD exectued in {} [s]\n", time);

    CUDA_CHECK(cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps));
    CUDA_CHECK(cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual));
    std::printf("residual |A - U*S*V**H|_F = %E \n", residual);
    std::printf("number of executed sweeps = %d \n", executed_sweeps);

    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    return Usvd;
}

void svd(
  const CSFTensor3& csf,
  thrust::device_vector<value_t>& sspTensor,
  DenseMatrix& U_to_update,
  const index_t subchunk_size,
  const bool on_gpu
) {
  unsigned svd_rows = csf.fidx[0].size();
  unsigned svd_cols = subchunk_size;
  fmt::print("SVD: nrows = {}; ncols = {}\n", svd_rows, svd_cols);
  thrust::host_vector<index_t> last_mode(csf.fidx[0]);
  if (on_gpu) {
    cusolverDnHandle_t cusolverH;
    CUDA_CHECK(cusolverDnCreate(&cusolverH));
    const auto Usvd = call_svdj(cusolverH, sspTensor, svd_rows, svd_cols);

    // TODO: Remove fill
    thrust::fill(U_to_update.d_values.begin(), U_to_update.d_values.end(), 0);
    for (unsigned row = 0; row < svd_rows; row++) {
      auto offset_it = Usvd.begin() + (row * last_mode.size());
      thrust::copy(offset_it, offset_it + U_to_update.ncols,
                   U_to_update.d_values.begin() + (last_mode[row] * U_to_update.ncols));
    }
    // fmt::print("To update = {}\n", U_to_update.d_values);
    CUDA_CHECK(cusolverDnDestroy(cusolverH))
  } else {
    thrust::host_vector<value_t> h_sspTensor(sspTensor); 
    thrust::host_vector<value_t> Usvd =
        call_svd(h_sspTensor, svd_rows, svd_cols, true);

    // fmt::print("Usvd = {}\n", Usvd);

    // TODO: Fill with zeros needed since values non zero at the beginning
    // FIX:  Initialize appropriate rows to zero
    thrust::fill(U_to_update.h_values.begin(), U_to_update.h_values.end(), 0);
    
    for (unsigned row = 0; row < svd_rows; row++) {
      auto offset_it = Usvd.begin() + (row * last_mode.size());
      thrust::copy(offset_it, offset_it + U_to_update.ncols,
                   U_to_update.h_values.begin() + (last_mode[row] * U_to_update.ncols));
    }
    U_to_update.to_device();
    // fmt::print("U_to_update = {}\n", U_to_update.d_values);
  }
}
