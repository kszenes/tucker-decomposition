#include "svd.cuh"

__global__
void copy_to_U(
  const double *__restrict__ in,
  const int rows,
  const int in_cols,
  const int out_cols,
  double *__restrict__ out
) {
  int tix = threadIdx.x + blockIdx.x * blockDim.x;

  if (tix < out_cols) {
    for (int tiy = threadIdx.y + blockIdx.y * blockDim.y;
        tiy < rows;
        tiy += gridDim.x * gridDim.y) {
      out[tix + tiy * out_cols] = in[tix + tiy * in_cols];
    }
  }
}

void call_copy(
  const double *__restrict__ in,
  const int rows,
  const int in_cols,
  const int out_cols,
  double *__restrict__ out
) {
  dim3 threads{
    32,
    32
  };
  dim3 blocks{
    (out_cols + threads.x - 1) / threads.x,
    (rows + threads.y - 1) / threads.y
  };
  copy_to_U<<<blocks, threads>>>(
    thrust::raw_pointer_cast(in),
    rows,
    in_cols,
    out_cols,
    thrust::raw_pointer_cast(out)
  );
  cudaDeviceSynchronize();
}

thrust::device_vector<value_t> transpose_matrix(
  const cublasHandle_t& cublasH,
  const thrust::device_vector<value_t>& matrix,
  const int ldm
) {
  GPUTimer timer;
  timer.start();
  fmt::print("TRANSPOSING MATRIX\n");
  const double alpha = 1.0;
  const double beta = 0.0;
  const int ldc = matrix.size() / ldm;

  thrust::device_vector<value_t> matrix_T(matrix.size());
  CUDA_CHECK(cublasDgeam(
    cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
    ldc, ldm, &alpha,
    thrust::raw_pointer_cast(matrix.data()), ldm,
    &beta, nullptr, ldc,
    thrust::raw_pointer_cast(matrix_T.data()), ldc
  ));
  auto time = timer.seconds();
  fmt::print("Transpose completed in {}\n", time);
  return matrix_T;
}

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
  GPUTimer timer;
  if (only_U) {
    info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 'S', 'N', nrows, ncols, matrix.data(), ncols,
        S.data(), nullptr, nrows, nullptr, 1, &work_query, lwork
    );
    if (info != 0) {
      std::runtime_error("SVD failed!");
    }
    lwork = static_cast<int>(work_query);
    std::vector<value_t> work(lwork);
    timer.start();
    info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 'S', 'N', nrows, ncols, matrix.data(), ncols,
        S.data(), nullptr, nrows, nullptr, 1, work.data(), lwork
    );
    auto time = timer.seconds();
    fmt::print("SVD: executed in {} [s]\n", time);
    if (info != 0) {
      std::runtime_error("SVD failed!");
    }
    // fmt::print("U = {}\nS = {}\n", U, S);
    return matrix;
  } else {
    thrust::host_vector<value_t> U(nrows * nrows);
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
    return U;
  }
}

thrust::device_vector<value_t> call_svd(
    const cublasHandle_t& cublasH,
    const cusolverDnHandle_t& cusolverH,
    thrust::device_vector<value_t> &sspTensor,
    const size_t svd_rows,
    const size_t svd_cols
) {
    int lwork = 0;
    double *d_work = nullptr;
    double *d_rwork = new double[std::min(svd_rows, svd_cols) - 1];
    int *devInfo = nullptr;
    signed char jobu = 'N';
    signed char jobvt = 'O';
    thrust::device_vector<double> ssp_copy;;
    thrust::device_vector<double> S(std::min(svd_rows, svd_cols));
    size_t m = 0, n = 0;

    const bool need_transpose = svd_cols < svd_rows;
    if (need_transpose) {
      jobu = 'O';
      jobvt = 'N';
      ssp_copy = transpose_matrix(cublasH, sspTensor, svd_cols);
      m = svd_rows;
      n = svd_cols;

    } else {
      jobu = 'N';
      jobvt = 'O';
      ssp_copy.resize(sspTensor.size());
      thrust::copy(sspTensor.begin(), sspTensor.end(), ssp_copy.begin());
      m = svd_cols;
      n = svd_rows;
    }

    CUDA_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));
    // NOTE: in order to compute U using row major, compute V instead!
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));


    GPUTimer svd_timer;
    svd_timer.start();
    CUDA_CHECK(cusolverDnDgesvd(
      cusolverH, jobu, jobvt, m, n,
      thrust::raw_pointer_cast(ssp_copy.data()), m,
      thrust::raw_pointer_cast(S.data()),
      nullptr, m, 
      nullptr, n,
      d_work, lwork, d_rwork, devInfo));
    auto svd_time = svd_timer.seconds();
    fmt::print("cuSOLVER Vanilla SVD routine: {} [s]\n", svd_time);
    // fmt::print("devInfo = {}\n", *devInfo);
    // if (code != CUSOLVER_STATUS_SUCCESS) {
    //   fmt::print("cuSOLVER error #{} with devInfo: {}\n", code, *devInfo);
    //   exit(1);
    // }

    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    delete[] d_rwork;
    return need_transpose ? transpose_matrix(cublasH, ssp_copy, svd_rows) : ssp_copy;
}
thrust::device_vector<value_t> call_svdj(
    const cublasHandle_t& cublasH,
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
    const int econ = 1;                                      /* econ = 1 for economy size */
    /* numerical results of gesvdj  */
    double residual = 0;
    int executed_sweeps = 0;

    auto svd_min = std::min(svd_rows, svd_cols);
    thrust::device_vector<double> Usvd(svd_rows * svd_min); // Actually V
    thrust::device_vector<double> Vsvd(svd_cols * svd_min); // Actually U
    thrust::device_vector<double> S(svd_min);
    thrust::device_vector<double> ssp_copy(sspTensor);

    /* step 2: configuration of gesvdj */
    CUDA_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    /* default value of tolerance is machine zero */
    CUDA_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));

    /* default value of max. sweeps is 100 */
    CUDA_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

    // fmt::print("svd_rows = {}, svd_cols = {}\n", svd_rows, svd_cols);
    // std::printf("tol = %E, default value is machine zero \n", tol);
    // std::printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    // std::printf("econ = %d \n", econ);

    /* step 4: query working space of SVD */
    CUDA_CHECK(cusolverDnDgesvdj_bufferSize(
        cusolverH, jobz, econ, svd_cols, svd_rows,        
        thrust::raw_pointer_cast(ssp_copy.data()), svd_cols,             
        thrust::raw_pointer_cast(S.data()),
        thrust::raw_pointer_cast(Vsvd.data()), svd_cols,
        thrust::raw_pointer_cast(Usvd.data()), svd_rows,
        &lwork, gesvdj_params));
    // NOTE: in order to compute U using row major, compute V instead!
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    GPUTimer timer;
    timer.start();
    CUDA_CHECK(cusolverDnDgesvdj(
        cusolverH, jobz, econ, svd_cols, svd_rows,        
        thrust::raw_pointer_cast(ssp_copy.data()), svd_cols,             
        thrust::raw_pointer_cast(S.data()),
        thrust::raw_pointer_cast(Vsvd.data()), svd_cols,
        thrust::raw_pointer_cast(Usvd.data()), svd_rows,
        d_work, lwork, devInfo, gesvdj_params));
    auto time = timer.seconds();
    fmt::print("cuSOLVER Jacobi SVD routine: {} [s]\n", time);

    CUDA_CHECK(cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps));
    CUDA_CHECK(cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual));
    // std::printf("residual |A - U*S*V**H|_F = %E \n", residual);
    // std::printf("number of executed sweeps = %d \n", executed_sweeps);

    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));

    return transpose_matrix(cublasH, Usvd, svd_rows);
    // return Usvd;
}

thrust::device_vector<value_t> call_svdp(
    const cublasHandle_t& cublasH,
    const cusolverDnHandle_t& cusolverH,
    thrust::device_vector<value_t> &sspTensor,
    const size_t svd_rows,
    const size_t svd_cols
) {
    size_t d_lwork = 0;
    size_t h_lwork = 0;
    void *d_work = nullptr;
    void *h_work = nullptr;
    int *devInfo = nullptr;

    /* configuration of gesvdj  */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 1;                                      /* econ = 1 for economy size */
    double h_err_sigma;
    /* numerical results of gesvdj  */

    auto svd_min = std::min(svd_rows, svd_cols);
    thrust::device_vector<double> Usvd(svd_rows * svd_min); // Actually V
    thrust::device_vector<double> Vsvd(svd_cols * svd_min); // Actually U
    thrust::device_vector<double> S(svd_min);
    thrust::device_vector<double> ssp_copy(sspTensor);

    std::printf("econ = %d \n", econ);

    /* step 4: query working space of SVD */
    CUDA_CHECK(cusolverDnXgesvdp_bufferSize(
        cusolverH, NULL,
        jobz, econ, svd_cols, svd_rows,
        CUDA_R_64F,        
        thrust::raw_pointer_cast(ssp_copy.data()), svd_cols,             
        CUDA_R_64F,
        thrust::raw_pointer_cast(S.data()),
        CUDA_R_64F,
        thrust::raw_pointer_cast(Vsvd.data()), svd_cols,
        CUDA_R_64F,
        thrust::raw_pointer_cast(Usvd.data()), svd_rows,
        CUDA_R_64F,
        &d_lwork, &h_lwork));
    // NOTE: in order to compute U using row major, compute V instead!
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * d_lwork));
    if (0 < h_lwork) {
      h_work = reinterpret_cast<void *>(malloc(h_lwork));
      if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
      }
    }

    GPUTimer timer;
    timer.start();
    CUDA_CHECK(cusolverDnXgesvdp(
        cusolverH, NULL,
        jobz, econ, svd_cols, svd_rows,
        CUDA_R_64F,        
        thrust::raw_pointer_cast(ssp_copy.data()), svd_cols,             
        CUDA_R_64F,
        thrust::raw_pointer_cast(S.data()),
        CUDA_R_64F,
        thrust::raw_pointer_cast(Vsvd.data()), svd_cols,
        CUDA_R_64F,
        thrust::raw_pointer_cast(Usvd.data()), svd_rows,
        CUDA_R_64F,
        d_work, d_lwork, h_work, h_lwork,
        devInfo, &h_err_sigma));
    auto time = timer.seconds();
    fmt::print("cuSOLVER POLAR SVD routine: {} [s]\n", time);


    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    return transpose_matrix(cublasH, Usvd, svd_rows);
    // return Usvd;
}

void svd(
  const CSFTensor3& csf,
  thrust::device_vector<value_t>& sspTensor,
  DenseMatrix& U_to_update,
  const index_t subchunk_size,
  const bool on_gpu,
  const cusolverDnHandle_t cusolverH,
  const cublasHandle_t cublasH,
  SVD_routine svd_routine
) {
  unsigned svd_rows = csf.fidx[0].size();
  unsigned svd_cols = subchunk_size;
  fmt::print("SVD: nrows = {}; ncols = {}\n", svd_rows, svd_cols);
  thrust::host_vector<index_t> last_mode(csf.fidx[0]);
  if (on_gpu) {
    int cols = 0;
    switch (svd_routine) {
      case SVD_routine::qr: {
        const auto Usvd = call_svd(cublasH, cusolverH, sspTensor, svd_rows, svd_cols);
        // for (unsigned row = 0; row < svd_rows; row++) {
        //   auto offset_it = Usvd.begin() + row * svd_cols;
        //   thrust::copy(offset_it, offset_it + U_to_update.ncols,
        //               U_to_update.d_values.begin() + (last_mode[row] * U_to_update.ncols));
        // }
        cols = svd_cols;
        call_copy(
            thrust::raw_pointer_cast(Usvd.data()), svd_rows, cols,
            U_to_update.ncols,
            thrust::raw_pointer_cast(U_to_update.d_values.data())
        );
        break;
      }
      case SVD_routine::jacobi: {
        const auto Usvd = call_svdj(cublasH, cusolverH, sspTensor, svd_rows, svd_cols);
        // for (unsigned row = 0; row < svd_rows; row++) {
        //   auto offset_it = Usvd.begin() + row * std::min(svd_rows, svd_cols);
        //   thrust::copy(
        //     offset_it, offset_it + U_to_update.ncols,
        //     U_to_update.d_values.begin() + (last_mode[row] * U_to_update.ncols));
        // }
        cols = std::min(svd_cols, svd_rows);
        call_copy(
          thrust::raw_pointer_cast(Usvd.data()),
          svd_rows,
          cols,
          U_to_update.ncols,
          thrust::raw_pointer_cast(U_to_update.d_values.data())
        );
        break;
      }
      case SVD_routine::polar: {
        const auto Usvd = call_svdp(cublasH, cusolverH, sspTensor, svd_rows, svd_cols);
        // for (unsigned row = 0; row < svd_rows; row++) {
        //   auto offset_it = Usvd.begin() + row * std::min(svd_rows, svd_cols);
        //   thrust::copy(offset_it, offset_it + U_to_update.ncols,
        //               U_to_update.d_values.begin() + (last_mode[row] * U_to_update.ncols));
        // }
        cols = std::min(svd_cols, svd_rows);
        call_copy(
          thrust::raw_pointer_cast(Usvd.data()),
          svd_rows,
          cols,
          U_to_update.ncols,
          thrust::raw_pointer_cast(U_to_update.d_values.data())
        );
        break;
      }
      default:
        std::runtime_error("Unknown SVD routine\n");
    }
  } else {
    thrust::host_vector<value_t> h_sspTensor(sspTensor); 
    thrust::host_vector<value_t> Usvd =
        call_svd(h_sspTensor, svd_rows, svd_cols, true);

    // fmt::print("Usvd = {}\n", Usvd);

    // TODO: Fill with zeros needed since values non zero at the beginning
    // FIX:  Initialize appropriate rows to zero
    // thrust::fill(U_to_update.h_values.begin(), U_to_update.h_values.end(), 0);
    for (unsigned row = 0; row < svd_rows; row++) {
      auto offset_it = Usvd.begin() + row * svd_cols;
      thrust::copy(offset_it, offset_it + U_to_update.ncols,
                   U_to_update.d_values.begin() + (last_mode[row] * U_to_update.ncols));
    }
  }
}
