#include "svd.cuh"

void svd(DenseMatrix& matrix, const size_t rank) {
  auto svd_min = std::min(matrix.nrows, matrix.ncols);
  std::vector<value_t> S(svd_min);
  int info = 0;;
  float work_query = 0.0;
  int lwork = -1;
  info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'O', 'N', 5, 4, matrix.h_values.data(), 4, S.data(), nullptr, 4, nullptr, 1, &work_query, lwork);
  if (info != 0) {
    std::runtime_error("SVD failed!");
  }
  lwork = static_cast<int>(work_query);
  std::vector<float> work(lwork);
  info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'O', 'N', 5, 4, matrix.h_values.data(), 4, S.data(), nullptr, 4, nullptr, 1, work.data(), lwork);
  if (info != 0) {
    std::runtime_error("SVD failed!");
  }

  std::cout << "Singular values: ";
  for (const auto& e : S) std::cout << e << ' ';

  matrix.print();
}