#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include "includes.cuh"
#include "macros.cuh"

// Currently stored in row major
class DenseMatrix {
public:
  explicit DenseMatrix() = default;
  explicit DenseMatrix(const index_t rows, const index_t cols,
                       const std::string &method = "random");
  explicit DenseMatrix(const std::string &filename,
                       const bool is_tranposed = false);

  value_t &operator()(const index_t row, const index_t col);
  void to_device();
  void to_host();
  void print(const bool sync = false);

  index_t nrows, ncols;
  thrust::host_vector<value_t> h_values;
  thrust::device_vector<value_t> d_values;
};

#endif /* DENSEMATRIX_H */