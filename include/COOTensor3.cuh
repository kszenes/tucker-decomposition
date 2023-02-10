#ifndef COOTENSOR3_H
#define COOTENSOR3_H

#include "includes.cuh"
#include "macros.cuh"

class COOTensor3 {
public:
  explicit COOTensor3() = default;
  explicit COOTensor3(const std::vector<index_t> &shape,
                        const index_t mode, const index_t new_chunk_size);
  explicit COOTensor3(const std::string &filename, const bool starts_at_zero);

  void to_device();
  void to_host();
  void print();

  index_t nmodes;
  index_t nnz;
  index_t chunk_size;
  std::vector<index_t> shape;
  std::vector<thrust::device_vector<index_t>> d_modes;
  std::vector<thrust::host_vector<index_t>> h_modes;
  thrust::device_vector<value_t> d_values;
  thrust::host_vector<value_t> h_values;
};

#endif /* COOTENSOR3_H */
