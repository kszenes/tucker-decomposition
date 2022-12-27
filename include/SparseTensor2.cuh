#ifndef SPARSETENSOR2_H
#define SPARSETENSOR2_H

#include "includes.cuh"
#include "macros.cuh"

class SparseTensor2 {
public:
  using IntIterator = thrust::device_vector<index_t>::iterator;
  using IteratorTuple = thrust::tuple<IntIterator, IntIterator>;
  using IndexTuple = thrust::tuple<index_t, index_t>;
  using ZipIterator = thrust::zip_iterator<IteratorTuple>;

  explicit SparseTensor2() = default;
  explicit SparseTensor2(const SparseTensor2& other);
  explicit SparseTensor2(const std::vector<index_t> &mode_sizes,
                        const index_t mode, const index_t new_chunk_size);

  void to_device();
  void to_host();
  void print();
  void sort_mode(const index_t mode);

  index_t nmodes;
  index_t nnz;
  index_t chunk_size;
  std::vector<index_t> mode_sizes;
  std::vector<thrust::device_vector<index_t>> d_modes;
  std::vector<thrust::host_vector<index_t>> h_modes;
  thrust::device_vector<value_t> d_values;
  thrust::host_vector<value_t> h_values;
  ZipIterator d_zip_it;
};

#endif /* SPARSETENSOR2_H */
