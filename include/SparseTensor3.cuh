#ifndef SPARSETENSOR3_H
#define SPARSETENSOR3_H

#include "includes.cuh"
#include "macros.cuh"

class SparseTensor3 {
public:
  using IntIterator = thrust::device_vector<index_t>::iterator;
  using IteratorTuple = thrust::tuple<IntIterator, IntIterator, IntIterator>;
  using IndexTuple = thrust::tuple<index_t, index_t, index_t>;
  using ZipIterator = thrust::zip_iterator<IteratorTuple>;

  explicit SparseTensor3() = default;
  explicit SparseTensor3(const SparseTensor3& other, const std::vector<index_t>& permutation);
  explicit SparseTensor3(const std::vector<index_t> &mode_sizes,
                        const index_t mode, const index_t new_chunk_size);
  explicit SparseTensor3(const std::string &filename, const bool starts_at_zero);

  void to_device();
  void to_host();
  void print();
  void sort_mode(const index_t mode);

  index_t nmodes;
  index_t nnz;
  index_t chunk_size;
  std::vector<index_t> mode_sizes;
  std::vector<index_t> mode_permutation;
  std::vector<thrust::device_vector<index_t>> d_modes;
  std::vector<thrust::host_vector<index_t>> h_modes;
  thrust::device_vector<value_t> d_values;
  thrust::host_vector<value_t> h_values;
  ZipIterator d_zip_it;
};


#endif /* SPARSETENSOR3_H */
