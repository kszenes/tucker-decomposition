#ifndef CSFTENSOR3_H
#define CSFTENSOR3_H


#include "includes.cuh"
#include "macros.cuh"
#include "COOTensor3.cuh"

class CSFTensor3 {
public:
  explicit CSFTensor3(const COOTensor3& coo_tensor, const int& mode);

  void buildCSFTensor3(
      std::vector<thrust::device_vector<index_t>> &coo_modes
  );
  void print() const;

  value_t norm() const;

  index_t nmodes;
  index_t nnz;
  std::vector<index_t> shape;
  std::vector<index_t> cyclic_permutation;
  thrust::device_vector<value_t> d_values;
  std::vector<thrust::device_vector<index_t>> fptr;
  std::vector<thrust::device_vector<index_t>> fidx;
};


#endif /* CSFTENSOR3_H */
