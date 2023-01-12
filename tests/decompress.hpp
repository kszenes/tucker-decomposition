#ifndef DECOMPRESS_H
#define DECOMPRESS_H

#include "includes.cuh"
#include "macros.cuh"

thrust::host_vector<index_t> decompress_mode(
  const thrust::host_vector<index_t>& fptr,
  const thrust::host_vector<index_t>& fidx
);

#endif /* DECOMPRESS_H */
