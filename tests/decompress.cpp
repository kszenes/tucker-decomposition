#include "decompress.hpp"

thrust::host_vector<index_t> decompress_mode(
  const thrust::host_vector<index_t>& fptr,
  const thrust::host_vector<index_t>& fidx
) {
  thrust::host_vector<index_t> decompressed_mode;
  for (index_t mode_idx = 0; mode_idx < fidx.size(); ++mode_idx) {
    for (index_t rep = fptr[mode_idx]; rep < fptr[mode_idx + 1]; ++rep) {
      decompressed_mode.push_back(fidx[mode_idx]);
    }
  }
  return decompressed_mode;
}