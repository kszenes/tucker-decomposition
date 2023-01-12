#ifndef TUCKER_H
#define TUCKER_H

#include <numeric>
#include <functional>

#include "includes.cuh"
#include "macros.cuh"
#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"
#include "svd.cuh"
#include "ttm.cuh"

void tucker_decomp(COOTensor3 &X, const std::vector<index_t> &ranks);

#endif /* TUCKER_H */
