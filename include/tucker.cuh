#ifndef TUCKER_H
#define TUCKER_H

#include <numeric>
#include <functional>
#include <unordered_map>

#include "includes.cuh"
#include "macros.cuh"
#include "COOTensor3.cuh"
#include "CSFTensor3.cuh"
#include "DenseMatrix.cuh"
#include "svd.cuh"
#include "ttm.cuh"
#include "numpy_verify.cuh"

void tucker_decomp(COOTensor3 &X, const std::vector<index_t> &ranks, const double tol, const int maxiter);

#endif /* TUCKER_H */
