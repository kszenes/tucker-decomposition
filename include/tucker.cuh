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
#include "parse_args.hpp"

void tucker_decomp(COOTensor3 &X, const Params& params);

#endif /* TUCKER_H */
