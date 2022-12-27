#include "macros.cuh"
#include "includes.cuh"
#include "DenseMatrix.cuh"

#include <mkl_lapacke.h>

void svd(DenseMatrix& matrix, const size_t rank);