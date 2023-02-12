#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include "svd.cuh"

struct Params {
  std::vector<index_t> ranks;
  std::string filename;
  SVD_routine svd_routine = SVD_routine::qr;
  bool print_verification = false;
  int maxiter = 100;
  double tol = 1e-5;
};

const Params parse_args(int argc, char* argv[]);

#endif /* PARSE_ARGS_H */
