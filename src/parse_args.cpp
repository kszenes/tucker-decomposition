#include "parse_args.hpp"

const Params parse_args(int argc, char* argv[]) {
  std::string filename(argv[1]);
  std::vector<index_t> matrixSizes;
  for (int i = 2; i < argc; ++i) {
    matrixSizes.push_back(atoi(argv[i]));
  }
  return Params{matrixSizes, filename};
}

