#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

std::pair<std::string, std::vector<index_t>> parse_args(int argc, char* argv[]) {
  std::string filename(argv[1]);

  std::vector<index_t> matrixSizes;

  for (int i = 2; i < argc; ++i) {
    matrixSizes.push_back(atoi(argv[i]));
  }

  return std::make_pair(filename, matrixSizes);
}

#endif /* PARSE_ARGS_H */
