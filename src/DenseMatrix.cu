#include "DenseMatrix.cuh"

// NOTE: transposes internally matrix
DenseMatrix::DenseMatrix(const std::string &filename,
                         const bool is_transposed) {
  std::cout << "Reading dense matrix from: " << filename << '\n';
  std::ifstream ifile(filename);
  index_t num_rows = 0, num_cols = 0;
  index_t index(0);
  value_t value(0.0);

  if (!ifile.is_open()) {
    std::cerr << "Failed to open " << filename << '\n';
    exit(1);
  }
  if (!(ifile >> index)) {
    std::cerr << "error occured during reading tensor\n";
    exit(1);
  };
  assert((index == 2) && "File contains with number of modes different from 2");

  ifile >> num_rows >> num_cols;
  // tranpose matrix
  if (is_transposed) {
    nrows = num_cols;
    ncols = num_rows;
  } else {
    nrows = num_rows;
    ncols = num_cols;
  }

  h_values.resize(nrows * ncols);

  for (unsigned row = 0; row < num_rows; ++row) {
    for (unsigned col = 0; col < num_cols; ++col) {
      if (!(ifile >> value)) {
        std::cerr << "Error occured during reading of tensor\n";
        exit(1);
      }
      if (is_transposed) {
        h_values[col * num_rows + row] = value;
      } else {
        h_values[row * num_cols + col] = value;
      }
    }
  }

  // Copy to device
  to_device();

  ifile.close();
}

DenseMatrix::DenseMatrix(const index_t rows, const index_t cols,
                         const std::string &method)
    : nrows(rows), ncols(cols) {
  h_values.resize(nrows * ncols);
  d_values.resize(nrows * ncols);
  if (method == "random") {
    std::cout << "Generating random normal matrix: " << rows << " x " << cols
              << '\n';
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<value_t> dist{0, 1};
    for (auto &e : h_values)
      e = dist(mt);

  } else if (method == "ones") {
    std::cout << "Generating matrix of ones: " << rows << " x " << cols;
    for (auto &e : h_values)
      e = 1.0;
  } else {
    std::cout << "Generating matrix of zeros: " << rows << " x " << cols;
    for (auto &e : h_values)
      e = 0.0;
  }

  d_values = h_values;
}

void DenseMatrix::to_device() { d_values = h_values; }

void DenseMatrix::to_host() { h_values = d_values; }

void DenseMatrix::print(const bool sync) {
  if (sync) {
    to_host();
  }
  std::string output =
      "\nMatrix size: " + std::to_string(nrows) + " x " + std::to_string(ncols);
  output += ":\n\n";
  for (unsigned row = 0; row < nrows; ++row) {
    for (unsigned col = 0; col < ncols; ++col) {
      output += std::to_string(h_values[row * ncols + col]) + "  ";
    }
    output += '\n';
  }
  std::cout << output << '\n';
}

value_t &DenseMatrix::operator()(const index_t row, const index_t col) {
  assert((row < nrows && col < ncols) && "Index out of bounds");
  return h_values[row * ncols + col];
}