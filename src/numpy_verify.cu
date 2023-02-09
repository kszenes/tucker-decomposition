
// Prints to file a python script that verifies result using numpy and tensorly

#include "numpy_verify.cuh"

void print_verification_script(
  const COOTensor3 &X,
  const std::vector<CSFTensor3>& CSFTensors,
  const std::vector<DenseMatrix>& factor_matrices,
  const thrust::device_vector<value_t>& coreTensor
) {
  const std::string filename{"verify.py"};
  fmt::print("Printing verification script to {}\n", filename);
  std::ofstream output(filename);
  output << fmt::format("import numpy as np\n");
  output << fmt::format("import tensorly as tl\n");
  output << fmt::format("from tensorly.decomposition import tucker");
      int i = 0;
      for (const auto& e : X.d_modes) {
        output << fmt::format("\nmode{} = np.array({})\n", i, e);
        ++i;
      }
      output << fmt::format("\nvals = np.array({})\n", X.d_values);
      output << fmt::format("\nX = np.zeros({})\n", CSFTensors.front().shape);
      if (factor_matrices.size() == 3) {
        output << fmt::format("\nX[mode0, mode1, mode2] = vals\n");
      } else {
        output << fmt::format("\nX[mode0, mode1, mode2, mode3] = vals\n");
      }
      output << fmt::format("# Factor matrices:\n");
      i = 0;
      for (const auto& e : factor_matrices) {
        output << fmt::format("\nU{} = np.array({}).reshape({},{})\n",
          i, e.d_values, e.nrows, e.ncols);
        ++i;
      }
      output << fmt::format("\ncore = np.array({}).reshape({}",
        coreTensor, factor_matrices[0].ncols);
      for (unsigned i = 1; i < factor_matrices.size(); ++i) {
        output << fmt::format(",{}", factor_matrices[i].ncols);
      }
      output << fmt::format(").transpose({})\n",
        CSFTensors.front().mode_permutation);

      if (factor_matrices.size() == 3) {
        output << fmt::format("out = np.einsum('ijk,pi,qj,rk->pqr', core, U0, U1, U2)\n");
        output << fmt::format("to_X = np.linalg.norm(X - out) / np.linalg.norm(X)\n");
        output << fmt::format("print('to_X: ', to_X)\n");
        output << fmt::format("core2 = np.einsum('ijk,ip,jq,kr->pqr', X, U0, U1, U2)\n");
        output << fmt::format("to_core = np.linalg.norm(core - core2) / np.linalg.norm(core)\n");
        output << fmt::format("print('to_core: ', to_core)\n");

        output << fmt::format("core_tl, factors_tl = tucker(X, [{},{},{}])\n",
          factor_matrices[0].ncols,
          factor_matrices[1].ncols,
          factor_matrices[2].ncols);
        output << fmt::format("print('Comparing results with tensorly')\n");
        output << fmt::format("rel_er_tl = np.linalg.norm((tl.tucker_to_tensor((core_tl, factors_tl))) - X) / np.linalg.norm(X)\n");
        output << fmt::format("print('rel_er_tl: ', rel_er_tl)\n");

      } else {
        output << fmt::format("out = np.einsum('ijkl,pi,qj,rk,sl->pqrs', core, U0, U1, U2, U3)\n");
        output << fmt::format("to_X = np.linalg.norm(X - out) / np.linalg.norm(X)\n");
        output << fmt::format("print('to_X: ', to_X)\n");
        output << fmt::format("core2 = np.einsum('ijkl,ip,jq,kr,ls->pqrs', X, U0, U1, U2, U3)\n");
        output << fmt::format("to_core = np.linalg.norm(core - core2) / np.linalg.norm(core)\n");
        output << fmt::format("print('to_core: ', to_core)\n");

        output << fmt::format("core_tl, factors_tl = tucker(X, [{},{},{},{}])\n",
          factor_matrices[0].ncols,
          factor_matrices[1].ncols,
          factor_matrices[2].ncols,
          factor_matrices[3].ncols);
        output << fmt::format("print('Comparing results with tensorly')\n");
        output << fmt::format("rel_er_tl = np.linalg.norm((tl.tucker_to_tensor((core_tl, factors_tl))) - X) / np.linalg.norm(X)\n");
        output << fmt::format("print('rel_er_tl: ', rel_er_tl)\n");

      }
}