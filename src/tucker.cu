#include "tucker.cuh"

void tucker_decomp(COOTensor3 &X, const std::vector<index_t> &ranks) {
  assert(X.nmodes == ranks.size() &&
         "Number of U sizes does not match X modes");

  std::ofstream output("out.txt");
  std::vector<double> residuals;

  double coreNorm = 0.0;
  // Core tensor size
  auto coreSize = std::accumulate(
    ranks.begin(), ranks.end(), 1, std::multiplies{});
  fmt::print("coreSize = {}\n", coreSize);
  // Sorting Xs
  std::vector<CSFTensor3> CSFTensors;
  CSFTensors.reserve(X.nmodes);
  std::vector<DenseMatrix> factor_matrices;
  factor_matrices.reserve(X.nmodes);
  GPUTimer csf_timer, total_timer;
  total_timer.start();
  csf_timer.start();
  for (unsigned mode = 0; mode < X.nmodes; ++mode) {
    CSFTensors.emplace_back(X, mode);
    // CSFTensors[mode].print();
    // factor_matrices.emplace_back(X.shape[mode], ranks[mode], "random");
    factor_matrices.emplace_back("/users/kszenes/ParTI/tucker-decomp/example_tensors/dense_5_5.tns", true);
    // factor_matrices.emplace_back("/users/kszenes/ParTI/tucker-decomp/example_tensors/dense_5_2.tns", true);
    // factor_matrices[mode].print();
  }

  // fmt::print("U matrices:\n");
  // for (const auto& e : factor_matrices) {
  //   fmt::print("U_vals = {}\n", e.d_values);
  // }

  auto time = csf_timer.seconds();
  fmt::print("Built all CSF tensors in: {}\n", time);
  auto originalNorm = CSFTensors[0].norm();

  // fmt::print("i = {}\nj = {}\nk = {}\nvals = {}\n",
  //   X.d_modes[0], X.d_modes[1], X.d_modes[2], X.d_values);


  // Testing
  // int mode = 0;
  // const CSFTensor3& csf = CSFTensors[mode];
  // csf.print();
  // auto sspTensor = ttm_chain(csf, factor_matrices);

  // auto subchunk_size = coreSize / factor_matrices[csf.cyclic_permutation.front()].ncols;

  // svd(
  //   csf, sspTensor,
  //   factor_matrices[csf.cyclic_permutation.front()],
  //   subchunk_size
  // );

  // fmt::print("sspTensor: size = {}; vals = {}\n", sspTensor.size(), sspTensor);
  // auto coreTensor =  contract_last_mode(
  //     csf, factor_matrices[csf.cyclic_permutation.front()],
  //     sspTensor, subchunk_size);
  

  auto fitold = 0.0;
  auto fit = 0.0;
  auto subchunk_size = coreSize;
  const int maxiter = 1;
  // Iteration loop
  for (int iter = 0; iter < maxiter; ++iter) {
    fitold = fit;

    fmt::print("\n=== Iteration {} ===\n\n", iter);
    thrust::device_vector<value_t> sspTensor;
    for (unsigned mode = 0; mode < X.nmodes; ++mode) {
      // auto mode = (mode_it + 1) % X.nmodes;
      fmt::print("\n--- TTM chain for mode {} ---\n\n", mode);
      // CSFTensors[mode].print();
      sspTensor = ttm_chain(CSFTensors[mode], factor_matrices);
      // fmt::print("Contraction 1: sspTensor = {}\n", sspTensor);
      subchunk_size = coreSize / factor_matrices[CSFTensors[mode].cyclic_permutation.front()].ncols;

      svd(
        CSFTensors[mode], sspTensor,
        factor_matrices[CSFTensors[mode].cyclic_permutation.front()],
        subchunk_size
      );
      fmt::print("U = {}\n", factor_matrices[CSFTensors[mode].cyclic_permutation.front()].d_values);
      CSFTensors[mode].print();
    }
    fmt::print("core computed through\n");
    auto coreTensor = contract_last_mode(
        CSFTensors.back(), factor_matrices,
        sspTensor, subchunk_size);

    coreNorm =  std::sqrt(thrust::transform_reduce(
      coreTensor.begin(), coreTensor.end(),
      thrust::square{}, 0.0, thrust::plus{}));
    fmt::print("\ncore = {}\n", coreTensor);

    auto normResidual = std::sqrt(std::abs(
      originalNorm * originalNorm - coreNorm * coreNorm));
    residuals.push_back(normResidual);
    fit = 1 - normResidual / originalNorm;
    auto fitchange = std::abs(fitold - fit);
    fmt::print("normCore = {}\n", coreNorm);
    fmt::print("normX = {}\n", originalNorm);
    fmt::print("normResidual = {}\n", normResidual);
    fmt::print("fitchange = {}\n", fitchange);
    double tol = 1e-4;

    if(iter != 0 && normResidual < tol) {

      fmt::print("\n\n === CONVERGED === \n\n");

      output << fmt::format(" --- RESUlTS --- ");
      output << fmt::format("Last CSF");
      CSFTensors.front().print();

      output << fmt::format("Original X:\n");
      for (const auto& e : X.d_modes) {
        output << fmt::format("\nmode = {}\n", e);
      }
      output << fmt::format("\nvals = {}\n", X.d_values);
      output << fmt::format("Factor matrices:\n");
      for (const auto& e : factor_matrices) {
        output << fmt::format("\nU = {}\n", e.d_values);
      }
      output << fmt::format("\ncore = {}\n", coreTensor);
      break;

    }
  }
  auto tot_time = total_timer.seconds();
  fmt::print("Tucker decomposition finished in {} [s]\n", tot_time);
  output << fmt::format("normX = {}\n", originalNorm);
  output << fmt::format("normCore = {}\n", coreNorm);
  output << fmt::format("residuals = {}\n", residuals);
} 
  