#include "tucker.cuh"


void tucker_decomp(COOTensor3 &X, const std::vector<index_t> &ranks, const double tol, const int maxiter) {
  std::unordered_map<std::string, CPUTimer> timers;
  timers["total"] = CPUTimer();
  timers["misc"] = CPUTimer();

  std::unordered_map<std::string, float> timings;
  timings["csf"] = 0.0f;
  timings["ttm"] = 0.0f;
  timings["svd"] = 0.0f;
  timings["core"] = 0.0f;
  timings["total"] = 0.0f;

  timers["total"].start();

  assert(X.nmodes == ranks.size() &&
         "Number of U sizes does not match X modes");
  auto use_gpu = true;
  cusolverDnHandle_t cusolverH;
  cublasHandle_t cublasH;
  CUDA_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cusolverDnCreate(&cusolverH));

  std::vector<double> fits;
  double core_sqnorm = 0.0;
  // Core tensor size
  auto coreSize = std::accumulate(
    ranks.begin(), ranks.end(), 1, std::multiplies{});
  // Sorting Xs
  timers["misc"].start();
  std::vector<CSFTensor3> CSFTensors;
  CSFTensors.reserve(X.nmodes);
  for (unsigned mode = 0; mode < X.nmodes; ++mode) {
    CSFTensors.emplace_back(X, mode);
  }
  timings["csf"] = timers["misc"].seconds();

  std::vector<DenseMatrix> factor_matrices;
  factor_matrices.reserve(X.nmodes);
  for (unsigned mode = 0; mode < X.nmodes; ++mode) {
    factor_matrices.emplace_back(X.shape[mode], ranks[mode], "random");
  }

  // === Print fidx and fptr sizes ===
  // int ii = 0;
  // for (const auto& e : CSFTensors) {
  //   auto fptr_size = std::accumulate(
  //     e.fptr.begin(), e.fptr.end(), 0,
  //     [](const auto& a, const auto& b){ return a + b.size(); });
  //   auto fidx_size = std::accumulate(
  //     e.fidx.begin(), e.fidx.end(), 0,
  //     [](const auto& a, const auto& b){ return a + b.size(); });
  //   fmt::print("Mode {}: fidx_size = {}, fptr_size = {}\n", ii, fidx_size, fptr_size);
  //   ++ii;
  // }

  auto original_sqnorm = CSFTensors[0].sqnorm();

  auto fitold = 0.0;
  auto fit = 0.0;
  auto subchunk_size = coreSize;
  // Iteration loop
  int iter = 0;
  for (; iter < maxiter; ++iter) {
    timers["iter"].start();
    fitold = fit;

    fmt::print("\n=== Iteration {} ===\n\n", iter);
    thrust::device_vector<value_t> sspTensor;
    for (unsigned mode_it = 0; mode_it < X.nmodes; ++mode_it) {
      // NOTE: This offset is only needed for numpy
      // verification script
      auto mode = (mode_it + 1) % X.nmodes;
      fmt::print("\n--- TTM chain for mode {} ---\n\n", mode);
      timers["misc"].start();
      sspTensor = ttm_chain(CSFTensors[mode], factor_matrices);
      auto ttm_time = timers["misc"].seconds();
      fmt::print("TTMC completed in {}\n", ttm_time);
      timings["ttm"] += ttm_time;

      subchunk_size = coreSize / factor_matrices[CSFTensors[mode].mode_permutation.front()].ncols;

      timers["misc"].start();
      svd(
        CSFTensors[mode], sspTensor,
        factor_matrices[CSFTensors[mode].mode_permutation.front()],
        subchunk_size, use_gpu, cusolverH, cublasH
      );
      auto svd_time = timers["misc"].seconds();
      fmt::print("SVD completed in {} [s]\n", svd_time);
      timings["svd"] += svd_time;
    }
    timers["misc"].start();
    auto coreTensor = contract_last_mode(
        CSFTensors.front(), factor_matrices,
        sspTensor, subchunk_size);
    auto core_time = timers["misc"].seconds();
    timings["core"] += core_time;

    core_sqnorm =  thrust::transform_reduce(
      coreTensor.begin(), coreTensor.end(),
      thrust::square{}, 0.0, thrust::plus{});
    auto iter_time = timers["iter"].seconds();
    fmt::print("\nIteration completed in {}\n", iter_time);

    auto sqresid = original_sqnorm - core_sqnorm;
    auto resid = std::sqrt(std::max(0.0, sqresid));
    fit = 1 - (resid / std::sqrt(original_sqnorm));
    fits.push_back(fit);
    auto fitchange = std::abs(fitold - fit);
    fmt::print("fit = {}; fitchange = {}\n", fit, fitchange);

    if(iter != 0 && fitchange < tol) {
      timings["total"] = timers["total"].seconds();
      fmt::print("\n\n === CONVERGED in {} Iterations === \n\n", iter);
      print_verification_script(X, CSFTensors, factor_matrices, coreTensor);
      break;

    }
  }
  fmt::print("======================\n");
  fmt::print("Tucker:    {} [s]\n", timings["total"]);
  fmt::print("CSF:       {} [s]\n", timings["csf"]);
  fmt::print("TTM:       {} [s]\n", timings["ttm"]);
  fmt::print("SVD:       {} [s]\n", timings["svd"]);
  fmt::print("Core:      {} [s]\n", timings["core"]);
  CUDA_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cublasDestroy(cublasH));
} 
  
