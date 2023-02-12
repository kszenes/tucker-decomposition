#include "tucker.cuh"

void tucker_decomp(COOTensor3 &X, const Params& params) {
  std::unordered_map<std::string, CPUTimer> timers;
  timers["total"] = CPUTimer();
  timers["misc"] = CPUTimer();

  std::unordered_map<std::string, float> timings;
  timings["csf"] = 0.0f;
  timings["ttm"] = 0.0f;
  timings["svd"] = 0.0f;
  timings["core"] = 0.0f;
  timings["total"] = 0.0f;
  timings["fit"] = 0.0f;

  timers["total"].start();

  assert(X.nmodes == params.ranks.size() &&
         "Number of U sizes does not match X modes");
  auto use_gpu = true;
  cusolverDnHandle_t cusolverH;
  cublasHandle_t cublasH;
  CUDA_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cusolverDnCreate(&cusolverH));

  double core_sqnorm = 0.0;
  // Core tensor size
  auto coreSize = std::accumulate(
    params.ranks.begin(), params.ranks.end(), 1, std::multiplies{});
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
    factor_matrices.emplace_back(X.shape[mode], params.ranks[mode], "random_seed");
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
  for (; iter < params.maxiter; ++iter) {
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
        subchunk_size, use_gpu, cusolverH, cublasH,
        params.svd_routine
      );
      auto svd_time = timers["misc"].seconds();
      fmt::print("SVD completed in {} [s]\n", svd_time);
      timings["svd"] += svd_time;
    }
    timers["misc"].start();
    // auto coreTensor = contract_last_mode(
    //     CSFTensors.front(), factor_matrices,
    //     sspTensor, subchunk_size);
    auto coreTensor = contract_last_mode(
        cublasH, CSFTensors.front(), factor_matrices,
        sspTensor, subchunk_size);
    auto core_time = timers["misc"].seconds();
    timings["core"] += core_time;

    timers["misc"].start();
    core_sqnorm =  thrust::transform_reduce(
      coreTensor.begin(), coreTensor.end(),
      thrust::square{}, 0.0, thrust::plus{});
    auto time_fit = timers["misc"].seconds();
    timings["fit"] += time_fit;

    auto sqresid = original_sqnorm - core_sqnorm;
    auto resid = std::sqrt(std::max(0.0, sqresid));
    fit = 1 - (resid / std::sqrt(original_sqnorm));
    auto fitchange = std::abs(fitold - fit);
    fmt::print("fit = {}; fitchange = {}\n", fit, fitchange);
    fmt::print("Fit computed in {}\n", time_fit);

    auto iter_time = timers["iter"].seconds();
    fmt::print("\nIteration completed in {}\n", iter_time);

    if(iter != 0 && fitchange < params.tol) {
      timings["total"] = timers["total"].seconds();
      fmt::print("\n\n === CONVERGED in {} Iterations === \n\n", iter);
      if (params.print_verification) {
        print_verification_script(X, CSFTensors, factor_matrices, coreTensor);
      }
      break;

    }
  }
  fmt::print("======================\n");
  fmt::print("Tucker:    {:4.5f} [s]\n", timings["total"]);
  fmt::print("CSF:       {:4.5f} [s]\n", timings["csf"]);
  fmt::print("TTM:       {:4.5f} [s]\n", timings["ttm"]);
  fmt::print("SVD:       {:4.5f} [s]\n", timings["svd"]);
  fmt::print("Core:      {:4.5f} [s]\n", timings["core"]);
  fmt::print("Fit:       {:4.5f} [s]\n", timings["fit"]);
  CUDA_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cublasDestroy(cublasH));
} 
  
