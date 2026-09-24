// MIT. Direct cuBLASLt algorithm selection for the small request experiment.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
static void checked(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLASLt status " + std::to_string(status));
}

class Plan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
 public:
  Plan(int m, int n, int k, size_t workspace_bytes) {
    try {
      checked(cublasLtCreate(&handle));
      checked(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t transpose = CUBLAS_OP_T;
      checked(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
      checked(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
      checked(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));
      checked(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
      checked(cublasLtMatmulPreferenceCreate(&preference));
      checked(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
      algorithms.resize(128);
      int count = 0;
      checked(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, c, preference, algorithms.size(), algorithms.data(), &count));
      algorithms.resize(count);
      if (!count) throw std::runtime_error("No cuBLASLt algorithms for this shape");
    } catch (...) { close(); throw; }
  }
  void close() noexcept {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (handle) cublasLtDestroy(handle);
    preference = nullptr; a = b = c = nullptr; operation = nullptr; handle = nullptr;
  }
  ~Plan() { close(); }
  Plan(const Plan&) = delete;
  int count() const { return algorithms.size(); }
  py::dict info(int index) const {
    auto &result = algorithms.at(index);
    py::dict out;
    out["workspace_bytes"] = result.workspaceSize;
    out["waves"] = result.wavesCount;
    for (auto attr : {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_STAGES_ID}) {
      int value = 0; size_t written = 0;
      checked(cublasLtMatmulAlgoConfigGetAttribute(&result.algo, attr, &value, sizeof(value), &written));
      out[py::str(std::to_string(attr))] = value;
    }
    return out;
  }
  void run(uintptr_t x, uintptr_t weight, uintptr_t output, uintptr_t workspace, size_t workspace_bytes, uintptr_t stream, int index) {
    float alpha = 1.0f, beta = 0.0f;
    const auto &algo = algorithms.at(index).algo;
    checked(cublasLtMatmul(handle, operation, &alpha, reinterpret_cast<void*>(weight), a,
      reinterpret_cast<void*>(x), b, &beta, reinterpret_cast<void*>(output), c,
      reinterpret_cast<void*>(output), c, &algo, reinterpret_cast<void*>(workspace), workspace_bytes,
      reinterpret_cast<cudaStream_t>(stream)));
  }
};

PYBIND11_MODULE(laya_frontier_lt, m) {
  py::class_<Plan>(m, "Plan")
    .def(py::init<int, int, int, size_t>())
    .def("count", &Plan::count)
    .def("info", &Plan::info)
    .def("run", &Plan::run);
}
