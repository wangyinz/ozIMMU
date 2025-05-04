#pragma once
#include <cutf/cublas.hpp>
#include <cutf/debug/time_breakdown.hpp>
#include <ozimmu/ozimmu.hpp>
#include <vector> 
#include <cuda_runtime.h> 

struct mtk::ozimmu::handle {
  // handlers
  // cublasHandle_t cublas_handle;
  // cudaStream_t cuda_stream;

  // Stream and Event Pools for parallel execution
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> events;
  std::vector<cublasHandle_t> cublas_handles; 

  // Main stream set by user (or default 0)
  cudaStream_t cuda_stream = 0; 

  // working memory
  void *working_memory_ptr;
  std::size_t current_working_memory_size;

  // profiling
  cutf::debug::time_breakdown::profiler profiler;

  // Malloc mode flag
  malloc_mode_t malloc_mode;

  // For auto mode
  enum { mantissa_loss_counter_length = 13 - 6 + 1 };
  unsigned long long int *d_mantissa_loss_counter_ptr;
  compute_mode_t last_auto_mode = mtk::ozimmu::dgemm;

  double avg_mantissa_loss_threshold = 0;

  std::uint32_t intercept_threshold_m;
  std::uint32_t intercept_threshold_n;
  std::uint32_t intercept_threshold_k;
};

namespace mtk {
namespace ozimmu {
cublasStatus_t cublasCreate_org(cublasHandle_t *handle_ptr);

cublasStatus_t cublasDestroy_org(cublasHandle_t handle_ptr);
} // namespace ozimmu
} // namespace mtk
