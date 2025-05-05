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


struct mtk::ozimmu::WorkspaceLayoutOffsets {
  // Base workspace offsets (example, adjust based on gemm_int8<T>)
  std::size_t a_max_exp_offset;
  std::size_t b_max_exp_offset;
  std::size_t a_int8_slices_offset;
  std::size_t b_int8_slices_offset;
  // ... add other base offsets (e.g., complex parts)

  std::size_t base_workspace_end_offset;

  // +++ ADDED +++ Offset for the embedded pointer array
  std::size_t pointer_array_offset; // Offset from workspace_base
  // -------------

  // Per-stream offsets (now relative to workspace_base OR could be kept relative to a later point)
  // Let's keep them relative to workspace_base for simplicity here.
  std::size_t per_stream_i32_start_offset;
  std::size_t per_stream_f64_start_offset;

  // Final reduction buffer offset (relative to workspace_base)
  std::size_t final_f64_reduction_offset;

  // +++ ADDED +++ Total size, useful for checks
  std::size_t total_calculated_size;
};

namespace mtk {
namespace ozimmu {
cublasStatus_t cublasCreate_org(cublasHandle_t *handle_ptr);

cublasStatus_t cublasDestroy_org(cublasHandle_t handle_ptr);

mtk::ozimmu::WorkspaceLayoutOffsets calculate_workspace_layout(
  std::size_t m, std::size_t n, std::size_t k,
  unsigned num_split,
  mtk::ozimmu::element_kind_t element_kind,
  std::size_t num_streams
);
} // namespace ozimmu
} // namespace mtk
