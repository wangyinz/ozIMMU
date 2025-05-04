#include "config.hpp"
#include "handle.hpp"
#include "utils.hpp"
#include <cutf/device.hpp>
#include <stdexcept>

// Define the size of the stream/handle pool
#define OZIMMU_STREAM_POOL_SIZE 12
#define OZIMMU_SM_COUNT 33

int mtk::ozimmu::create(mtk::ozimmu::handle_t *h,
                        mtk::ozimmu::malloc_mode_t mm) {
  ozIMMU_log("Initializing ozIMMU handle");
  auto handle = (*h = new mtk::ozimmu::handle);

  // Initialize Stream, Event, and cuBLAS Handle Pools
  handle->streams.resize(OZIMMU_STREAM_POOL_SIZE);
  handle->events.resize(OZIMMU_STREAM_POOL_SIZE);
  handle->cublas_handles.resize(OZIMMU_STREAM_POOL_SIZE);

  for (int i = 0; i < OZIMMU_STREAM_POOL_SIZE; ++i) {
    // Create non-blocking streams
    CUTF_CHECK_ERROR(cudaStreamCreateWithFlags(&handle->streams[i], cudaStreamNonBlocking));
    // Create events (default flags)
    CUTF_CHECK_ERROR(cudaEventCreate(&handle->events[i]));
    // Create cuBLAS handles
    CUTF_CHECK_ERROR(cublasCreate_org(&handle->cublas_handles[i]));
    // Associate each cuBLAS handle with its stream
    CUTF_CHECK_ERROR(cublasSetStream(handle->cublas_handles[i], handle->streams[i]));
    cublasSetSmCountTarget(handle->cublas_handles[i], OZIMMU_SM_COUNT);
  }

  handle->current_working_memory_size = 0;
  handle->working_memory_ptr = nullptr;
  handle->malloc_mode = mm;
  handle->cuda_stream = 0;

  // Disable profiling by default
  mtk::ozimmu::disable_profiling(*h);

  CUTF_CHECK_ERROR(cudaMalloc(&(handle->d_mantissa_loss_counter_ptr),
                              sizeof(unsigned long long int) *
                                  handle->mantissa_loss_counter_length));

  handle->intercept_threshold_m = std::stoul(
      ozIMMU_load_env_if_defined("OZIMMU_INTERCEPT_THRESHOLD_M", "1024"));
  handle->intercept_threshold_n = std::stoul(
      ozIMMU_load_env_if_defined("OZIMMU_INTERCEPT_THRESHOLD_N", "1024"));
  handle->intercept_threshold_k = std::stoul(
      ozIMMU_load_env_if_defined("OZIMMU_INTERCEPT_THRESHOLD_K", "1024"));

  return 0;
}

int mtk::ozimmu::destroy(mtk::ozimmu::handle_t handle) {
  if (handle) {
    ozIMMU_log("Destroying ozIMMU handle");

    // Destroy Stream, Event, and cuBLAS Handle Pools
    for (int i = 0; i < OZIMMU_STREAM_POOL_SIZE; ++i) {
        if (handle->cublas_handles.size() > i && handle->cublas_handles[i]) {
            // Ensure the handle is still valid before destroying
            cublasDestroy_org(handle->cublas_handles[i]);
        }
        if (handle->events.size() > i && handle->events[i]) {
            cudaEventDestroy(handle->events[i]);
        }
        if (handle->streams.size() > i && handle->streams[i]) {
            cudaStreamDestroy(handle->streams[i]);
        }
    }
    handle->cublas_handles.clear();
    handle->events.clear();
    handle->streams.clear();

    delete handle;
    handle = nullptr;
  }

  return 0;
}

void mtk::ozimmu::set_cuda_stream(mtk::ozimmu::handle_t handle,
                                  cudaStream_t cuda_stream) {
  // Set cuda stream to cuBLAS handler
  // CUTF_CHECK_ERROR(cublasSetStream(handle->cublas_handle, cuda_stream));

  // Set ozimmu handler
  handle->cuda_stream = cuda_stream;
}

// Helper function to calculate required workspace size, including per-stream buffers
std::size_t calculate_total_working_memory(
  mtk::ozimmu::handle_t handle,
  std::size_t base_gemm_workspace_size,
  std::size_t m, std::size_t n,
  mtk::ozimmu::element_kind_t element_kind)
{
  const std::size_t num_streams = handle->streams.size();
  if (num_streams == 0) {
       throw std::runtime_error("Stream pool not initialized in handle.");
  }

  // Size for intermediate int32 results (per stream)
  const std::size_t size_per_stream_i32 = m * n * sizeof(std::int32_t);

  // Size for intermediate double accumulators (per stream)
  const std::size_t size_per_stream_f64 = m * n * sizeof(double) * (element_kind == mtk::ozimmu::real ? 1 : 2); // Double storage for complex accumulation parts if needed conceptually

  // Size for the final reduced double result (only one needed)
  const std::size_t size_final_reduction_f64 = m * n * sizeof(double) * (element_kind == mtk::ozimmu::real ? 1 : 2);

  return base_gemm_workspace_size +
         (size_per_stream_i32 * num_streams) +
         (size_per_stream_f64 * num_streams) +
         size_final_reduction_f64; // Add space for per-stream buffers and final reduction target
}

std::size_t
mtk::ozimmu::reallocate_working_memory(mtk::ozimmu::handle_t handle,
                                       const std::size_t size_in_byte) {
  if (size_in_byte > handle->current_working_memory_size) {
    handle->current_working_memory_size = size_in_byte;

    ozIMMU_log("Reallocated memory : " + std::to_string(size_in_byte) + " B");

    if (handle->working_memory_ptr != nullptr) {
      if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
        CUTF_CHECK_ERROR(cudaFree(handle->working_memory_ptr));
      } else {
        CUTF_CHECK_ERROR(
            cudaFreeAsync(handle->working_memory_ptr, handle->cuda_stream));
      }
    }

    // Realloc
    if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
      CUTF_CHECK_ERROR(cudaMalloc(&(handle->working_memory_ptr),
                                  handle->current_working_memory_size));
    } else {
      CUTF_CHECK_ERROR(cudaMallocAsync(&(handle->working_memory_ptr),
                                       handle->current_working_memory_size,
                                       handle->cuda_stream));
    }

    return size_in_byte;
  }
  return 0;
}

std::size_t mtk::ozimmu::reallocate_working_memory(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::gemm_list_t gemm_list) {
  
  if (gemm_list.empty()) {
    return 0; // Nothing to allocate for
  }
  std::size_t max_m = 0;
  std::size_t max_n = 0;
  std::size_t max_base_gemm_workspace = 0;
  mtk::ozimmu::element_kind_t effective_element_kind = mtk::ozimmu::real;

  for (const auto gemm : gemm_list) {
    const auto op_A = std::get<0>(gemm);
    const auto op_B = std::get<1>(gemm);
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    const auto mode = std::get<6>(gemm);

    // TODO: Need to verify this is necessary. It seems m n will always be the same.
    // Track max dimensions needed for per-stream buffers
    max_m = std::max(max_m, m);
    max_n = std::max(max_n, n);
    if (element_kind == mtk::ozimmu::complx) {
        effective_element_kind = mtk::ozimmu::complx;
    }

    const auto working_memory_A =
        mtk::ozimmu::detail::calculate_working_memory_size(
            op_A, m, k, mode, detail::matrix_A, element_kind);
    const auto working_memory_B =
        mtk::ozimmu::detail::calculate_working_memory_size(
            op_B, k, n, mode, detail::matrix_B, element_kind);
    // Estimate space for max_exp arrays (needs adjustment based on gemm_int8 implementation)
    // Example: (m + n) doubles for real, (m+m + n+n) doubles for complex? -> Check gemm_int8 layout
    std::size_t exp_size = (element_kind == mtk::ozimmu::real) ?
                           (m + n) * sizeof(double) :
                           (m * 2 + n * 2) * sizeof(double);
    // const auto working_memory_C_fp32 =
    //     m * n * mtk::ozimmu::get_data_size_in_byte(fp32);
    // const auto working_memory_C_fp64 =
    //     m * n * mtk::ozimmu::get_data_size_in_byte(fp64) *
    //     (element_kind == mtk::ozimmu::real ? 1 : 2);
    std::size_t etc = 0;
    // if (mode >= mtk::ozimmu::fp64_int8_3 && mode <= mtk::ozimmu::fp64_int8_18) ?
    if (mode == mtk::ozimmu::fp64_int8_3 || mode == mtk::ozimmu::fp64_int8_4 ||
        mode == mtk::ozimmu::fp64_int8_5 || mode == mtk::ozimmu::fp64_int8_6 ||
        mode == mtk::ozimmu::fp64_int8_7 || mode == mtk::ozimmu::fp64_int8_8 ||
        mode == mtk::ozimmu::fp64_int8_9 || mode == mtk::ozimmu::fp64_int8_10 ||
        mode == mtk::ozimmu::fp64_int8_11 ||
        mode == mtk::ozimmu::fp64_int8_12 ||
        mode == mtk::ozimmu::fp64_int8_13 ||
        mode == mtk::ozimmu::fp64_int8_14 ||
        mode == mtk::ozimmu::fp64_int8_15 ||
        mode == mtk::ozimmu::fp64_int8_16 ||
        mode == mtk::ozimmu::fp64_int8_17 ||
        mode == mtk::ozimmu::fp64_int8_18) {
      etc = (m + n) * mtk::ozimmu::get_data_size_in_byte(fp64) *
            (element_kind == mtk::ozimmu::real ? 1 : 2);
    }
    // Accumulate base workspace size (excluding per-stream buffers)
    // IMPORTANT: This calculation MUST match the layout used in gemm_int8 *before* the per-stream buffers.
    // The original calculation might need refinement. Let's use a simplified placeholder based on original code.
    // Placeholder: Assuming working_memory_A/B cover the int8 slices and exp_size covers the max_exp arrays.
    // Revisit this calculation based on the final gemm_int8<T> implementation layout.
    std::size_t current_base_size = working_memory_A + working_memory_B + exp_size + etc;

    max_base_gemm_workspace = std::max(max_base_gemm_workspace, current_base_size);
  }

    // Calculate total size including per-stream buffers based on max M/N
    const std::size_t total_required_size = calculate_total_working_memory(
      handle, max_base_gemm_workspace, max_m, max_n, effective_element_kind);

  // Now perform the reallocation if needed
  if (total_required_size > handle->current_working_memory_size) {
    handle->current_working_memory_size = total_required_size;

    ozIMMU_log("Reallocated memory for GEMM list: " + std::to_string(total_required_size) + " B");

    // Free existing buffer (using cuda_stream if async)
    if (handle->working_memory_ptr != nullptr) {
      if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
        CUTF_CHECK_ERROR(cudaFree(handle->working_memory_ptr));
      } else {
        CUTF_CHECK_ERROR(
            cudaFreeAsync(handle->working_memory_ptr, handle->cuda_stream));
         // Sync needed before reallocation if using async free
         // CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
      }
    }

    // Alloc new buffer (using cuda_stream if async)
    if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
      CUTF_CHECK_ERROR(cudaMalloc(&(handle->working_memory_ptr),
                                   handle->current_working_memory_size));
    } else {
      CUTF_CHECK_ERROR(cudaMallocAsync(&(handle->working_memory_ptr),
                                        handle->current_working_memory_size,
                                        handle->cuda_stream));
        // Sync needed before use if using async malloc
       // CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
    }

    return total_required_size;
  }
  return 0; // No reallocation needed or occurred

}

std::string
mtk::ozimmu::get_compute_mode_name_str(const mtk::ozimmu::compute_mode_t mode) {
  switch (mode) {
  case mtk::ozimmu::sgemm:
    return "sgemm";
  case mtk::ozimmu::dgemm:
    return "dgemm";
  case mtk::ozimmu::fp64_int8_3:
    return "fp64_int8_3";
  case mtk::ozimmu::fp64_int8_4:
    return "fp64_int8_4";
  case mtk::ozimmu::fp64_int8_5:
    return "fp64_int8_5";
  case mtk::ozimmu::fp64_int8_6:
    return "fp64_int8_6";
  case mtk::ozimmu::fp64_int8_7:
    return "fp64_int8_7";
  case mtk::ozimmu::fp64_int8_8:
    return "fp64_int8_8";
  case mtk::ozimmu::fp64_int8_9:
    return "fp64_int8_9";
  case mtk::ozimmu::fp64_int8_10:
    return "fp64_int8_10";
  case mtk::ozimmu::fp64_int8_11:
    return "fp64_int8_11";
  case mtk::ozimmu::fp64_int8_12:
    return "fp64_int8_12";
  case mtk::ozimmu::fp64_int8_13:
    return "fp64_int8_13";
  case mtk::ozimmu::fp64_int8_14:
    return "fp64_int8_14";
  case mtk::ozimmu::fp64_int8_15:
    return "fp64_int8_15";
  case mtk::ozimmu::fp64_int8_16:
    return "fp64_int8_16";
  case mtk::ozimmu::fp64_int8_17:
    return "fp64_int8_17";
  case mtk::ozimmu::fp64_int8_18:
    return "fp64_int8_18";
  case mtk::ozimmu::fp64_int8_auto:
    return "fp64_int8_auto";
  default:
    break;
  }
  OZIMMU_NOT_IMPLEMENTED;
  return "";
}

mtk::ozimmu::data_t
mtk::ozimmu::get_output_type(const mtk::ozimmu::compute_mode_t compute_mode) {
  switch (compute_mode) {
  case mtk::ozimmu::sgemm:
    return mtk::ozimmu::fp32;

  case mtk::ozimmu::fp64_int8_4:
  case mtk::ozimmu::fp64_int8_3:
  case mtk::ozimmu::fp64_int8_5:
  case mtk::ozimmu::fp64_int8_6:
  case mtk::ozimmu::fp64_int8_7:
  case mtk::ozimmu::fp64_int8_8:
  case mtk::ozimmu::fp64_int8_9:
  case mtk::ozimmu::fp64_int8_10:
  case mtk::ozimmu::fp64_int8_11:
  case mtk::ozimmu::fp64_int8_12:
  case mtk::ozimmu::fp64_int8_13:
  case mtk::ozimmu::fp64_int8_14:
  case mtk::ozimmu::fp64_int8_15:
  case mtk::ozimmu::fp64_int8_16:
  case mtk::ozimmu::fp64_int8_17:
  case mtk::ozimmu::fp64_int8_18:
  case mtk::ozimmu::fp64_int8_auto:
  case mtk::ozimmu::dgemm:
    return mtk::ozimmu::fp64;

  default:
    break;
  }
  OZIMMU_NOT_IMPLEMENTED;
  return mtk::ozimmu::original;
}

std::size_t mtk::ozimmu::get_data_size_in_byte(const mtk::ozimmu::data_t d) {
  switch (d) {
  case mtk::ozimmu::fp64:
    return 8;
  case mtk::ozimmu::fp32:
    return 4;
  case mtk::ozimmu::fp16:
    return 2;
  case mtk::ozimmu::original:
    return 0;
  case mtk::ozimmu::int8:
    return 1;
  default:
    OZIMMU_NOT_IMPLEMENTED;
    break;
  }
  return 0;
}

void mtk::ozimmu::enable_profiling(mtk::ozimmu::handle_t handle) {
  handle->profiler.enable_measurement();
}

void mtk::ozimmu::disable_profiling(mtk::ozimmu::handle_t handle) {
  handle->profiler.disable_measurement();
}

void mtk::ozimmu::print_profiler_result(mtk::ozimmu::handle_t handle,
                                        const std::string tag, const bool csv) {
  if (!csv) {
    handle->profiler.print_result(tag);
  } else {
    handle->profiler.print_result_csv(tag);
  }
}

void mtk::ozimmu::clear_profiler_result(mtk::ozimmu::handle_t handle) {
  handle->profiler.clear();
}

void mtk::ozimmu::set_auto_mantissa_loss_threashold(
    mtk::ozimmu::handle_t handle, const double threshold) {
  handle->avg_mantissa_loss_threshold = threshold;
}

double get_auto_mantissa_loss_threashold(mtk::ozimmu::handle_t handle) {
  return handle->avg_mantissa_loss_threshold;
}
