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

    // Free main working memory (contains everything now)
    if (handle->working_memory_ptr) {
      cudaError_t err = cudaFree(handle->working_memory_ptr);
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
            ozIMMU_log("Warning: cudaFree failed for working_memory_ptr during destroy: " + std::string(cudaGetErrorString(err)));
        }
      handle->working_memory_ptr = nullptr;
    }
    // Free mantissa counter
    if (handle->d_mantissa_loss_counter_ptr) {
        cudaError_t err = cudaFree(handle->d_mantissa_loss_counter_ptr);
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
            ozIMMU_log("Warning: cudaFree failed for d_mantissa_loss_counter_ptr during destroy: " + std::string(cudaGetErrorString(err)));
        }
        handle->d_mantissa_loss_counter_ptr = nullptr;
    }

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

mtk::ozimmu::WorkspaceLayoutOffsets mtk::ozimmu::calculate_workspace_layout(
  std::size_t m, std::size_t n, std::size_t k,
  unsigned num_split,
  mtk::ozimmu::element_kind_t element_kind,
  std::size_t num_streams
) {
  WorkspaceLayoutOffsets offsets;
  std::size_t current_offset = 0;

  // --- BASE WORKSPACE LAYOUT (Mirror gemm_int8<double>/complex logic) ---
  // *** This section MUST be IDENTICAL to the layout assumed in gemm_int8 ***
  if (element_kind == mtk::ozimmu::real) {
      // Layout from gemm_int8<double>
      offsets.a_max_exp_offset = current_offset;
      current_offset += m * sizeof(double);
      offsets.b_max_exp_offset = current_offset;
      current_offset += n * sizeof(double);

      const auto ld_int8_a = mtk::ozimmu::get_slice_ld<std::int8_t>(m, k, mtk::ozimmu::op_t);
      const auto ld_int8_b = mtk::ozimmu::get_slice_ld<std::int8_t>(k, n, mtk::ozimmu::op_n);
      const auto num_int8_a_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(m, k, mtk::ozimmu::op_t);
      const auto num_int8_b_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(k, n, mtk::ozimmu::op_n);

      offsets.a_int8_slices_offset = current_offset;
      std::size_t size_split_A = num_int8_a_slice_elements * num_split * sizeof(std::int8_t);
      current_offset += size_split_A;
      offsets.b_int8_slices_offset = current_offset;
      std::size_t size_split_B = num_int8_b_slice_elements * num_split * sizeof(std::int8_t);
      current_offset += size_split_B;
      // Add other base buffers if needed...
  } else { // complx
      // *** Complex layout needs careful verification ***
      offsets.a_max_exp_offset = current_offset; current_offset += m * sizeof(double); // a_real
      /* a_imag_offset = */ current_offset += m * sizeof(double); // a_imag
      offsets.b_max_exp_offset = current_offset; current_offset += n * sizeof(double); // b_real
      /* b_imag_offset = */ current_offset += n * sizeof(double); // b_imag

       const auto ld_int8_a = mtk::ozimmu::get_slice_ld<std::int8_t>(m, k, mtk::ozimmu::op_t);
       const auto ld_int8_b = mtk::ozimmu::get_slice_ld<std::int8_t>(k, n, mtk::ozimmu::op_n);
       const auto num_int8_a_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(m, k, mtk::ozimmu::op_t);
       const auto num_int8_b_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(k, n, mtk::ozimmu::op_n);

       std::size_t size_split_A_cmplx = num_int8_a_slice_elements * num_split * sizeof(std::int8_t) * 2;
       std::size_t size_split_B_cmplx = num_int8_b_slice_elements * num_split * sizeof(std::int8_t) * 2;

       offsets.a_int8_slices_offset = current_offset; current_offset += size_split_A_cmplx / 2; // a_real
       /* a_imag_int8_offset = */ current_offset += size_split_A_cmplx / 2; // a_imag
       offsets.b_int8_slices_offset = current_offset; current_offset += size_split_B_cmplx / 2; // b_real
       /* b_imag_int8_offset = */ current_offset += size_split_B_cmplx / 2; // b_imag
  }
  offsets.base_workspace_end_offset = current_offset; // Mark end of primary data

  // --- ADD SPACE FOR THE POINTER ARRAY ---
  offsets.pointer_array_offset = current_offset; // Pointer array starts here
  const std::size_t pointer_array_size = num_streams * sizeof(double*);
  current_offset += pointer_array_size;
  // ---

  // --- PER-STREAM BUFFERS ---
  const std::size_t size_per_stream_i32 = m * n * sizeof(std::int32_t);
  const std::size_t size_per_stream_f64 = m * n * sizeof(double); // Size of ONE f64 buffer

  offsets.per_stream_i32_start_offset = current_offset;
  current_offset += size_per_stream_i32 * num_streams;

  offsets.per_stream_f64_start_offset = current_offset;
  current_offset += size_per_stream_f64 * num_streams;

  offsets.final_f64_reduction_offset = current_offset;
  current_offset += size_per_stream_f64; // Add size of final reduction buffer

  offsets.total_calculated_size = current_offset; // Store total calculated size

  return offsets;
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

// --- REALLOCATE WORKING MEMORY (UPDATED) ---
std::size_t mtk::ozimmu::reallocate_working_memory(
  mtk::ozimmu::handle_t handle, const mtk::ozimmu::gemm_list_t gemm_list) {

  if (gemm_list.empty()) return 0;
  if (handle->streams.empty()) {
       ozIMMU_log("ERROR: Cannot reallocate working memory, stream pool size is zero.");
       return 0;
  }

  // --- Find max dimensions and representative parameters (Same as before) ---
  std::size_t max_m = 0;
  std::size_t max_n = 0;
  std::size_t max_k = 0;
  unsigned max_num_split = 0;
  mtk::ozimmu::element_kind_t effective_element_kind = mtk::ozimmu::real;
  mtk::ozimmu::compute_mode_t representative_mode = mtk::ozimmu::dgemm;
  for (const auto& gemm : gemm_list) {
    const auto op_A = std::get<0>(gemm);
    const auto op_B = std::get<1>(gemm);
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    const auto mode = std::get<6>(gemm);

    max_m = std::max(max_m, m);
    max_n = std::max(max_n, n);
    max_k = std::max(max_k, k);
    if (element_kind == mtk::ozimmu::complx) {
        effective_element_kind = mtk::ozimmu::complx;
    }
    if (mode >= mtk::ozimmu::fp64_int8_3 && mode <= mtk::ozimmu::fp64_int8_18) {
          const unsigned current_num_split = mtk::ozimmu::detail::get_split_config(mode)
                                            .matrix_A_split_types.size() - 1;
          max_num_split = std::max(max_num_split, current_num_split);
          representative_mode = mode;
      } else if (mode == mtk::ozimmu::dgemm) {
          representative_mode = mode;
      }
  }
  // ---

  // Calculate the definitive layout based on max dimensions
  const std::size_t num_streams = handle->streams.size();
  const WorkspaceLayoutOffsets layout = calculate_workspace_layout(
      max_m, max_n, max_k, max_num_split, effective_element_kind, num_streams
  );

  // Total required size is now directly from the layout calculation
  const std::size_t total_required_size = layout.total_calculated_size;

  // --- Perform Reallocation if Needed ---
  bool reallocated = false;
  if (total_required_size > handle->current_working_memory_size) {
      reallocated = true;
      handle->current_working_memory_size = total_required_size;
      ozIMMU_log("Reallocating working memory: " + std::to_string(total_required_size) + " B");

      // Free existing buffer
      if (handle->working_memory_ptr != nullptr) {
          if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
              CUTF_CHECK_ERROR(cudaFree(handle->working_memory_ptr));
          } else {
              CUTF_CHECK_ERROR(cudaFreeAsync(handle->working_memory_ptr, handle->cuda_stream));
          }
          handle->working_memory_ptr = nullptr;
      }

      // Alloc new single buffer
      if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
          CUTF_CHECK_ERROR(cudaMalloc(&(handle->working_memory_ptr),
                                       handle->current_working_memory_size));
      } else {
          CUTF_CHECK_ERROR(cudaMallocAsync(&(handle->working_memory_ptr),
                                            handle->current_working_memory_size,
                                            handle->cuda_stream));
           // Sync might be needed before memcpy below if using async malloc
           // CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
      }
  }

  // --- Update Embedded Device Pointer Array Contents ---
  // This needs to happen *every time* reallocate is called, as the target
  // pointers change based on max_m/max_n used for the current layout,
  // even if the total buffer size didn't change.
  if (handle->working_memory_ptr != nullptr && num_streams > 0) {
      std::vector<double*> h_per_stream_f64_ptrs(num_streams); // Host array to hold target ptrs

      std::uint8_t* workspace_base = reinterpret_cast<std::uint8_t*>(handle->working_memory_ptr);
      const std::size_t size_per_stream_f64 = max_m * max_n * sizeof(double);

      // Calculate the actual device addresses for the target f64 buffers
      for (std::size_t i = 0; i < num_streams; ++i) {
          std::size_t f64_buffer_offset_in_workspace = layout.per_stream_f64_start_offset + (i * size_per_stream_f64);
          h_per_stream_f64_ptrs[i] = reinterpret_cast<double*>(workspace_base + f64_buffer_offset_in_workspace);
      }

      // Calculate the location *within* the workspace where the pointer array resides
      double** d_pointer_array_location_in_workspace = reinterpret_cast<double**>(
          workspace_base + layout.pointer_array_offset
      );

      // Copy the host array of pointers (h_per_stream_f64_ptrs) to the
      // designated location within the device workspace (d_pointer_array_location_in_workspace)
      cudaMemcpyKind copyKind = cudaMemcpyHostToDevice; // Async typically okay
      if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
           CUTF_CHECK_ERROR(cudaMemcpy(d_pointer_array_location_in_workspace,
                                   h_per_stream_f64_ptrs.data(),
                                   num_streams * sizeof(double*),
                                   copyKind));
      } else {
           CUTF_CHECK_ERROR(cudaMemcpyAsync(d_pointer_array_location_in_workspace,
                                   h_per_stream_f64_ptrs.data(),
                                   num_streams * sizeof(double*),
                                   copyKind, handle->cuda_stream));
           // Sync might be needed after async memcpy if other streams rely on it immediately
           // CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
      }
  }

  return reallocated ? total_required_size : 0;
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
