#include "config.hpp"
#include "handle.hpp"
#include "split.hpp"
#include "utils.hpp"
#include <cutf/cublas.hpp>

namespace {

// Kernel to sum results from multiple per-stream buffers
// Assumes per_stream_buffers is an array of pointers to the start of each buffer
// Writes the sum to output_buffer
template <class T>
__global__ void reduce_sum_kernel(T** per_stream_buffers,
                                  T* output_buffer,
                                  std::size_t num_buffers,
                                  std::size_t elements_per_buffer) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements_per_buffer) {
        return;
    }

    T sum = 0;
    for (std::size_t i = 0; i < num_buffers; ++i) {
        // Check if the buffer pointer is valid before dereferencing
        if (per_stream_buffers[i] != nullptr) {
             sum += per_stream_buffers[i][tid];
        }
    }
    output_buffer[tid] = sum;
}

// Host function to launch the reduction kernel
template <class T>
void reduce_sum_buffers(T** d_per_stream_buffer_ptrs, // Device pointer to array of device pointers
                        T* d_output_buffer,
                        std::size_t num_buffers,
                        std::size_t elements_per_buffer,
                        cudaStream_t stream) {
    if (num_buffers == 0 || elements_per_buffer == 0) return;

    constexpr std::size_t block_size = 256;
    reduce_sum_kernel<T><<<(elements_per_buffer + block_size - 1) / block_size, block_size, 0, stream>>>(
        d_per_stream_buffer_ptrs,
        d_output_buffer,
        num_buffers,
        elements_per_buffer);
    CUTF_CHECK_ERROR(cudaGetLastError()); // Check for launch errors
}

template <class T>
std::size_t split_core(void *const split_ptr, const mtk::ozimmu::operation_t op,
                       const std::size_t m, const std::size_t n,
                       const T *const src_ptr, const std::size_t ld,
                       const std::vector<mtk::ozimmu::data_t> data_type_list,
                       const mtk::ozimmu::detail::matrix_t matrix,
                       const T *const two_to_alpha_ptr,
                       cudaStream_t cuda_stream) {
  const auto num_split = data_type_list.size() - 1;
  std::size_t offset = 0;

  if (num_split <= 1) {
    // Do nothing
  } else {
    OZIMMU_NOT_IMPLEMENTED;
  }

  return offset;
}

//=====
// This function is added
//=====
void split_AB_int8_nearest(
	mtk::ozimmu::handle_t handle,
	const mtk::ozimmu::operation_t op_A,
	const mtk::ozimmu::operation_t op_B,
	const std::size_t m,
	const std::size_t n,
	const std::size_t k,
	const double* const a_ptr, const std::size_t lda,
	double* const sft_a,
	std::int8_t* const working_a_ptr, const std::uint32_t ld_int8_a,
	const double* const b_ptr, const std::size_t ldb,
	double* const sft_b,
	std::int8_t* const working_b_ptr, const std::uint32_t ld_int8_b,
	const std::int8_t num_split,
	const std::int8_t bits
) {
	handle->profiler.start_timer_sync("split_A_near");
	mtk::ozimmu::split_int8_nearest(
			working_a_ptr, ld_int8_a,
			sft_a,
			m, k,
			a_ptr, lda,
			op_A,
			mtk::ozimmu::detail::matrix_A,
			num_split,
			bits,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_A_near");

	handle->profiler.start_timer_sync("split_B_near");
	mtk::ozimmu::split_int8_nearest(
			working_b_ptr, ld_int8_b,
			sft_b,
			k, n,
			b_ptr, ldb,
			op_B,
			mtk::ozimmu::detail::matrix_B,
			num_split,
			bits,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_B_near");
}

template <class T>
void split_AB_int8(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const T *const a_ptr,
    const std::size_t lda, double *const a_max_exp_ptr,
    std::int8_t *const working_a_ptr, const std::uint32_t ld_int8_a,
    const T *const b_ptr, const std::size_t ldb, double *const b_max_exp_ptr,
    std::int8_t *const working_b_ptr, const std::uint32_t ld_int8_b,
    const unsigned num_split, const unsigned bits_per_int8) {
  handle->profiler.start_timer_sync("split_A");
  mtk::ozimmu::split_int8<T>(working_a_ptr, ld_int8_a, a_max_exp_ptr, m, k,
                             a_ptr, lda, op_A, mtk::ozimmu::detail::matrix_A,
                             num_split, bits_per_int8, handle->cuda_stream);
  handle->profiler.stop_timer_sync("split_A");

  handle->profiler.start_timer_sync("split_B");
  mtk::ozimmu::split_int8<T>(working_b_ptr, ld_int8_b, b_max_exp_ptr, k, n,
                             b_ptr, ldb, op_B, mtk::ozimmu::detail::matrix_B,
                             num_split, bits_per_int8, handle->cuda_stream);
  handle->profiler.stop_timer_sync("split_B");
}

cudaDataType_t to_cudaDataType_t(const mtk::ozimmu::data_t d) {
  switch (d) {
  case mtk::ozimmu::fp32:
    return CUDA_R_32F;
  case mtk::ozimmu::fp16:
    return CUDA_R_16F;
  default:
    break;
  }
  OZIMMU_NOT_IMPLEMENTED;
  return CUDA_R_32F;
}

cublasOperation_t to_cublasOperation_t(const mtk::ozimmu::operation_t op) {
  switch (op) {
  case mtk::ozimmu::op_n:
    return CUBLAS_OP_N;
  case mtk::ozimmu::op_t:
    return CUBLAS_OP_T;
  default:
    break;
  }
  OZIMMU_NOT_IMPLEMENTED;
  return CUBLAS_OP_N;
}

//=====
// This function is added
//=====
__global__ void accumulate_in_f64_kernel_2(
	const std::size_t m,
	double* const f64_ptr,
	const std::int32_t* i32_ptr,
	const std::size_t length,
	const double* const sft_a,
	const double* const sft_b,
	const double scale
) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}

	const auto mi = tid % m;
	const auto ni = tid / m;
	f64_ptr[tid] +=  static_cast<double>(i32_ptr[tid])*sft_a[mi]*sft_b[ni]*scale;
}

//=====
// This function is added
//=====
void accumulate_in_f64_2(
	const std::size_t m,
	double* const f64_ptr,
	const std::int32_t* i32_ptr,
	const std::size_t length,
	const double* const sft_a,
	const double* const sft_b,
	const double sft,
	cudaStream_t cuda_stream
) {
	constexpr std::size_t block_size = 256;
	accumulate_in_f64_kernel_2
		<<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				m,
				f64_ptr,
				i32_ptr,
				length,
				sft_a,
				sft_b,
				sft
			);
}

//=====
// This function is added
//=====
__global__ void axby_kernel_2(
	const std::size_t m,
	const std::size_t n,
	const double a,
	const double* const x_ptr,
	const double b,
	double* const y_ptr,
	const std::size_t ldy
	) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m * n) {
    return;
  }

  const auto mi = tid % m;
  const auto ni = tid / m;
  const auto memory_index = ni * ldy + mi;

  if (b != 0) {
    y_ptr[memory_index] = a * x_ptr[tid] + b * y_ptr[memory_index];
  } else {
    y_ptr[memory_index] = a * x_ptr[tid];
  }
}

//=====
// This function is added
//=====
void axby_2(
	const std::size_t m,
	const std::size_t n,
	const double a,
	const double* const x_ptr,
	const double b,
	double* const y_ptr,
	const std::size_t ldy,
	cudaStream_t cuda_stream
	) {
  constexpr std::size_t block_size = 256;
  axby_kernel_2
    <<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
        m, n,
        a,
        x_ptr,
        b,
        y_ptr, ldy
      );
}

__global__ void accumulate_in_f64_kernel(double *const f64_ptr,
                                         const std::int32_t *i32_ptr,
                                         const std::size_t length,
                                         const double scale) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length) {
    return;
  }

  f64_ptr[tid] +=
      static_cast<double>(static_cast<std::int64_t>(i32_ptr[tid]) << 32) *
      scale;
}

void accumulate_in_f64(double *const f64_ptr, const std::int32_t *i32_ptr,
                       const std::size_t length,
                       const std::int32_t mantissa_rshift,
                       cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;
  const auto scale = cutf::experimental::fp::reinterpret_as_fp(
      static_cast<std::uint64_t>(
          (cutf::experimental::fp::get_bias<double>() - mantissa_rshift))
      << cutf::experimental::fp::get_mantissa_size<double>());
  accumulate_in_f64_kernel<<<(length + block_size - 1) / block_size, block_size,
                             0, cuda_stream>>>(f64_ptr, i32_ptr, length, scale);
}

template <class T>
__global__ void init_accumulator_buffer_kernel(T *const dp_ptr,
                                               const std::size_t length) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length) {
    return;
  }

  dp_ptr[tid] = 0;
}

template <class T>
void init_accumulator_buffer(T *const dp_ptr, const std::size_t length,
                             cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;
  init_accumulator_buffer_kernel<T>
      <<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
          dp_ptr, length);
}

__global__ void axby_kernel(const std::size_t m, const std::size_t n,
                            const double a, const double *const x_ptr,
                            const double b, double *const y_ptr,
                            const std::size_t ldy,
                            const double *const a_max_exp_ptr,
                            const double *const b_max_exp_ptr) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m * n) {
    return;
  }

  const auto mi = tid % m;
  const auto ni = tid / m;

  const auto memory_index = ni * ldy + mi;

  const auto x =
      x_ptr[tid] / (1l << 44) * a_max_exp_ptr[mi] * b_max_exp_ptr[ni];

  if (b != 0) {
    y_ptr[memory_index] = a * x + b * y_ptr[memory_index];
  } else {
    y_ptr[memory_index] = a * x;
  }
}

void axby(const std::size_t m, const std::size_t n, const double a,
          const double *const x_ptr, const double b, double *const y_ptr,
          const std::size_t ldy, const double *const a_max_exp_ptr,
          const double *const b_max_exp_ptr, cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;
  axby_kernel<<<(m * n + block_size - 1) / block_size, block_size, 0,
                cuda_stream>>>(m, n, a, x_ptr, b, y_ptr, ldy, a_max_exp_ptr,
                               b_max_exp_ptr);
}

__global__ void axy_complex_kernel(const std::size_t m, const std::size_t n,
                                   const cuDoubleComplex a,
                                   const double *const x_ptr,
                                   cuDoubleComplex *const y_ptr,
                                   const std::size_t ldy,
                                   const double *const a_max_exp_ptr,
                                   const double *const b_max_exp_ptr) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m * n) {
    return;
  }

  const auto mi = tid % m;
  const auto ni = tid / m;

  const auto memory_index = ni * ldy + mi;

  const auto x =
      x_ptr[tid] / (1l << 44) * a_max_exp_ptr[mi] * b_max_exp_ptr[ni];

  auto y = y_ptr[memory_index];

  y.x = a.x * x + y.x;
  y.y = a.y * x + y.y;

  y_ptr[memory_index] = y;
}

void axy_complex(const std::size_t m, const std::size_t n,
                 const cuDoubleComplex a, const double *const x_ptr,
                 cuDoubleComplex *const y_ptr, const std::size_t ldy,
                 const double *const a_max_exp_ptr,
                 const double *const b_max_exp_ptr, cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;
  axy_complex_kernel<<<(m * n + block_size - 1) / block_size, block_size, 0,
                       cuda_stream>>>(m, n, a, x_ptr, y_ptr, ldy, a_max_exp_ptr,
                                      b_max_exp_ptr);
}

template <bool is_beta_zero>
__global__ void init_c_complex_kernel(const std::size_t m, const std::size_t n,
                                      cuDoubleComplex *const c_ptr,
                                      const std::size_t ldc,
                                      const cuDoubleComplex beta) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m * n) {
    return;
  }

  const auto mi = tid % m;
  const auto ni = tid / m;

  const auto memory_index = ni * ldc + mi;

  if (is_beta_zero) {
    c_ptr[memory_index] = make_cuDoubleComplex(0, 0);
  } else {
    auto c = c_ptr[memory_index];
    c.x = c.x * beta.x - c.y * beta.y;
    c.y = c.y * beta.x + c.x * beta.y;

    c_ptr[memory_index] = c;
  }
}

void init_c_complex(const std::size_t m, const std::size_t n,
                    cuDoubleComplex *const c_ptr, const std::size_t ldc,
                    const cuDoubleComplex beta, cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;

  if (beta.x == 0 && beta.y == 0) {
    init_c_complex_kernel<true>
        <<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
            m, n, c_ptr, ldc, beta);
  } else {
    init_c_complex_kernel<false>
        <<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
            m, n, c_ptr, ldc, beta);
  }
}

cublasStatus_t cublasGemmEx_org(cublasHandle_t handle, cublasOperation_t transa,
                                cublasOperation_t transb, int m, int n, int k,
                                const void *alpha, const void *A,
                                cudaDataType_t Atype, int lda, const void *B,
                                cudaDataType_t Btype, int ldb, const void *beta,
                                void *C, cudaDataType_t Ctype, int ldc,
                                cublasComputeType_t computeType,
                                cublasGemmAlgo_t algo) {
  const std::string cublas_library_name = "libcublas.so";
  const std::string cublas_function_name = "cublasGemmEx";
  cublasStatus_t (*func_ptr)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const void *, const void *, cudaDataType_t, int, const void *,
      cudaDataType_t, int, const void *, void *, cudaDataType_t, int,
      cublasComputeType_t, cublasGemmAlgo_t);
  *(void **)(&func_ptr) = ozIMMU_get_function_pointer(
    cublas_function_name.c_str(), cublas_library_name.c_str());

  const auto res =
      (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
                  Btype, ldb, beta, C, Ctype, ldc, computeType, algo);

  return res;
}

void matmul_core(
    mtk::ozimmu::handle_t handle, // Keep handle for profiler access etc.
    cublasHandle_t cublas_handle, // Pass specific cuBLAS handle
    cudaStream_t stream,          // Pass specific stream for potential internal use (though cublas uses handle's stream)
    const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const void *const a_ptr,
    const std::size_t lda, const mtk::ozimmu::data_t type_a,
    const void *const b_ptr, const std::size_t ldb,
    const mtk::ozimmu::data_t type_b, const int beta_i, void *const c_ptr,
    const mtk::ozimmu::detail::gemm_pair_config_t &gemm_pair_config,
    const mtk::ozimmu::compute_mode_t compute_mode,
    const void *const a_working_memory_ptr, const std::size_t ld_w_a,
    const void *const b_working_memory_ptr, const std::size_t ld_w_b) {
  const auto gemm_mode = gemm_pair_config.gemm_mode;
  const auto split_config = mtk::ozimmu::detail::get_split_config(compute_mode);
  const auto lda_r = gemm_pair_config.A_id == 0 ? lda : ld_w_a;
  const auto ldb_r = gemm_pair_config.B_id == 0 ? ldb : ld_w_b;

  const auto num_int8_a_slice_elements = ld_w_a * m;
  const auto num_int8_b_slice_elements = ld_w_b * n;

  std::size_t A_working_ptr_offset = 0;
  for (unsigned i = 0; i < gemm_pair_config.A_id; i++) {
    const auto t = split_config.matrix_A_split_types[i];
    A_working_ptr_offset +=
        num_int8_a_slice_elements * mtk::ozimmu::get_data_size_in_byte(t);
  }

  std::size_t B_working_ptr_offset = 0;
  for (unsigned i = 0; i < gemm_pair_config.B_id; i++) {
    const auto t = split_config.matrix_B_split_types[i];
    B_working_ptr_offset +=
        num_int8_b_slice_elements * mtk::ozimmu::get_data_size_in_byte(t);
  }

  const void *const a_working_ptr =
      reinterpret_cast<const std::uint8_t *>(a_working_memory_ptr) +
      A_working_ptr_offset;
  const void *const b_working_ptr =
      reinterpret_cast<const std::uint8_t *>(b_working_memory_ptr) +
      B_working_ptr_offset;

  const void *const a_ptr_r =
      gemm_pair_config.A_id == 0 ? a_ptr : a_working_ptr;
  const void *const b_ptr_r =
      gemm_pair_config.B_id == 0 ? b_ptr : b_working_ptr;
  void *const c_ptr_r = c_ptr;

  const auto profile_label = mtk::ozimmu::detail::gemm_mode_str(gemm_mode);
  handle->profiler.start_timer_sync(profile_label);
  switch (gemm_mode) {
  case mtk::ozimmu::detail::int8tc: {
    const int alpha_i = 1;
    const auto op_A_r =
        gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
    const auto op_B_r =
        gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;

    CUTF_CHECK_ERROR_M(
        cublasGemmEx_org(cublas_handle, op_A_r, op_B_r, m, n, k,
                         &alpha_i, a_ptr_r, CUDA_R_8I, lda_r, b_ptr_r,
                         CUDA_R_8I, ldb_r, &beta_i, c_ptr_r, CUDA_R_32I, m,
                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        ("GemmEx(int8)-m" + std::to_string(m) + "-n" + std::to_string(n) +
         "-k" + std::to_string(k)));
  } break;
  default:
    OZIMMU_NOT_IMPLEMENTED;
  }
  handle->profiler.stop_timer_sync(profile_label);
}

template <class T>
int gemm_int8(mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
              const mtk::ozimmu::operation_t op_B, const std::size_t m,
              const std::size_t n, const std::size_t k, const T *alpha,
              const T *const a_ptr, const std::size_t lda, const T *const b_ptr,
              const std::size_t ldb, const T *beta, T *const c_ptr,
              std::size_t ldc, const mtk::ozimmu::compute_mode_t compute_mode);

//=====
// This function is changed
//=====
template <>
int gemm_int8<double>(mtk::ozimmu::handle_t handle,
                      const mtk::ozimmu::operation_t op_A,
                      const mtk::ozimmu::operation_t op_B, const std::size_t m,
                      const std::size_t n, const std::size_t k,
                      const double *alpha, const double *const a_ptr,
                      const std::size_t lda, const double *const b_ptr,
                      const std::size_t ldb, const double *beta,
                      double *const c_ptr, std::size_t ldc,
                      const mtk::ozimmu::compute_mode_t compute_mode) {
  const std::size_t num_streams = handle->streams.size();

  const auto split_config_obj = mtk::ozimmu::detail::get_split_config(compute_mode);
  const std::int8_t num_split = mtk::ozimmu::detail::get_split_config(compute_mode)
                                  .matrix_A_split_types.size() -
                              1;
  const std::int8_t bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);

  const mtk::ozimmu::WorkspaceLayoutOffsets layout = calculate_workspace_layout(
    m, n, k, num_split, mtk::ozimmu::real, num_streams
  );

  // Part 1: Base workspace (split A/B, max_exp arrays)
  // IMPORTANT: Recalculate the exact size needed here, matching reallocate_working_memory's base calculation.
  const auto ld_int8_a = mtk::ozimmu::get_slice_ld<std::int8_t>(m, k, mtk::ozimmu::op_t);
  const auto ld_int8_b = mtk::ozimmu::get_slice_ld<std::int8_t>(k, n, mtk::ozimmu::op_n);
  const auto num_int8_a_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(m, k, mtk::ozimmu::op_t);
  const auto num_int8_b_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(k, n, mtk::ozimmu::op_n);

  // Base Workspace Pointers (Order and size must be precise)
  std::uint8_t *workspace_base = reinterpret_cast<std::uint8_t*>(handle->working_memory_ptr);

  // Base Workspace Pointers
  double* a_max_exp_ptr = reinterpret_cast<double*>(workspace_base + layout.a_max_exp_offset);
  double* b_max_exp_ptr = reinterpret_cast<double*>(workspace_base + layout.b_max_exp_offset);
  std::int8_t* a_int8_slices_ptr = reinterpret_cast<std::int8_t*>(workspace_base + layout.a_int8_slices_offset);
  std::int8_t* b_int8_slices_ptr = reinterpret_cast<std::int8_t*>(workspace_base + layout.b_int8_slices_offset);

  // +++ Calculate pointer TO the embedded pointer array +++
  double** d_per_stream_f64_pointers_array_in_workspace = reinterpret_cast<double**>(
      workspace_base + layout.pointer_array_offset
  );
  
  // Part 2: Per-stream workspace
  const std::size_t size_per_stream_i32 = m * n * sizeof(std::int32_t);
  const std::size_t size_per_stream_f64 = m * n * sizeof(double);
  std::vector<std::int32_t*> d_per_stream_c_i32_ptrs(num_streams);
  std::vector<double*> d_per_stream_c_f64_ptrs(num_streams);

  for (std::size_t i = 0; i < num_streams; ++i) {
        d_per_stream_c_i32_ptrs[i] = reinterpret_cast<std::int32_t*>(
            workspace_base + layout.per_stream_i32_start_offset + (i * size_per_stream_i32)
        );
        d_per_stream_c_f64_ptrs[i] = reinterpret_cast<double*>(
            workspace_base + layout.per_stream_f64_start_offset + (i * size_per_stream_f64)
        );
  }
  // Final reduction buffer pointer
  double* d_final_reduced_f64_ptr = reinterpret_cast<double*>(
    workspace_base + layout.final_f64_reduction_offset
  );
  // --- End Workspace Layout ---

  // --- Setup Phase (on cuda_stream) ---
  cudaEvent_t split_done_event, init_done_event;
  CUTF_CHECK_ERROR(cudaEventCreate(&split_done_event));
  CUTF_CHECK_ERROR(cudaEventCreate(&init_done_event));

  split_AB_int8_nearest(
    handle,
    op_A,
    op_B,
    m, n, k, a_ptr, lda,
    a_max_exp_ptr,
    a_int8_slices_ptr, ld_int8_a,
    b_ptr, ldb,
    b_max_exp_ptr,
    b_int8_slices_ptr, ld_int8_b,
    num_split,
    bits_per_int8
    );
  // Record event when split is done on its stream
  CUTF_CHECK_ERROR(cudaEventRecord(split_done_event, handle->cuda_stream));

  // Initialize ALL per-stream f64 buffers using the STARTING pointer
  if (!d_per_stream_c_f64_ptrs.empty()) {
    init_accumulator_buffer<double>(
        d_per_stream_c_f64_ptrs[0], // Pointer to the very first f64 buffer
        m * n * num_streams,        // Total elements in all f64 buffers
        handle->cuda_stream
    );
  }
  CUTF_CHECK_ERROR(cudaEventRecord(init_done_event, handle->cuda_stream));
 
  const auto &gemm_pair_config_list =
      mtk::ozimmu::detail::get_split_config(compute_mode).gemm_pair_config_list;

  std::size_t gemm_pair_config_list_N = gemm_pair_config_list.size();
  for (size_t i = 0; i < gemm_pair_config_list_N; ++i) {
    const auto& gemm_pair_config = gemm_pair_config_list[i];
    const std::size_t stream_idx = i % num_streams;
    cudaStream_t current_stream = handle->streams[stream_idx];
    cudaEvent_t current_event = handle->events[stream_idx]; // Use the pre-allocated events

    // Make current stream wait for setup phase
    if (i < num_streams) {
      CUTF_CHECK_ERROR(cudaStreamWaitEvent(current_stream, split_done_event, 0));
      CUTF_CHECK_ERROR(cudaStreamWaitEvent(current_stream, init_done_event, 0));
    }

    std::int32_t* current_c_i32_ptr = d_per_stream_c_i32_ptrs[stream_idx];
    double* current_c_f64_ptr = d_per_stream_c_f64_ptrs[stream_idx];

    // Call matmul_core (passes k, uses beta_i = 0)
    matmul_core(handle, /* stream_idx if multi-handle, */ handle->cublas_handles[stream_idx], current_stream,
                op_A, op_B, m, n, ld_int8_a, // Pass original k
                a_ptr, lda, mtk::ozimmu::fp64, b_ptr, ldb, mtk::ozimmu::fp64,
                0, // beta_i = 0 for int32 output buffer
                current_c_i32_ptr, // Output: Per-stream int32 buffer
                gemm_pair_config, compute_mode,
                a_int8_slices_ptr, // Input: Shared split data
                ld_int8_a,
                b_int8_slices_ptr, // Input: Shared split data
                ld_int8_b);

    // Accumulate the int32 result into the per-stream double buffer
    const double scale = std::ldexp(1.0, -bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2));

    accumulate_in_f64_2(
        m,
        current_c_f64_ptr, // Accumulate here
        current_c_i32_ptr, // Read from here
        m * n,
        a_max_exp_ptr,     // Shared scaling factors
        b_max_exp_ptr,     // Shared scaling factors
        scale,             // Scale for this specific split pair
        current_stream);   // Run on the current stream

    // Record event when this stream's work for this iteration is done
    if (i + num_streams >= gemm_pair_config_list_N) {
      CUTF_CHECK_ERROR(cudaEventRecord(current_event, current_stream));
    }
  }

  // --- Synchronization and Final Reduction/Copy (on cuda_stream) ---
  // Alternative, potentially cleaner wait:
  for (size_t i = 0; i < std::min(num_streams, gemm_pair_config_list.size()); ++i) {
     CUTF_CHECK_ERROR(cudaStreamWaitEvent(handle->cuda_stream, handle->events[i], 0));
  }


  // Reduce results from all per-stream double buffers
  reduce_sum_buffers<double>(
      d_per_stream_f64_pointers_array_in_workspace, // Device array of pointers
      d_final_reduced_f64_ptr,     // Output buffer
      num_streams,                 // Number of buffers to reduce
      m * n,                       // Elements per buffer
      handle->cuda_stream);        // Perform reduction on the user stream

  // Final step: alpha * reduced_result + beta * C -> C
  axby_2(
      m, n,
      *alpha,
      d_final_reduced_f64_ptr, // Input is the reduced sum
      *beta,
      c_ptr, ldc,              // Final output matrix
      handle->cuda_stream);    // Run on the user stream

  // --- Cleanup ---
  // Destroy temporary events
  CUTF_CHECK_ERROR(cudaEventDestroy(split_done_event));
  CUTF_CHECK_ERROR(cudaEventDestroy(init_done_event));

  // Optional: Synchronize the final stream if caller expects completion on return
  // CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));

  return 0;
}

template <>
int gemm_int8<cuDoubleComplex>(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *const a_ptr, const std::size_t lda,
    const cuDoubleComplex *const b_ptr, const std::size_t ldb,
    const cuDoubleComplex *beta, cuDoubleComplex *const c_ptr, std::size_t ldc,
    const mtk::ozimmu::compute_mode_t compute_mode) {
  // --- Basic Setup & Workspace Calculation ---
  const std::size_t num_streams = handle->streams.size();
    if (num_streams == 0) {
      ozIMMU_log("ERROR: Stream pool size is zero.");
      return 1; // Or throw an exception
  }
  const unsigned num_split = mtk::ozimmu::detail::get_split_config(compute_mode)
                                  .matrix_A_split_types.size() - 1;
  const int32_t bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);
  const auto &gemm_pair_config_list =
      mtk::ozimmu::detail::get_split_config(compute_mode).gemm_pair_config_list;

  // --- Workspace Layout Calculation ---
  // Complex case needs more careful layout calculation
  const auto ld_int8_a = mtk::ozimmu::get_slice_ld<std::int8_t>(m, k, mtk::ozimmu::op_t);
  const auto ld_int8_b = mtk::ozimmu::get_slice_ld<std::int8_t>(k, n, mtk::ozimmu::op_n);
  const auto num_int8_a_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(m, k, mtk::ozimmu::op_t);
  const auto num_int8_b_slice_elements = mtk::ozimmu::get_slice_num_elements<std::int8_t>(k, n, mtk::ozimmu::op_n);

  // Base Workspace: split A (real/imag), split B (real/imag), max_exp (a_real/imag, b_real/imag)
  std::size_t size_split_A_cmplx = num_int8_a_slice_elements * num_split * sizeof(std::int8_t) * 2; // Real + Imag parts
  std::size_t size_split_B_cmplx = num_int8_b_slice_elements * num_split * sizeof(std::int8_t) * 2;
  std::size_t size_max_exp_cmplx = (m * 2 + n * 2) * sizeof(double); // a_real, a_imag, b_real, b_imag

  std::size_t base_workspace_offset = 0;
  std::uint8_t *base_workspace_ptr = reinterpret_cast<std::uint8_t*>(handle->working_memory_ptr);

  // Max exponents pointers
  double *const a_real_max_exp_ptr = reinterpret_cast<double*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += m * sizeof(double);
  double *const a_imag_max_exp_ptr = reinterpret_cast<double*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += m * sizeof(double);
  double *const b_real_max_exp_ptr = reinterpret_cast<double*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += n * sizeof(double);
  double *const b_imag_max_exp_ptr = reinterpret_cast<double*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += n * sizeof(double);

  // Split int8 pointers (real/imag for A and B) - ORDER MATTERS! Match split_AB_int8<cuDoubleComplex>
  std::int8_t* a_int8_real_ptr = reinterpret_cast<std::int8_t*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += size_split_A_cmplx / 2;
  std::int8_t* a_int8_imag_ptr = reinterpret_cast<std::int8_t*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += size_split_A_cmplx / 2;
  std::int8_t* b_int8_real_ptr = reinterpret_cast<std::int8_t*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += size_split_B_cmplx / 2;
  std::int8_t* b_int8_imag_ptr = reinterpret_cast<std::int8_t*>(base_workspace_ptr + base_workspace_offset);
  base_workspace_offset += size_split_B_cmplx / 2;


  // Per-stream Workspace: int32 (size m*n), double (size m*n, used for real part accumulation)
  // We accumulate real parts of the 4 sub-products separately per stream.
  const std::size_t size_per_stream_i32 = m * n * sizeof(std::int32_t);
  const std::size_t size_per_stream_f64 = m * n * sizeof(double); // Accumulating real parts
  const std::size_t size_final_reduction_f64 = size_per_stream_f64;

  std::vector<std::int32_t*> d_per_stream_c_i32_ptrs(num_streams);
  std::vector<double*> d_per_stream_c_f64_ptrs(num_streams); // For accumulating real parts

  std::uint8_t* per_stream_base_ptr = base_workspace_ptr + base_workspace_offset;
  std::size_t current_per_stream_offset = 0;

  for (std::size_t i = 0; i < num_streams; ++i) {
      d_per_stream_c_i32_ptrs[i] = reinterpret_cast<std::int32_t*>(per_stream_base_ptr + current_per_stream_offset);
      current_per_stream_offset += size_per_stream_i32;
  }
    for (std::size_t i = 0; i < num_streams; ++i) {
      d_per_stream_c_f64_ptrs[i] = reinterpret_cast<double*>(per_stream_base_ptr + current_per_stream_offset);
      current_per_stream_offset += size_per_stream_f64;
  }
  double* d_final_reduced_f64_ptr = reinterpret_cast<double*>(per_stream_base_ptr + current_per_stream_offset);


  // --- Split Complex Data ---
  // Pointers needed by split_AB_int8<cuDoubleComplex>
  // IMPORTANT: Ensure the order matches how split_AB_int8 writes!
  std::int8_t *a_working_ptr_split = reinterpret_cast<std::int8_t*>(handle->working_memory_ptr) + (size_max_exp_cmplx);
  std::int8_t *b_working_ptr_split = a_working_ptr_split + size_split_A_cmplx;

  split_AB_int8<cuDoubleComplex>(
    handle, op_A, op_B, m, n, k, a_ptr, lda,
    a_real_max_exp_ptr,    // Output exp real A
    a_working_ptr_split,   // Output int8 real A (first half)
    ld_int8_a,
    b_ptr, ldb,
    b_real_max_exp_ptr,    // Output exp real B
    b_working_ptr_split,   // Output int8 real B (first half)
    ld_int8_b, num_split, bits_per_int8
    // Needs stream, use cuda_stream
    // mtk::ozimmu::split_int8<cuDoubleComplex>(..., handle->cuda_stream);
    );
  // Need pointers to imag parts for max_exp arrays (calculated above)
  // a_imag_max_exp_ptr, b_imag_max_exp_ptr

  CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream)); // Sync split

  // --- Initialise Output C ---
  init_c_complex(m, n, c_ptr, ldc, *beta, handle->cuda_stream);
  CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream)); // Sync init


  // Device array of pointers for reduction kernel
  double** h_per_stream_f64_ptrs = new double*[num_streams];
  for(size_t i=0; i<num_streams; ++i) h_per_stream_f64_ptrs[i] = d_per_stream_c_f64_ptrs[i];
  double** d_per_stream_f64_ptrs_array = nullptr;
  CUTF_CHECK_ERROR(cudaMalloc(&d_per_stream_f64_ptrs_array, num_streams * sizeof(double*)));
  CUTF_CHECK_ERROR(cudaMemcpy(d_per_stream_f64_ptrs_array, h_per_stream_f64_ptrs, num_streams * sizeof(double*), cudaMemcpyHostToDevice));
  delete[] h_per_stream_f64_ptrs;


  // --- Loop over 4 complex multiplications (real*real, imag*imag, real*imag, imag*real) ---
  const double *a_max_exp_ptr_list[] = {a_real_max_exp_ptr, a_imag_max_exp_ptr};
  const double *b_max_exp_ptr_list[] = {b_real_max_exp_ptr, b_imag_max_exp_ptr};
  // Pointers to the actual int8 data slices (real/imag)
  const std::int8_t *a_int8_slices_list[] = {a_int8_real_ptr, a_int8_imag_ptr};
  const std::int8_t *b_int8_slices_list[] = {b_int8_real_ptr, b_int8_imag_ptr};


  for (const auto p : std::vector<std::pair<unsigned, unsigned>>{{0, 0}, {1, 1}, {0, 1}, {1, 0}}) { // RR, II, RI, IR

      // Initialize per-stream double accumulators for this sub-product
      for (std::size_t i = 0; i < num_streams; ++i) {
          init_accumulator_buffer<double>(d_per_stream_c_f64_ptrs[i], m * n, handle->streams[i]);
      }

      // Inner loop launching tasks onto streams
      for (size_t i = 0; i < gemm_pair_config_list.size(); ++i) {
          const auto& gemm_pair_config = gemm_pair_config_list[i];
          const std::size_t stream_idx = i % num_streams;
          cudaStream_t current_stream = handle->streams[stream_idx];
          cublasHandle_t current_cublas_handle = handle->cublas_handles[stream_idx];
          cudaEvent_t current_event = handle->events[stream_idx];

          std::int32_t* current_c_i32_ptr = d_per_stream_c_i32_ptrs[stream_idx];
          double* current_c_f64_ptr = d_per_stream_c_f64_ptrs[stream_idx];

          // Select correct int8 input data for this sub-product (RR, II, RI, IR)
          const void* current_a_int8_working_ptr = a_int8_slices_list[p.first];
          const void* current_b_int8_working_ptr = b_int8_slices_list[p.second];


          matmul_core(handle, current_cublas_handle, current_stream,
                      mtk::ozimmu::op_t, mtk::ozimmu::op_n, // Assuming T, N for int8? Verify required ops
                      m, n,
                      ld_int8_a, // K value? Check usage
                      a_ptr, lda, mtk::ozimmu::fp64, b_ptr, ldb, mtk::ozimmu::fp64,
                      0, // beta_i = 0 for int32 output
                      current_c_i32_ptr, // Output int32
                      gemm_pair_config, compute_mode,
                      current_a_int8_working_ptr, // Input A int8 (real or imag)
                      ld_int8_a,
                      current_b_int8_working_ptr, // Input B int8 (real or imag)
                      ld_int8_b);

          // Accumulate into per-stream double buffer
            const double scale = ldexp(1.0, -bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2)
                                        // Adjust scale based on original accumulate_in_f64?
                                        // + (7 - bits_per_int8) * 2 // Original complex adjustment? Verify necessity
                                        );

          // Use accumulate_in_f64 for complex case (adds scaled int32<<32) ? Check definition
            accumulate_in_f64( // Or accumulate_in_f64_2 if that's more appropriate?
              current_c_f64_ptr, // Accumulate here
              current_c_i32_ptr, // Read from here
              m * n,
              // Mantissa shift calculation needs to be verified from original accumulate_in_f64
                bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2) - (7-bits_per_int8)*2, // Placeholder shift
              current_stream);

          CUTF_CHECK_ERROR(cudaEventRecord(current_event, current_stream));
      } // End inner loop (gemm_pair_config_list)

      // --- Sync, Reduce, and Combine for this sub-product (RR, II, RI, IR) ---
      for (std::size_t i = 0; i < num_streams; ++i) {
            // Check if event is valid before waiting
            bool event_recorded = false;
            for(size_t j=0; j < gemm_pair_config_list.size(); ++j) {
                if ((j % num_streams) == i) {
                    event_recorded = true;
                    break;
                }
            }
            if (event_recorded) {
                CUTF_CHECK_ERROR(cudaStreamWaitEvent(handle->cuda_stream, handle->events[i], 0));
            }
      }

      // Reduce per-stream double results
      reduce_sum_buffers<double>(
          d_per_stream_f64_ptrs_array,
          d_final_reduced_f64_ptr, // Output reduced real part
          num_streams,
          m * n,
          handle->cuda_stream);

      // Combine the reduced result into the final complex C matrix
      // Determine the correct complex alpha factor for this sub-product
      double alpha_real_part = 0.0;
      double alpha_imag_part = 0.0;
      if (p.first == 0 && p.second == 0) { // RR
          alpha_real_part = alpha->x;
          alpha_imag_part = alpha->y;
      } else if (p.first == 1 && p.second == 1) { // II
          alpha_real_part = -alpha->x;
          alpha_imag_part = -alpha->y;
      } else if (p.first == 0 && p.second == 1) { // RI
            alpha_real_part = -alpha->y; // -Im(alpha)
            alpha_imag_part = alpha->x;  // Re(alpha)
      } else { // IR (p.first == 1 && p.second == 0)
            alpha_real_part = -alpha->y; // -Im(alpha)
            alpha_imag_part = alpha->x;  // Re(alpha)
      }

        axy_complex(m, n, make_cuDoubleComplex(alpha_real_part, alpha_imag_part),
                    d_final_reduced_f64_ptr, // Input: reduced real part of sub-product
                    c_ptr, ldc,              // Output: final C matrix (accumulates)
                    a_max_exp_ptr_list[p.first], // Scaling factors
                    b_max_exp_ptr_list[p.second],
                    handle->cuda_stream);    // Run on user stream

  } // End outer loop (p)


  // --- Cleanup ---
  CUTF_CHECK_ERROR(cudaFree(d_per_stream_f64_ptrs_array));

  // Synchronize the final stream if the caller expects completion
  // CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));

  return 0;
}


} // unnamed namespace

int mtk::ozimmu::gemm(mtk::ozimmu::handle_t handle,
                      const mtk::ozimmu::operation_t op_A,
                      const mtk::ozimmu::operation_t op_B, const std::size_t m,
                      const std::size_t n, const std::size_t k,
                      const void *alpha, const void *const a_ptr,
                      const std::size_t lda, const void *const b_ptr,
                      const std::size_t ldb, const void *beta,
                      void *const c_ptr, std::size_t ldc,
                      const mtk::ozimmu::compute_mode_t compute_mode,
                      const mtk::ozimmu::element_kind_t element_kind) {
  // Arguments validation
  int arg_error = 0;
  arg_error |= check_gemm_shape(op_A, m, k, lda, "A");
  arg_error |= check_gemm_shape(op_B, k, n, ldb, "B");
  arg_error |= check_gemm_shape(mtk::ozimmu::op_n, m, n, ldc, "C");
  if (element_kind == mtk::ozimmu::real) {
    arg_error |= check_address_alignment<double>(
        reinterpret_cast<const double *>(a_ptr), "A");
    arg_error |= check_address_alignment<double>(
        reinterpret_cast<const double *>(b_ptr), "B");
    arg_error |= check_address_alignment<double>(
        reinterpret_cast<const double *>(c_ptr), "B");
  } else {
    arg_error |= check_address_alignment<cuDoubleComplex>(
        reinterpret_cast<const cuDoubleComplex *>(a_ptr), "A");
    arg_error |= check_address_alignment<cuDoubleComplex>(
        reinterpret_cast<const cuDoubleComplex *>(b_ptr), "B");
    arg_error |= check_address_alignment<cuDoubleComplex>(
        reinterpret_cast<const cuDoubleComplex *>(c_ptr), "B");
  }
  if (arg_error) {
    return 1;
  }

  mtk::ozimmu::data_t input_type;
  switch (compute_mode) {
  case mtk::ozimmu::sgemm:
    input_type = mtk::ozimmu::fp32;
    break;
  case mtk::ozimmu::dgemm:
  case mtk::ozimmu::fp64_int8_3:
  case mtk::ozimmu::fp64_int8_4:
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
    input_type = mtk::ozimmu::fp64;
    break;
  default:
    OZIMMU_NOT_IMPLEMENTED;
  }

  gemm_list_t gemm_list = {
      std::tuple<mtk::ozimmu::operation_t, mtk::ozimmu::operation_t,
                 std::size_t, std::size_t, std::size_t,
                 mtk::ozimmu::element_kind_t, mtk::ozimmu::compute_mode_t>{
          op_A, op_B, m, n, k, element_kind, compute_mode}};
  mtk::ozimmu::reallocate_working_memory(handle, gemm_list);

  if (input_type == mtk::ozimmu::fp64) {
    if (compute_mode == mtk::ozimmu::fp64_int8_3 ||
        compute_mode == mtk::ozimmu::fp64_int8_4 ||
        compute_mode == mtk::ozimmu::fp64_int8_5 ||
        compute_mode == mtk::ozimmu::fp64_int8_6 ||
        compute_mode == mtk::ozimmu::fp64_int8_7 ||
        compute_mode == mtk::ozimmu::fp64_int8_8 ||
        compute_mode == mtk::ozimmu::fp64_int8_9 ||
        compute_mode == mtk::ozimmu::fp64_int8_10 ||
        compute_mode == mtk::ozimmu::fp64_int8_11 ||
        compute_mode == mtk::ozimmu::fp64_int8_12 ||
        compute_mode == mtk::ozimmu::fp64_int8_13 ||
        compute_mode == mtk::ozimmu::fp64_int8_14 ||
        compute_mode == mtk::ozimmu::fp64_int8_15 ||
        compute_mode == mtk::ozimmu::fp64_int8_16 ||
        compute_mode == mtk::ozimmu::fp64_int8_17 ||
        compute_mode == mtk::ozimmu::fp64_int8_18) {
      if (element_kind == mtk::ozimmu::real) {
        using T = double;
        gemm_int8(handle, op_A, op_B, m, n, k,
                  reinterpret_cast<const T *>(alpha),
                  reinterpret_cast<const T *>(a_ptr), lda,
                  reinterpret_cast<const T *>(b_ptr), ldb,
                  reinterpret_cast<const T *>(beta),
                  reinterpret_cast<T *>(c_ptr), ldc, compute_mode);
      } else {
        using T = cuDoubleComplex;
        gemm_int8(handle, op_A, op_B, m, n, k,
                  reinterpret_cast<const T *>(alpha),
                  reinterpret_cast<const T *>(a_ptr), lda,
                  reinterpret_cast<const T *>(b_ptr), ldb,
                  reinterpret_cast<const T *>(beta),
                  reinterpret_cast<T *>(c_ptr), ldc, compute_mode);
      }
    } else if (compute_mode == mtk::ozimmu::fp64_int8_auto) {
      const auto auto_mode = mtk::ozimmu::auto_mode_select(
          handle, op_A, op_B, m, n, k, a_ptr, lda, b_ptr, ldb, element_kind,
          handle->avg_mantissa_loss_threshold);
      ozIMMU_log("AUTO selected mode = " +
                 mtk::ozimmu::get_compute_mode_name_str(auto_mode) +
                 ", threshold average mantissa loss = " +
                 std::to_string(handle->avg_mantissa_loss_threshold));
      return mtk::ozimmu::gemm(handle, op_A, op_B, m, n, k, alpha, a_ptr, lda,
                               b_ptr, ldb, beta, c_ptr, ldc, auto_mode,
                               element_kind);
    } else if (compute_mode == mtk::ozimmu::dgemm) {
      const auto dtype =
          element_kind == mtk::ozimmu::real ? CUDA_R_64F : CUDA_C_64F;
      cublasGemmEx_org(handle->cublas_handles[0], to_cublasOperation_t(op_A),
                       to_cublasOperation_t(op_B), m, n, k, alpha, a_ptr, dtype,
                       lda, b_ptr, dtype, ldb, beta, c_ptr, dtype, ldc,
                       CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
    } else {
      OZIMMU_NOT_IMPLEMENTED;
    }
  } else {
    OZIMMU_NOT_IMPLEMENTED;
  }
  return 0;
}