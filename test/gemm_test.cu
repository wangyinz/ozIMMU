#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <getopt.h>

// Error checking macro for CUDA
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Error checking macro for cuBLAS
#define CUBLAS_CHECK(err) do { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Function for cublasDgemm test
void testCublasDgemm(int m, int n, int k, bool transposeA, bool transposeB, int iterations) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int lda = transposeA ? k : m;
    int ldb = transposeB ? n : k;
    int ldc = m;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, lda * (transposeA ? m : k) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, ldb * (transposeB ? k : n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, ldc * n * sizeof(double)));

    // Initialize host matrices
    double *h_A = (double *)malloc(lda * (transposeA ? m : k) * sizeof(double));
    double *h_B = (double *)malloc(ldb * (transposeB ? k : n) * sizeof(double));
    for (int i = 0; i < lda * (transposeA ? m : k); i++) {
        h_A[i] = (double)(rand() % 100) / 100.0;
    }
    for (int i = 0; i < ldb * (transposeB ? k : n); i++) {
        h_B[i] = (double)(rand() % 100) / 100.0;
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A, lda * (transposeA ? m : k) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, ldb * (transposeB ? k : n) * sizeof(double), cudaMemcpyHostToDevice));

    // Set transpose operations
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    double alpha = 1.0;
    double beta = 0.0;

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0.0;

    // Warm-up
    CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUBLAS_CHECK(cublasDgemm(handle, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute TFLOPS
    double flops = 2.0 * (double)m * (double)n * (double)k * iterations;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;
    printf("| cublasDgemm        | %10.3f | %7.3f |\n", milliseconds, tflops);

    // Clean up
    free(h_A);
    free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
}

// Function for cublasGemmEx test
void testCublasGemmEx(int m, int n, int k, bool transposeA, bool transposeB, int iterations) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int lda = transposeA ? k : m;
    int ldb = transposeB ? n : k;
    int ldc = m;

    // Allocate device memory
    int8_t *d_A_int8, *d_B_int8;
    int32_t *d_C_int32;
    CUDA_CHECK(cudaMalloc(&d_A_int8, lda * (transposeA ? m : k) * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B_int8, ldb * (transposeB ? k : n) * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C_int32, ldc * n * sizeof(int32_t)));

    // Initialize host matrices
    int8_t *h_A_int8 = (int8_t *)malloc(lda * (transposeA ? m : k) * sizeof(int8_t));
    int8_t *h_B_int8 = (int8_t *)malloc(ldb * (transposeB ? k : n) * sizeof(int8_t));
    for (int i = 0; i < lda * (transposeA ? m : k); i++) {
        h_A_int8[i] = (int8_t)(rand() % 100); // Direct int8 values
    }
    for (int i = 0; i < ldb * (transposeB ? k : n); i++) {
        h_B_int8[i] = (int8_t)(rand() % 100); // Direct int8 values
    }

    CUDA_CHECK(cudaMemcpy(d_A_int8, h_A_int8, lda * (transposeA ? m : k) * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_int8, h_B_int8, ldb * (transposeB ? k : n) * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Set transpose operations
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    int32_t alpha_int32 = 1;
    int32_t beta_int32 = 0;

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0.0;

    // Warm-up
    CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_int32, d_A_int8, CUDA_R_8I, lda,
                              d_B_int8, CUDA_R_8I, ldb, &beta_int32, d_C_int32, CUDA_R_32I, ldc,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, &alpha_int32, d_A_int8, CUDA_R_8I, lda,
                                  d_B_int8, CUDA_R_8I, ldb, &beta_int32, d_C_int32, CUDA_R_32I, ldc,
                                  CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute TOPS
    double ops = 2.0 * (double)m * (double)n * (double)k * iterations;
    double tops = (ops / (milliseconds / 1000.0)) / 1e12;
    printf("| cublasGemmEx (int8)| %10.3f | %7.3f |\n", milliseconds, tops);

    // Clean up
    free(h_A_int8);
    free(h_B_int8);
    CUDA_CHECK(cudaFree(d_A_int8));
    CUDA_CHECK(cudaFree(d_B_int8));
    CUDA_CHECK(cudaFree(d_C_int32));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
}

int main(int argc, char *argv[]) {
    int m = 4096, n = 4096, k = 4096, iterations = 10;
    bool transposeA = true, transposeB = false, verbose = false;

    // Define long options
static struct option long_options[] = {
        {"m", required_argument, 0, 'm'},
        {"n", required_argument, 0, 'n'},
        {"k", required_argument, 0, 'k'},
        {"transposeA", required_argument, 0, 'a'},
        {"transposeB", required_argument, 0, 'b'},
        {"iterations", required_argument, 0, 'i'},
        {"verbose", no_argument, 0, 'v'},
        {"mn", required_argument, 0, '1'},
        {"mk", required_argument, 0, '2'},
        {"nk", required_argument, 0, '3'},
        {"mnk", required_argument, 0, '4'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:n:k:a:b:i:v1:2:3:4:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case 'a':
                transposeA = atoi(optarg) != 0;
                break;
            case 'b':
                transposeB = atoi(optarg) != 0;
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            case '1': // -mn or --mn
                m = n = atoi(optarg);
                break;
            case '2': // -mk or --mk
                m = k = atoi(optarg);
                break;
            case '3': // -nk or --nk
                n = k = atoi(optarg);
                break;
            case '4': // -mnk or --mnk
                m = n = k = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [--m|-m] <m> [--n|-n] <n> [--k|-k] <k> [--mn] <m=n> [--mk] <m=k> [--nk] <n=k> [--mnk] <m=n=k> [--transposeA|-a] <0/1> [--transposeB|-b] <0/1> [--iterations|-i] <iterations> [--verbose|-v]\n", argv[0]);
                fprintf(stderr, "Defaults: m=4096, n=4096, k=4096, transposeA=1, transposeB=0, iterations=10\n");
                return EXIT_FAILURE;
        }    
    }	
    
    if (verbose) {
        printf("Matrix dimensions: m=%d, n=%d, k=%d\n", m, n, k);
        printf("Transpose A: %s, Transpose B: %s\n", transposeA ? "Yes" : "No", transposeB ? "Yes" : "No");
        printf("Iterations: %d\n", iterations);
    }

    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0) {
        fprintf(stderr, "Error: m, n, k, and iterations must be positive integers\n");
        return EXIT_FAILURE;
    }

    printf("+--------------------+------------+---------+\n");
    printf("| Operation          | Time (ms)  | T*OPS  |\n");
    printf("+--------------------+------------+---------+\n");
    testCublasDgemm(m, n, k, transposeA, transposeB, iterations);
    testCublasGemmEx(m, n, k, transposeA, transposeB, iterations);
    printf("+--------------------+------------+---------+\n");

    return 0;
}
