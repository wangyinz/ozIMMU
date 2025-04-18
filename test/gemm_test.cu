#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <pthread.h>

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
    cudaStream_t malloc_stream;
    cudaStreamCreate(&malloc_stream);
    cublasSetStream(handle, malloc_stream);
    CUDA_CHECK(cudaMallocAsync(&d_A, lda * (transposeA ? m : k) * sizeof(double), malloc_stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, ldb * (transposeB ? k : n) * sizeof(double), malloc_stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, ldc * n * sizeof(double), malloc_stream));

    // Initialize host matrices
    double *h_A = (double *)malloc(lda * (transposeA ? m : k) * sizeof(double));
    double *h_B = (double *)malloc(ldb * (transposeB ? k : n) * sizeof(double));
    for (int i = 0; i < lda * (transposeA ? m : k); i++) {
        h_A[i] = (double)(rand() % 100) / 100.0;
    }
    for (int i = 0; i < ldb * (transposeB ? k : n); i++) {
        h_B[i] = (double)(rand() % 100) / 100.0;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(malloc_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, lda * (transposeA ? m : k) * sizeof(double), cudaMemcpyHostToDevice, malloc_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, ldb * (transposeB ? k : n) * sizeof(double), cudaMemcpyHostToDevice, malloc_stream));

    // Set transpose operations
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    double alpha = 1.0;
    double beta = 0.0;

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamSynchronize(malloc_stream));
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
    printf("| Dgemm              | %10.3f | %8.3f |\n", milliseconds/iterations, tflops);

    // Clean up
    free(h_A);
    free(h_B);
    CUDA_CHECK(cudaFreeAsync(d_A, malloc_stream));
    CUDA_CHECK(cudaFreeAsync(d_B, malloc_stream));
    CUDA_CHECK(cudaFreeAsync(d_C, malloc_stream));
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
    cudaStream_t malloc_stream;
    cudaStreamCreate(&malloc_stream);
    cublasSetStream(handle, malloc_stream);
    CUDA_CHECK(cudaMallocAsync(&d_A_int8, lda * (transposeA ? m : k) * sizeof(int8_t), malloc_stream));
    CUDA_CHECK(cudaMallocAsync(&d_B_int8, ldb * (transposeB ? k : n) * sizeof(int8_t), malloc_stream));
    CUDA_CHECK(cudaMallocAsync(&d_C_int32, ldc * n * sizeof(int32_t), malloc_stream));

    // Initialize host matrices
    int8_t *h_A_int8 = (int8_t *)malloc(lda * (transposeA ? m : k) * sizeof(int8_t));
    int8_t *h_B_int8 = (int8_t *)malloc(ldb * (transposeB ? k : n) * sizeof(int8_t));
    for (int i = 0; i < lda * (transposeA ? m : k); i++) {
        h_A_int8[i] = (int8_t)(rand() % 100); // Direct int8 values
    }
    for (int i = 0; i < ldb * (transposeB ? k : n); i++) {
        h_B_int8[i] = (int8_t)(rand() % 100); // Direct int8 values
    }

    CUDA_CHECK(cudaStreamSynchronize(malloc_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A_int8, h_A_int8, lda * (transposeA ? m : k) * sizeof(int8_t), cudaMemcpyHostToDevice, malloc_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_int8, h_B_int8, ldb * (transposeB ? k : n) * sizeof(int8_t), cudaMemcpyHostToDevice, malloc_stream));

    // Set transpose operations
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    int32_t alpha_int32 = 1;
    int32_t beta_int32 = 0;

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamSynchronize(malloc_stream));
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
    printf("| GemmEx(int8)       | %10.3f | %8.3f |\n", milliseconds/iterations, tops);

    // Clean up
    free(h_A_int8);
    free(h_B_int8);
    CUDA_CHECK(cudaFreeAsync(d_A_int8, malloc_stream));
    CUDA_CHECK(cudaFreeAsync(d_B_int8, malloc_stream));
    CUDA_CHECK(cudaFreeAsync(d_C_int32, malloc_stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
}

void testCublasLtMatmul(int m, int n, int k, bool transposeA, bool transposeB, int iterations) {
    // Enforce TN format for IMMA
    //if (!transposeA || transposeB) {
    //    printf("cublasLtMatmul: Skipping test (IMMA requires A transposed, B non-transposed)\n");
    //    return;
    //}

    // Check IMMA requirements
    //if (m % 4 != 0 || k % 4 != 0) {
    //    printf("cublasLtMatmul: Skipping test (m and k must be multiples of 4 for IMMA)\n");
    //    return;
    //}

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int lda = transposeA ? k : m; //k A is transposed (k x m)
    int ldb = transposeB ? n : k; //k B is non-transposed (k x n)
    int ldc = m; //C is m x n
    if (lda % 4 != 0 || ldb % 4 != 0 || ldc % 4 != 0) {
        printf("cublasLtMatmul: Skipping test (leading dimensions must be multiples of 4 for IMMA)\n");
        CUBLAS_CHECK(cublasDestroy(handle));
        return;
    }

    // Allocate device memory (cudaMalloc typically provides 256-byte alignment)
    int8_t *d_A_int8, *d_B_int8;
    int32_t *d_C_int32;
    cudaStream_t malloc_stream;
    cudaStreamCreate(&malloc_stream);
    cublasSetStream(handle, malloc_stream);
    CUDA_CHECK(cudaMallocAsync((void**)&d_A_int8, lda * (transposeA ? m : k) * sizeof(int8_t), malloc_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_B_int8, ldb * (transposeB ? k : n) * sizeof(int8_t), malloc_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_C_int32, ldc * n * sizeof(int32_t), malloc_stream));

    // Verify alignment (optional, for debugging)
    if ((uintptr_t)d_A_int8 % 16 != 0 || (uintptr_t)d_B_int8 % 16 != 0 || (uintptr_t)d_C_int32 % 16 != 0) {
        printf("cublasLtMatmul: Warning: Memory not 16-byte aligned\n");
    }

    // Initialize host matrices
    int8_t *h_A_int8 = (int8_t *)malloc(lda * (transposeA ? m : k) * sizeof(int8_t));
    int8_t *h_B_int8 = (int8_t *)malloc(ldb * (transposeB ? k : n) * sizeof(int8_t));
    for (int i = 0; i < lda * (transposeA ? m : k); i++) {
        h_A_int8[i] = (int8_t)(rand() % 100);
    }
    for (int i = 0; i < ldb * (transposeB ? k : n); i++) {
        h_B_int8[i] = (int8_t)(rand() % 100);
    }

    CUDA_CHECK(cudaStreamSynchronize(malloc_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A_int8, h_A_int8, lda * (transposeA ? m : k) * sizeof(int8_t), cudaMemcpyHostToDevice, malloc_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_int8, h_B_int8, ldb * (transposeB ? k : n) * sizeof(int8_t), cudaMemcpyHostToDevice, malloc_stream));

    // Set up matrix descriptors (default COL-major order)
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, lda, (transposeA ? m : k), lda)); // A: k x m (transposed)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, ldb, (transposeB ? k : n), ldb)); // B: k x n (non-transposed)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc)); // C: m x n

    // Set up matmul descriptor
    cublasLtMatmulDesc_t matmulDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // Scalars
    int32_t alpha_int32 = 1;
    int32_t beta_int32 = 0;

    // Workspace
    void *workspace;
    size_t workspaceSize = 1024 * 1024 * 1; // 32MB for sm90
    CUDA_CHECK(cudaMallocAsync(&workspace, workspaceSize, malloc_stream));

    // Algorithm selection (optional, using default for simplicity)
    //cublasLtMatmulAlgo_t algo;
    // For better performance, you could use cublasLtMatmulAlgoGetHeuristic here

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamSynchronize(malloc_stream));
    float milliseconds = 0.0;

    // Warm-up
    CUBLAS_CHECK(cublasLtMatmul((cublasLtHandle_t)handle, matmulDesc, &alpha_int32, d_A_int8, Adesc, d_B_int8, Bdesc,
                                &beta_int32, d_C_int32, Cdesc, d_C_int32, Cdesc, NULL, workspace, workspaceSize, 0));

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUBLAS_CHECK(cublasLtMatmul((cublasLtHandle_t)handle, matmulDesc, &alpha_int32, d_A_int8, Adesc, d_B_int8, Bdesc,
                                    &beta_int32, d_C_int32, Cdesc, d_C_int32, Cdesc, NULL, workspace, workspaceSize, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute TOPS
    double ops = 2.0 * (double)m * (double)n * (double)k * iterations;
    double tops = (ops / (milliseconds / 1000.0)) / 1e12;
    printf("| LtMatmul(int8)     | %10.3f | %8.3f |\n", milliseconds/iterations, tops);

    // Clean up
    free(h_A_int8);
    free(h_B_int8);
    CUDA_CHECK(cudaFreeAsync(d_A_int8 , malloc_stream));
    CUDA_CHECK(cudaFreeAsync(d_B_int8 , malloc_stream));
    CUDA_CHECK(cudaFreeAsync(d_C_int32, malloc_stream));
    CUDA_CHECK(cudaFreeAsync(workspace, malloc_stream));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
}

// Struct to pass parameters to pthread function
struct GemmParams {
    int m, n, k, iterations;
    bool transposeA, transposeB;
};

// Thread functions for each GEMM test
void* fnRunDgemm(void* arg) {
    GemmParams* params = (GemmParams*)arg;
    testCublasDgemm(params->m, params->n, params->k, params->transposeA, params->transposeB, params->iterations);
    return NULL;
}

void* fnRunGemmEx(void* arg) {
    GemmParams* params = (GemmParams*)arg;
    testCublasGemmEx(params->m, params->n, params->k, params->transposeA, params->transposeB, params->iterations);
    return NULL;
}

void* fnRunLtMatmul(void* arg) {
    GemmParams* params = (GemmParams*)arg;
    testCublasLtMatmul(params->m, params->n, params->k, params->transposeA, params->transposeB, params->iterations);
    return NULL;
}

int main(int argc, char *argv[]) {
    int m = 4096, n = 4096, k = 4096, iterations = 10;
    bool transposeA = true, transposeB = false, verbose = false;
    bool runDgemm = true, runGemmEx = true, runLtMatmul = true;
    bool parallel = false;

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
	{"dgemm", required_argument, 0, 'd'},      
	{"gemmex", required_argument, 0, 'g'},
	{"ltmatmul", required_argument, 0, 'l'},
        {"parallel", required_argument, 0, 'p'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:n:k:a:b:i:v1:2:3:4:d:g:l:p:", long_options, &option_index)) != -1) {
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
            case 'd': // -dgemm or --dgemm
                runDgemm = atoi(optarg) != 0;
                break;
            case 'g': // -gemmex or --gemmex
                runGemmEx = atoi(optarg) != 0;
                break;
            case 'l': // -ltmatmul or --ltmatmul
                runLtMatmul = atoi(optarg) != 0;
                break;
            case 'p': // -parallel or --parallel
                parallel = atoi(optarg) != 0;
                break;
            default:
                fprintf(stderr, "Usage: %s [--m|-m] <m> [--n|-n] <n> [--k|-k] <k> [--mn] <m=n> [--mk] <m=k> [--nk] <n=k> [--mnk] <m=n=k> [--transposeA|-a] <0/1> [--transposeB|-b] <0/1> [--iterations|-i] <iterations> [--verbose|-v] [--dgemm|-d] <0/1> [--gemmex|-g] <0/1> [--ltmatmul|-l] <0/1> [--parallel|-p] <0/1>\n", argv[0]);
                fprintf(stderr, "Defaults: m=4096, n=4096, k=4096, transposeA=1, transposeB=0, iterations=10, dgemm=1, gemmex=1, ltmatmul=1, parallel=0\n");
                return EXIT_FAILURE;
        }    
    }	
    
    if (verbose) {
        printf("Matrix dimensions: m=%d, n=%d, k=%d\n", m, n, k);
        printf("Transpose A: %s, Transpose B: %s\n", transposeA ? "Yes" : "No", transposeB ? "Yes" : "No");
        printf("Iterations: %d\n", iterations);
        printf("Tests enabled: dgemm=%s, gemmex=%s, ltmatmul=%s\n",
               runDgemm ? "Yes" : "No", runGemmEx ? "Yes" : "No", runLtMatmul ? "Yes" : "No");
        printf("Parallel execution: %s\n", parallel ? "Yes" : "No");
    }

    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0) {
        fprintf(stderr, "Error: m, n, k, and iterations must be positive integers\n");
        return EXIT_FAILURE;
    }

    printf("+--------------------+------------+----------+\n");
    printf("| Operation          | Time (ms)  | T*OPS    |\n");
    printf("+--------------------+------------+----------+\n");
    // Prepare parameters for threads
    GemmParams params = {m, n, k, iterations, transposeA, transposeB};

    if (parallel) {
        // Count number of enabled tests
        int numTests = (runDgemm ? 1 : 0) + (runGemmEx ? 1 : 0) + (runLtMatmul ? 1 : 0);
        if (numTests == 0) {
            printf("No tests enabled, exiting\n");
            return 0;
        }

        pthread_t threads[3];
        int threadCount = 0;

        // Launch threads for enabled tests
        if (runDgemm) {
            if (pthread_create(&threads[threadCount++], NULL, fnRunDgemm, &params) != 0) {
                fprintf(stderr, "Error creating dgemm thread\n");
                return EXIT_FAILURE;
            }
        }
        if (runGemmEx) {
            if (pthread_create(&threads[threadCount++], NULL, fnRunGemmEx, &params) != 0) {
                fprintf(stderr, "Error creating gemmex thread\n");
                return EXIT_FAILURE;
            }
        }
        if (runLtMatmul) {
            if (pthread_create(&threads[threadCount++], NULL, fnRunLtMatmul, &params) != 0) {
                fprintf(stderr, "Error creating ltmatmul thread\n");
                return EXIT_FAILURE;
            }
        }

        // Wait for all threads to complete
        for (int i = 0; i < threadCount; i++) {
            if (pthread_join(threads[i], NULL) != 0) {
                fprintf(stderr, "Error joining thread %d\n", i);
                return EXIT_FAILURE;
            }
        }
    } else {
        // Sequential execution
        if (runDgemm) {
            testCublasDgemm(m, n, k, transposeA, transposeB, iterations);
        }
        if (runGemmEx) {
            testCublasGemmEx(m, n, k, transposeA, transposeB, iterations);
        }
        if (runLtMatmul) {
            testCublasLtMatmul(m, n, k, transposeA, transposeB, iterations);
        }
    }

    printf("+--------------------+------------+----------+\n");

    return 0;
}
