
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
cublasStatus_t  gpu_blas_mmul(const cublasHandle_t * handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    // cublasHandle_t handle;
    // checkCublas(cublasCreate(&handle));

    // Do the actual multiplication
    return (cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));

    // Destroy the handle
    // cublasDestroy(handle);
}