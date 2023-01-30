#include <cublas_v2.h>

cublasStatus_t gpu_blas_mmul(const cublasHandle_t * handle, const float *A, const float *B, float *C, const int m, const int k, const int n);