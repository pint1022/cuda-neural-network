#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>
#include <cublas_v2.h>

#include "cublas_func.cuh"
#include <assert.h>

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<double()> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F();
    }
  }
}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<double(int, int)> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F(i, j);
    }
  }
}

template<typename T>
void print_matrix(T* M, int rows, int cols) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
        std::cout << M[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}


template<typename T>
T maxDiff(T* A1, T* A2, int rows, int cols){
  T maxDiff = A1[0] - A2[0];
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++){
      T diff = abs(A1[i * cols + j] - A2[i * cols + j]);
      if( diff > maxDiff) {
          maxDiff = diff;
      }
    }
  }

  
  return maxDiff;
}

template<typename T>
void check_copy(T* dM, T* hM, int d_size, char* label) {
  T* cp_host = (T*) malloc(d_size*sizeof(T));

  std::cout << "\n\n" << label << std::endl;

	cudaMemcpy(cp_host, dM, d_size, cudaMemcpyDeviceToHost);
  std::cout << "CPU original" << std::endl;
  print_matrix<T>( hM, 10, 10);
  std::cout << "GPU copy" << std::endl;
  print_matrix<T>( cp_host, 10, 10);

  free(cp_host);
}

#define ROW_TILE_WIDTH  32
#define COL_TILE_WIDTH  32

#define EPSILON         (1e-5)

template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int cRows, int cCols)
{
  __shared__ T shATile[ROW_TILE_WIDTH][COL_TILE_WIDTH];
  __shared__ T shBTile[ROW_TILE_WIDTH][COL_TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  T pValue = 0;

  // iterate for width/COL_TILE_WIDTH number of times
  // to compute the C tile
  for(int p = 0; p < width/COL_TILE_WIDTH; p++) {
    //load values to tiles from A and B
    shATile[threadIdx.y][threadIdx.x] = A[row * width + p * ROW_TILE_WIDTH + threadIdx.x];
    shBTile[threadIdx.y][threadIdx.x] = B[(p * COL_TILE_WIDTH + threadIdx.y) * cCols + col];

    // wait until all threads finish loading values
    __syncthreads();
    // update pValue for this thread
    for(int i = 0; i < COL_TILE_WIDTH;  i++) pValue += shATile[threadIdx.y][i] * shBTile[i][threadIdx.x];
    // wait until all threads finish computing pValue before overwriting
    __syncthreads();

  }
  
  C[row * cCols + col] = pValue;

}

template<typename T>
void naive_matrix_multiply_cpu(T *A, T *B, T* C, int width, int C_rows, int C_cols){
  
  for(int i = 0; i < C_rows; i++)
    for(int j = 0; j < C_cols; j++){
      T value = 0.0f;
      for(int k = 0; k < width; k++){
        value += A[i * width + k] * B[k * C_cols + j];
      }

     
      C[i * C_cols + j] = value;
    }
}
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

extern "C" double time_matmul(double* A_cpu, double *B_cpu, double *C_host, int a_row, int a_col, int b_row, int b_col, int flag) {
  int A_rows = a_row;
  int A_cols = a_col;
  int B_cols = b_col;

  int B_rows = A_cols;
  int C_rows = A_rows;
  int C_cols = B_cols;

  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  double *A, *B, *C;

  // timing
  cudaEvent_t start_gpu, stop_gpu;
  float gpu_time_ms = 0;

  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&A, A_size*sizeof(double));
  cudaMalloc(&B, B_size*sizeof(double));
  cudaMalloc(&C, C_size*sizeof(double));

  // initialize_matrix<double>(A_cpu, A_rows, A_cols, rand_numbers);
	cudaMemcpy(A, A_cpu, A_size * sizeof(double), cudaMemcpyHostToDevice); 
  const char* env_p;
  if(env_p = std::getenv("ALNAIR_DBG")) {
    if (strlen(env_p) > 0) {
      std::cout << "A size: " << A_size << ", B size: " << B_size << ", C Size: " << C_size << std::endl;  
      std::cout << "A: " << A_rows << "x" << A_cols << ", B: " << B_rows << "x" << B_cols <<  ", C: " << C_rows << "x" << C_cols << std::endl;  
      check_copy<double>(A, A_cpu, A_size, "A matrix");
    }
  } 

  // initialize_matrix<double>(B_cpu, B_rows, B_cols, rand_numbers);
	cudaMemcpy(B, B_cpu, B_size * sizeof(double), cudaMemcpyHostToDevice);

  // launch kernel

  // char kernel_name[20];

  if (flag == 1) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));
    // strcpy(kernel_name, "cublas");

    cudaEventRecord(start_gpu);
    stat = gpu_blas_mmul(&handle, (const float*) A, (const float *) B, (float *)C,  A_cols, C_rows, C_cols);
    cudaEventRecord(stop_gpu);

    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasSgemmBatched failed" << std::endl;
        exit(1);
    }
    assert(!cudaGetLastError());
    // Destroy the handle
    checkCublas(cublasDestroy(handle));    
  } else {
    // strcpy(kernel_name, "tiled");

    dim3 dim_grid(C_cols/COL_TILE_WIDTH, C_rows/ROW_TILE_WIDTH, 1);
    dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

    cudaEventRecord(start_gpu);
    naive_matrix_multiply<double><<<dim_grid, dim_block>>>(A, B, C, A_cols, C_rows, C_cols);
    cudaEventRecord(stop_gpu);

    // Wait for GPU to finish before accessing on host
  }
  cudaDeviceSynchronize();  
  cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
  cudaMemcpy(C_host, C, C_size * sizeof(double), cudaMemcpyDeviceToHost);
  

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return gpu_time_ms;
}

extern "C" void perform_matmul(double* A_cpu, double *B_cpu, double *C_host, int a_row, int a_col, int b_row, int b_col, int flag) {
  // int A_rows = 1 << 8;
  // int A_cols = 1 << 10;
  // int B_cols = 1 << 11;

  // int B_rows = A_cols;
  // int C_rows = A_rows;
  // int C_cols = B_cols;

  int A_rows = a_row;
  int A_cols = a_col;
  int B_cols = b_col;

  int B_rows = A_cols;
  int C_rows = A_rows;
  int C_cols = B_cols;

  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  double *A, *B, *C;
  // double *A_cpu, *B_cpu, *C_cpu;
  double *C_cpu;
  // timing
  cudaEvent_t start_gpu, stop_gpu;
  float gpu_time_ms = 0;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&A, A_size*sizeof(double));

  cudaMalloc(&B, B_size*sizeof(double));
  cudaMalloc(&C, C_size*sizeof(double));


  // initialize_matrix<double>(A_cpu, A_rows, A_cols, rand_numbers);
	cudaMemcpy(A, A_cpu, A_size * sizeof(double), cudaMemcpyHostToDevice); 
  const char* env_p;
  if(env_p = std::getenv("ALNAIR_DBG")) {
    if (strlen(env_p) > 0) {
      std::cout << "A size: " << A_size << ", B size: " << B_size << ", C Size: " << C_size << std::endl;  
      std::cout << "A: " << A_rows << "x" << A_cols << ", B: " << B_rows << "x" << B_cols <<  ", C: " << C_rows << "x" << C_cols << std::endl;  
      check_copy<double>(A, A_cpu, A_size, "A matrix");
    }
  } 

  // initialize_matrix<double>(B_cpu, B_rows, B_cols, rand_numbers);
	cudaMemcpy(B, B_cpu, B_size * sizeof(double), cudaMemcpyHostToDevice);

  // launch kernel

  char kernel_name[20];
  if (flag == 1) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));
    strcpy(kernel_name, "cublas");

    cudaEventRecord(start_gpu);
    stat = gpu_blas_mmul(&handle, (const float*) A_cpu, (const float *) B_cpu, (float *)C,  A_cols, C_rows, C_cols);
    cudaEventRecord(stop_gpu);
      if(stat != CUBLAS_STATUS_SUCCESS){
         	std::cerr << "cublasSgemmBatched failed" << std::endl;
	        exit(1);
      }
      assert(!cudaGetLastError());
    // Destroy the handle
    checkCublas(cublasDestroy(handle));        
  } else {
    strcpy(kernel_name, "tiled");

    dim3 dim_grid(C_cols/COL_TILE_WIDTH, C_rows/ROW_TILE_WIDTH, 1);
    dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

    naive_matrix_multiply<double><<<dim_grid, dim_block>>>(A, B, C, A_cols, C_rows, C_cols);
    cudaEventRecord(stop_gpu);

    // Wait for GPU to finish before accessing on host
  }
  cudaDeviceSynchronize();

  cudaMemcpy(C_host, C, C_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop_gpu);

  
  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}