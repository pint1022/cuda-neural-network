#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>
#include "../../src/cuda/util_func.cuh"

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

#define EPSILON         (1e-5)

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
int main(void)
{
  int A_rows = 1 << 8;
  int A_cols = 1 << 10;
  int B_cols = 1 << 11;

  int B_rows = A_cols;
  int C_rows = A_rows;
  int C_cols = B_cols;
  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  double *A, *B, *C, *C_host;
  double *A_cpu, *B_cpu, *C_cpu;
  // timing
  cudaEvent_t start_gpu, stop_gpu;
  float gpu_time_ms = 0;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  
  std::cout << "A size: " << A_size << ", B size: " << B_size << ", C Size: " << C_size << std::endl;  
  std::cout << "A: " << A_rows << "x" << A_cols << ", B: " << B_rows << "x" << B_cols <<  ", C: " << C_rows << "x" << C_cols << std::endl;  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&A, A_size*sizeof(double));

  cudaMalloc(&B, B_size*sizeof(double));
  cudaMalloc(&C, C_size*sizeof(double));
  C_host = (double*) malloc(C_size*sizeof(double));


  A_cpu = (double*) malloc(A_size*sizeof(double));
  B_cpu = (double*) malloc(B_size*sizeof(double));
  C_cpu = (double*) malloc(C_size*sizeof(double));

  // initialize A and B matrices
  auto all_ones = []() -> double {
    return 1.0f;
  };

  srand (time(NULL));
  auto rand_numbers = []() -> double {
    return static_cast<double>(rand())/(static_cast<double>(RAND_MAX/1000));
  };

  auto index_based = [](int i, int j) -> double {
    return j;
  };

  initialize_matrix<double>(A_cpu, A_rows, A_cols, rand_numbers);
	cudaMemcpy(A, A_cpu, A_size * sizeof(double), cudaMemcpyHostToDevice);  

  initialize_matrix<double>(B_cpu, B_rows, B_cols, rand_numbers);
	cudaMemcpy(B, B_cpu, B_size * sizeof(double), cudaMemcpyHostToDevice);


  // launch kernel

  cudaEventRecord(start_gpu);
  perform_matmul(A, B, C, A_rows, A_cols, B_rows, B_cols, 0);
  cudaEventRecord(stop_gpu);

  // Wait for GPU to finish before accessing on host
	cudaMemcpy(C_host, C, C_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop_gpu);
  cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
  

  // check results on CPU
  auto t1 = std::chrono::system_clock::now();
  naive_matrix_multiply_cpu<double>(A_cpu, B_cpu, C_cpu, A_cols, C_rows, C_cols);
  auto t2 = std::chrono::system_clock::now();

  if(fabsf(maxDiff<double>(C_host, C_cpu, C_rows, C_cols)) <= (double)EPSILON )
     std::cout << "PASS" << std::endl;
  else {
     std::cout << "FAIL" << std::endl;
     std::cout << "GPU result [0:9, 0:9]" << std::endl;
     print_matrix<double>( C_host, 10, 10);
     std::cout << "CPU result [0:9, 0:9]" << std::endl;
     print_matrix<double>( C_cpu, 10, 10);

  }

  auto cpu_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f;
  std::cout << "GPU time = " << gpu_time_ms << "ms" << std::endl;
  std::cout << "CPU time = " << cpu_time_ms << "ms" << std::endl;
  std::cout << "Speedup = " << cpu_time_ms/gpu_time_ms << std::endl;
  
  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(C_host);

  free(A_cpu);
  free(B_cpu);
  free(C_cpu);  
}

