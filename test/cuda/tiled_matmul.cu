#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>
#include "util_func.cuh"

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
  float *A, *B, *C, *C_host;
  float *A_cpu, *B_cpu, *C_cpu;
  // timing
  cudaEvent_t start_gpu, stop_gpu;
  float gpu_time_ms = 0;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  
  std::cout << "A size: " << A_size << ", B size: " << B_size << ", C Size: " << C_size << std::endl;  
  std::cout << "A: " << A_rows << "x" << A_cols << ", B: " << B_rows << "x" << B_cols <<  ", C: " << C_rows << "x" << C_cols << std::endl;  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&A, A_size*sizeof(float));

  cudaMalloc(&B, B_size*sizeof(float));
  cudaMalloc(&C, C_size*sizeof(float));
  C_host = (float*) malloc(C_size*sizeof(float));


  A_cpu = (float*) malloc(A_size*sizeof(float));
  B_cpu = (float*) malloc(B_size*sizeof(float));
  C_cpu = (float*) malloc(C_size*sizeof(float));

  // initialize A and B matrices
  auto all_ones = []() -> float {
    return 1.0f;
  };

  srand (time(NULL));
  auto rand_numbers = []() -> float {
    return static_cast<float>(rand())/(static_cast<float>(RAND_MAX/1000));
  };

  auto index_based = [](int i, int j) -> float {
    return j;
  };

  initialize_matrix<float>(A_cpu, A_rows, A_cols, rand_numbers);
	cudaMemcpy(A, A_cpu, A_size * sizeof(float), cudaMemcpyHostToDevice);  

  initialize_matrix<float>(B_cpu, B_rows, B_cols, rand_numbers);
	cudaMemcpy(B, B_cpu, B_size * sizeof(float), cudaMemcpyHostToDevice);


  // launch kernel

  dim3 dim_grid(C_cols/COL_TILE_WIDTH, C_rows/ROW_TILE_WIDTH, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

  cudaEventRecord(start_gpu);
  naive_matrix_multiply<float><<<dim_grid, dim_block>>>(A, B, C, A_cols, C_rows, C_cols);
  cudaEventRecord(stop_gpu);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
	cudaMemcpy(C_host, C, C_size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop_gpu);
  cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
  

  // check results on CPU
  auto t1 = std::chrono::system_clock::now();
  naive_matrix_multiply_cpu<float>(A_cpu, B_cpu, C_cpu, A_cols, C_rows, C_cols);
  auto t2 = std::chrono::system_clock::now();

  if(fabsf(maxDiff<float>(C_host, C_cpu, C_rows, C_cols)) <= (float)EPSILON )
     std::cout << "PASS" << std::endl;
  else {
     std::cout << "FAIL" << std::endl;
     std::cout << "GPU result [0:9, 0:9]" << std::endl;
     print_matrix<float>( C_host, 10, 10);
     std::cout << "CPU result [0:9, 0:9]" << std::endl;
     print_matrix<float>( C_cpu, 10, 10);

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

