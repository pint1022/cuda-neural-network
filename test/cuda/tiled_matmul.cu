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
  double  *C_host;
  double *A_cpu, *B_cpu, *C_cpu;
  
  // Allocate Unified Memory – accessible from CPU or GPU
  C_host = (double*) malloc(C_size*sizeof(double));


  A_cpu = (double*) malloc(A_size*sizeof(double));
  B_cpu = (double*) malloc(B_size*sizeof(double));

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
  initialize_matrix<double>(B_cpu, B_rows, B_cols, rand_numbers);


  // launch kernel

  perform_matmul(A_cpu, B_cpu, C_host, A_rows, A_cols, B_rows, B_cols, 0);
  

  // if(fabsf(maxDiff<double>(C_host, C_cpu, C_rows, C_cols)) <= (double)EPSILON )
  //    std::cout << "PASS" << std::endl;
  // else {
  //    std::cout << "FAIL" << std::endl;
  //    std::cout << "GPU result [0:9, 0:9]" << std::endl;
  //    print_matrix<double>( C_host, 10, 10);
  //    std::cout << "CPU result [0:9, 0:9]" << std::endl;
  //    print_matrix<double>( C_cpu, 10, 10);
  // }
  
  // Free memory

  free(C_host);

  free(A_cpu);
  free(B_cpu);
}

