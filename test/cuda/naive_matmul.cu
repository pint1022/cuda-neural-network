#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cassert>

#define ROW_TILE_WIDTH 32
#define COL_TILE_WIDTH 32

template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int C_rows, int C_cols)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;   
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( row < C_rows && col < C_cols ){
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[row * width + k] * B[k * C_cols + col];
    }
    // store result
    C[row * C_cols + col] = value;
  }
  

}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float()> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F();
    }
  }
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

template<typename T>
bool check_equal(T* A1, T* A2, int rows, int cols){
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++){
      if(abs(A1[i * cols + j] - A2[i * cols + j]) > 0.00001){
          return false;
      }
    }
  
  return true;
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

int check_data_move() 
{
    float *a_h, *b_h; // host data
    float *a_d, *b_d; // device data
    int N = 14, nBytes, i ;

    nBytes = N*sizeof(float);

    a_h = (float *)malloc(nBytes);
    b_h = (float *)malloc(nBytes);
    
    cudaMalloc((void **) &a_d, nBytes);
    cudaMalloc((void **) &b_d, nBytes);
    
    for (i=0; i<N; i++) 
        a_h[i] = 100.f + i;
    
    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);
    
    for (i=0; i< N; i++) assert( a_h[i] == b_h[i] );
    
    free(a_h); free(b_h); cudaFree(a_d); cudaFree(b_d);
    
    return 0;  
}

int main(void)
{
  int A_rows = 1 << 8;
  int A_cols = 1 << 10;
  int B_cols = 1 << 11;
  // int A_rows = 35;
  // int A_cols = 43;
  // int B_cols = 47;

  int B_rows = A_cols;
  int C_rows = A_rows;
  int C_cols = B_cols;
  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  float *A, *B, *C, *C_host;
  float *A_cpu, *B_cpu, *C_cpu;

  check_data_move();

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
    auto f = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/1000));
    int n = static_cast<int>(f);
    return static_cast<float>(n);
  };

  initialize_matrix<float>(A_cpu, A_rows, A_cols, rand_numbers);
	cudaMemcpy(A, A_cpu, A_size * sizeof(float), cudaMemcpyHostToDevice);  
	// cudaMemcpy(A, A_cpu, A_size, cudaMemcpyHostToDevice);  
  // check_copy(A, A_cpu, A_size*sizeof(float), "A");

  initialize_matrix<float>(B_cpu, B_rows, B_cols, rand_numbers);
	cudaMemcpy(B, B_cpu, B_size * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(B, B_cpu, B_size, cudaMemcpyHostToDevice);
  // check_copy(B, B_cpu, B_size*sizeof(float), "B");

  dim3 dim_grid(C_cols/COL_TILE_WIDTH, C_rows/ROW_TILE_WIDTH, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

  naive_matrix_multiply<float><<<dim_grid, dim_block>>>(A, B, C, A_cols, C_rows, C_cols);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
	cudaMemcpy(C_host, C, C_size * sizeof(float), cudaMemcpyDeviceToHost);

  // check results
  naive_matrix_multiply_cpu<float>(A_cpu, B_cpu, C_cpu, A_cols, C_rows, C_cols);

  // check_copy(C, C_cpu, C_size*sizeof(float), "C");
  
  if(check_equal<float>(C_host, C_cpu, C_rows, C_cols))
    std::cout << "PASS" << std::endl;
  else {
     std::cout << "FAIL" << std::endl;
     std::cout << "GPU result [0:9, 0:9]" << std::endl;
     print_matrix<float>( C_host, 10, 10);
     std::cout << "\n\nCPU result [0:9, 0:9]" << std::endl;
     print_matrix<float>( C_cpu, 10, 10);

  }

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  free(C_host);

  free(A_cpu);
  free(B_cpu);
  free(C_cpu);

  return 0; 
}
