#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>
#include <string>
#include <vector>


#include "../../src/cuda/util_func.cuh"
template<typename T>
void host_check_copy(T* dM, T* hM, int d_size, char* label) {

  std::cout << "\n\n" << label << std::endl;
  std::cout << "CPU original" << std::endl;
  print_matrix<T>( hM, 10, 10);
  std::cout << "GPU copy" << std::endl;
  print_matrix<T>( dM, 10, 10);
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

#define EPSILON         (1e-5)

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float()> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F();
    }
  }
}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float(int, int)> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F(i, j);
    }
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

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t-l,--lib library\tSpecify cublas or custom lib"
              << std::endl;
}

int main(int argc, char * argv[])
{
  if (argc < 2) {
      show_usage(argv[0]);
      return 1;
  }
  std::vector <std::string> sources;
  std::string lib;
  int m = 10, n = 10, k = 10;

  for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if ((arg == "-h") || (arg == "--help")) {
          show_usage(argv[0]);
          return 0;
      } else if ((arg == "-l") || (arg == "--lib")) {
          if (i + 1 < argc) { // Make sure we aren't at the end of argv!
              lib = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
              std::cout << "kernel lib: "<< lib << std::endl;

          } else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--library option requires one argument." << std::endl;
              return 1;
          }  
      } else if (arg == "-m") {
          if (i + 1 < argc) { // Make sure we aren't at the end of argv!
              m = atoi(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
              std::cout << "m: "<< m << std::endl;

          } else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "-m option requires one argument." << std::endl;
              return 1;
          }  
      } else if (arg == "-n")  {
          if (i + 1 < argc) { // Make sure we aren't at the end of argv!
              n = atoi(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
              std::cout << "n: "<< n << std::endl;

          } else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "-n option requires one argument." << std::endl;
              return 1;
          }  
      } else if (arg == "-k") {
          if (i + 1 < argc) { // Make sure we aren't at the end of argv!
              k = atoi(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
              std::cout << "k: "<< k << std::endl;

          } else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "-k option requires one argument." << std::endl;
              return 1;
          }  
      } else {
          sources.push_back(argv[i]);
      }
  }

  int A_rows = 1 << m;
  int A_cols = 1 << k;
  int B_cols = 1 << n;

  int B_rows = A_cols;
  int C_rows = A_rows;
  int C_cols = B_cols;
  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  float  *C_host;
  float *A_cpu, *B_cpu, *C_cpu;
  
  // Allocate Unified Memory – accessible from CPU or GPU
  C_host = (float*) malloc(C_size*sizeof(float));
  C_cpu = (float*) malloc(C_size*sizeof(float));
  A_cpu = (float*) malloc(A_size*sizeof(float));
  B_cpu = (float*) malloc(B_size*sizeof(float));

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
  initialize_matrix<float>(B_cpu, B_rows, B_cols, rand_numbers);


  // launch kernel

  float gpu_time_ms;
  if (lib == "cublas")
        gpu_time_ms = time_matmul(A_cpu, B_cpu, C_host, A_rows, A_cols, B_rows, B_cols, 1);
  else
        gpu_time_ms = time_matmul(A_cpu, B_cpu, C_host, A_rows, A_cols, B_rows, B_cols, 0);


  auto t1 = std::chrono::system_clock::now();
  naive_matrix_multiply_cpu<float>(A_cpu, B_cpu, C_cpu, A_cols, C_rows, C_cols);
  auto t2 = std::chrono::system_clock::now();

  const char* env_p;
  if(env_p = std::getenv("ALNAIR_DBG")) {
    if (strlen(env_p) > 0) {

      host_check_copy<float>(C_host, C_cpu, C_size, "C matrix");
    }
  } 

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
  std::cout << lib << " GPU time = " << gpu_time_ms << "ms" << std::endl;
  std::cout << "CPU time = " << cpu_time_ms << "ms" << std::endl;
  std::cout << "Speedup = " << cpu_time_ms/gpu_time_ms << std::endl;


  free(C_host);
  free(C_cpu);

  free(A_cpu);
  free(B_cpu);
}

