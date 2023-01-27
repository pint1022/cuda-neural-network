#ifndef __UTIL_FUNC__
#define __UTIL_FUNC__


#include <functional>

// template<typename T>
// void initialize_matrix(T* M, int rows, int cols, std::function<double()> F);

// template<typename T>
// void initialize_matrix(T* M, int rows, int cols, std::function<double(int, int)> F);

template<typename T>
void print_matrix(T* M, int rows, int cols);


template<typename T>
T maxDiff(T* A1, T* A2, int rows, int cols);

template<typename T>
void check_copy(T* dM, T* hM, int d_size, char* label);

extern "C" void perform_matmul(double* A, double *B, double *C, int a_row, int a_col, int b_row, int b_col, int flag);
extern "C" double time_matmul(double* A_cpu, double *B_cpu, double *C_host, int a_row, int a_col, int b_row, int b_col, int flag);

#endif __UTIL_FUNC__