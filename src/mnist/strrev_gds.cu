#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "cufile.h"
#include <utils.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <vector>

#include <dataset_gds.cuh>
#include <layer.cuh>

//   This example computes the norm [1] of a vector.  The norm is 
// computed by squaring all numbers in the vector, summing the 
// squares, and taking the square root of the sum of squares.  In
// Thrust this operation is efficiently implemented with the 
// transform_reduce() algorith.  Specifically, we first transform
// x -> x^2 and the compute a standard plus reduction.  Since there
// is no built-in functor for squaring numbers, we define our own
// square functor.
//
// [1] http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

#define KB(x) ((x)*1024L)
#define TESTFILE "/mnt/test"

__global__ void hello(char *str) {
	printf("Hello World!\n");
	printf("buf: %s\n", str);
}

__global__ void strrev(char *str, int *len) {
	int size = 0;
	while (str[size] != '\0') {
		size++;
	}
	for(int i=0;i<size/2;++i) {
		char t = str[i];
		str[i] = str[size-1-i];
		str[size-1-i] = t;
	}
	/*
	printf("buf: %s\n", str);
	printf("size: %d\n", size);
	*/
	*len = size;
}

__global__ void g_reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
//   return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
//          ((unsigned int)ch3 << 8) + ch4;
}

unsigned int reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
         ((unsigned int)ch3 << 8) + ch4;
}

void test_stream(char* file_name) {
    int fd;
    int ret;
    
    fd = open(file_name, O_RDONLY);                                

    if (fd != -1) {
        char *gpumem_buf;

        unsigned int magic_number = 0;
        unsigned int number_of_images = 0;
        unsigned int n_rows = 0;
        unsigned int n_cols = 0;


        read(fd, (char*)&magic_number, sizeof(magic_number));
        read(fd, (char*)&number_of_images, sizeof(number_of_images));
        read(fd, (char*)&n_rows, sizeof(n_rows));
        read(fd, (char*)&n_cols, sizeof(n_cols));

        magic_number = reverse_int(magic_number);
        number_of_images = reverse_int(number_of_images);
        n_rows = reverse_int(n_rows);
        n_cols = reverse_int(n_cols);

        std::cout << file_name << std::endl;
        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << number_of_images << std::endl;
        std::cout << "rows = " << n_rows << std::endl;
        std::cout << "cols = " << n_cols << std::endl;

		int bufsize = n_rows * n_cols * sizeof(char) * number_of_images;
		int n_bufsize = n_rows * n_cols * sizeof(float);
        std::cout << "bufsize = " << bufsize << std::endl;

		if (bufsize > 0 ) {
	  		// thrust::device_vector<char> data(bufsize);
			// gpumem_buf = (char*)thrust::raw_pointer_cast(&data[0]);

			off_t file_offset = 16;
			off_t mem_offset = 0;
			CUfileDescr_t cf_desc; 
			CUfileHandle_t cf_handle;

			cuFileDriverOpen();
			cudaMalloc(&gpumem_buf, bufsize);

			cf_desc.handle.fd = fd;
			cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
			cuFileHandleRegister(&cf_handle, &cf_desc);
			cuFileBufRegister((char*)gpumem_buf, bufsize, 0);

			ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);
			printf("size %d\n", ret);
			if (ret > 0)

			cudaFree(gpumem_buf);
			cuFileDriverClose();
		}
		close(fd);
    }
}

void test(char * file_name) {
	int fd;
	int ret;
	int *sys_len;
	int *gpu_len;
	char *system_buf;
	char *gpumem_buf;

	int bufsize=KB(8);
	int parasize=KB(1);


	system_buf = (char*)malloc(bufsize);
	sys_len = (int*)malloc(sizeof(int)+1);

	cudaMalloc(&gpumem_buf, bufsize);
	cudaMalloc(&gpu_len, parasize);
	off_t file_offset = 0;
	off_t mem_offset = 0;

	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;

	cuFileDriverOpen();
	fd = open(file_name, O_RDWR | O_DIRECT);

	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	cuFileBufRegister((char*)gpumem_buf, bufsize, 0);

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d", ret); 
	}

	/*
	hello<<<1,1>>>(gpumem_buf);
	*/
	strrev<<<1,1>>>(gpumem_buf, gpu_len);

	cudaMemcpy(sys_len, gpu_len, sizeof(int), cudaMemcpyDeviceToHost);
	printf("sys_len : %d\n", *sys_len); 
	ret = cuFileWrite(cf_handle, (char*)gpumem_buf, *sys_len, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileWrite failed : %d", ret); 
	}

	cudaMemcpy(system_buf, gpumem_buf, bufsize, cudaMemcpyDeviceToHost);
	printf("system_buf: %s\n", system_buf);
	printf("See also %s\n", file_name);

	cuFileBufDeregister((char*)gpumem_buf);

	cudaFree(gpumem_buf);
	cudaFree(gpu_len);
	free(system_buf);
	free(sys_len);

	close(fd);
	cuFileDriverClose();
}

void test_numpy(char * file_name) {
	int fd;
	int ret;
	char *gpumem_buf, *meta;
	int *sys_len;
	int *gpu_len;
	int parasize=KB(1);

	int bufsize = KB(4);
	// int n_bufsize = n_rows * n_cols * sizeof(float);
	off_t file_offset = 0;
	off_t mem_offset = 0;
	int metasize=16;

	sys_len = (int*)malloc(parasize);

	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;

	cuFileDriverOpen();
	fd = open(file_name, O_RDWR|O_DIRECT);
	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	unsigned int magic_number = 0;
	unsigned int number_of_images = 0;
	unsigned int n_rows = 0;
	unsigned int n_cols = 0;


	// thrust::device_vector<char> data_tt(bufsize);
	// gpumem_buf = (char*)thrust::raw_pointer_cast(&data[0]);	
	cudaMalloc(&meta, metasize);
	// cuFileBufRegister((char*)meta, metasize, 0);

	ret = cuFileRead(cf_handle, (char*)meta, metasize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		printf("ret %d\n", ret);
	}

	cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
	magic_number = reverse_int(((int*)sys_len)[0]);
	number_of_images = reverse_int(((int*)sys_len)[1]);
	n_rows = reverse_int(((int*)sys_len)[2]);
	n_cols = reverse_int(((int*)sys_len)[3]);

	std::cout << file_name << std::endl;
	std::cout << "magic number = " << magic_number << std::endl;
	std::cout << "number of images = " << number_of_images << std::endl;
	std::cout << "rows = " << n_rows << std::endl;
	std::cout << "cols = " << n_cols << std::endl;
	bufsize = n_rows * n_cols * sizeof(char) * number_of_images;

	cudaFree(gpumem_buf);

	cudaMalloc(&gpumem_buf, bufsize);
	file_offset = 4 * sizeof(int);
	mem_offset = 0;

	cuFileBufRegister((char*)gpumem_buf, bufsize, 0);

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		printf("ret %d\n", ret);
	}

	cuFileBufDeregister((char*)gpumem_buf);
	cudaFree(gpumem_buf);
	close(fd);
	cuFileDriverClose();
}

template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

void test_transform() 
{
    float x[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);

	print_vector("d_x", d_x);

    // setup arguments
    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
	thrust::device_ptr<float> d_ptr = d_x.data();	

    // float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );
    float norm = std::sqrt( thrust::transform_reduce(d_ptr, d_ptr + sizeof(x), unary_op, init, binary_op) );

    std::cout << "norm is " << norm << std::endl;	
}
int main(int argc, char *argv[])
{
	std::unique_ptr<DataSetGDS> dataset;
    dataset.reset(new DataSetGDS(argv[1], false));

	// test(argv[1]);
	// char * mnist_data="/home/steven/dev/DataLoaders_DALI/cuda-neural-network/build/mnist_data/train-images-idx3-ubyte";
	// test_stream(argv[1]);
	// test_numpy(argv[1]);
	test_transform();
}
