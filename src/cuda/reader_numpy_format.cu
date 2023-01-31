#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <vector>
#include "cufile.h"
#include "reader_api.cuh"

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

char* gds_read_numpy(char * file_name, int length,  int *row, int *col) {
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
	} 
	// else {
	// 	printf("ret %d\n", ret);
	// }

	sys_len = (int*)malloc(parasize);
	cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
	magic_number = reverse_int(((int*)sys_len)[0]);
	number_of_images = reverse_int(((int*)sys_len)[1]);
	n_rows = reverse_int(((int*)sys_len)[2]);
	n_cols = reverse_int(((int*)sys_len)[3]);
	free(sys_len);
	cudaFree(meta);


	bufsize = n_rows * n_cols * sizeof(char) * number_of_images;
	cudaMalloc(&gpumem_buf, bufsize);
	file_offset = metasize;
	mem_offset = 0;

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);

	char * output = NULL;
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		// printf("ret %d data, should be 16\n", ret);
		output = (char*) malloc(bufsize);
		cudaMemcpy(output, gpumem_buf, bufsize, cudaMemcpyDeviceToHost);
		*row = n_rows;
		*col = n_cols;		
	}

	cudaFree(gpumem_buf);
	close(fd);
	cuFileHandleDeregister(&cf_handle);
	cuFileDriverClose();

	return output;
}

char* read_numpy(char * file_name, int length,  int *row, int *col) {
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
	} 
	// else {
	// 	printf("ret %d\n", ret);
	// }

	sys_len = (int*)malloc(parasize);
	cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
	magic_number = reverse_int(((int*)sys_len)[0]);
	number_of_images = reverse_int(((int*)sys_len)[1]);
	n_rows = reverse_int(((int*)sys_len)[2]);
	n_cols = reverse_int(((int*)sys_len)[3]);
	free(sys_len);
	cudaFree(meta);


	bufsize = n_rows * n_cols * sizeof(char) * number_of_images;
	cudaMalloc(&gpumem_buf, bufsize);
	file_offset = metasize;
	mem_offset = 0;

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);

	char * output = NULL;
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		// printf("ret %d data, should be 16\n", ret);
		output = (char*) malloc(bufsize);
		cudaMemcpy(output, gpumem_buf, bufsize, cudaMemcpyDeviceToHost);
		*row = n_rows;
		*col = n_cols;		
	}

	cudaFree(gpumem_buf);
	close(fd);
	cuFileHandleDeregister(&cf_handle);
	cuFileDriverClose();

	return output;
}
