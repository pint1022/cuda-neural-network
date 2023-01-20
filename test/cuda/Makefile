NVCC=nvcc
CUDAFLAGS = -std=c++14 -g -G -arch=sm_80 
CUDAINCLUDE = -I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I/usr/local/cuda/include  -I ./
CUDALIB = -lcudart -L/usr/local/cuda/lib
#-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

all : naive_matrix_multiply tiled_matrix_multiply

naive_matrix_multiply : naive_matmul.cu
	${NVCC} $(CUDAINCLUDE) $(CUDAFLAGS) $^ -o $@  $(CUDALIB)

tiled_matrix_multiply : tiled_matmul.cu
	${NVCC} $(CUDAINCLUDE) $(CUDAFLAGS) $^ -o $@ $(CUDALIB)

clean :
	rm -f naive_matrix_multiply tiled_matrix_multiply