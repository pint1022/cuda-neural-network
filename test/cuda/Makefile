NVCC=nvcc
SM := 80

CFLAGS = -std=c++14 -g -G
GENCODE_FLAGS = -gencode arch=compute_$(SM),code=sm_$(SM)

CUDAINCLUDE = -I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I/usr/local/cuda/include  -I ./ -I./cuda
CUDALIB = -lcudart -L/usr/local/cuda/lib -L./ -lgdskernel
#-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

OBJ = util_func.o main.o

all : naive_matrix_multiply tiled_matrix_multiply

libgdskernel.a: util_func.o
	rm -f libgdskernel.a
	ar cru libgdskernel.a util_func.o temp_util_func.o 
	ranlib libgdskernel.a

naive_matrix_multiply : naive_matmul.cu
	${NVCC} $(CUDAINCLUDE) $(CFLAGS) $^ -o $@  $(CUDALIB) $(GENCODE_FLAGS)

# tiled_matrix_multiply : tiled_matmul.cu libgdskernel.a
# 	${NVCC} $(CUDAINCLUDE) $(CFLAGS) $(OBJ) $^ -o $@ $(CUDALIB) $(GENCODE_FLAGS)

tiled_matrix_multiply : main.o libgdskernel.a
	${NVCC} --link -o matmul main.o libgdskernel.a 

util_func.o : util_func.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_util_func.o util_func.cu  $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o util_func.o temp_util_func.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

main.o: tiled_matmul.cpp
	g++ -c tiled_matmul.cpp -o main.o

clean :
	rm -f naive_matrix_multiply tiled_matrix_multiply