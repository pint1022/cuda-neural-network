CC=g++
NVCC=nvcc
SM := 80

GDSCUDADIR=../src/cuda
SRCDIR=./cuda
BUILDDIR=./build

CFLAGS = -std=c++14 -g -G
GENCODE_FLAGS = -gencode arch=compute_$(SM),code=sm_$(SM)

CUDAINCLUDE = -I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I/usr/local/cuda/include  -I../src/cuda -I./cuda
CUDALIB = -lcudart -L/usr/local/cuda/lib -L./build -lgdskernel -lcublas -ldl 
#-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

OBJ = $(BUILDDIR)/util_func.o $(BUILDDIR)/main.o

all : $(BUILDDIR)/naive_matrix_multiply $(BUILDDIR)/tiled_matrix_multiply $(BUILDDIR)/libgdskernel.a

$(BUILDDIR)/libgdskernel.a: $(BUILDDIR)/util_func.o $(BUILDDIR)/cublas_func.o
	rm -f $(BUILDDIR)/libgdskernel.a
	ar cru $(BUILDDIR)/libgdskernel.a $(BUILDDIR)/util_func.o $(BUILDDIR)/temp_util_func.o 
	ar cru $(BUILDDIR)/libgdskernel.a $(BUILDDIR)/cublas_func.o $(BUILDDIR)/temp_cublas_func.o 
	ranlib $(BUILDDIR)/libgdskernel.a

$(BUILDDIR)/naive_matrix_multiply : $(SRCDIR)/naive_matmul.cu  $(BUILDDIR)/libgdskernel.a
	${NVCC} $(CUDAINCLUDE) $(CFLAGS) $^ -o $@  $(CUDALIB) $(GENCODE_FLAGS)

# tiled_matrix_multiply : tiled_matmul.cu libgdskernel.a
# 	${NVCC} $(CUDAINCLUDE) $(CFLAGS) $(OBJ) $^ -o $@ $(CUDALIB) $(GENCODE_FLAGS)

$(BUILDDIR)/tiled_matrix_multiply : $(BUILDDIR)/main.o $(BUILDDIR)/libgdskernel.a
	${NVCC} --link $(CUDALIB) -o $(BUILDDIR)/matmul $(BUILDDIR)/main.o $(BUILDDIR)/libgdskernel.a 

$(BUILDDIR)/util_func.o : $(GDSCUDADIR)/util_func.cu $(GDSCUDADIR)/cublas_func.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_util_func.o $(GDSCUDADIR)/util_func.cu   $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/util_func.o $(BUILDDIR)/temp_util_func.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

$(BUILDDIR)/cublas_func.o : $(GDSCUDADIR)/cublas_func.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_cublas_func.o $(GDSCUDADIR)/cublas_func.cu   $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/cublas_func.o $(BUILDDIR)/temp_cublas_func.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

$(BUILDDIR)/main.o: $(SRCDIR)/tiled_matmul.cu
	$(NVCC) $(CFLAGS) -c  $(SRCDIR)/tiled_matmul.cu -o $(BUILDDIR)/main.o

clean :
	rm -f $(BUILDDIR)/naive_matrix_multiply $(BUILDDIR)/tiled_matrix_multiply $(BUILDDIR)/*.o $(BUILDDIR)/*.a