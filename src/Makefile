CC=g++
NVCC=nvcc

SM := 80
GENCODE_FLAGS = -gencode arch=compute_$(SM),code=sm_$(SM)


CUDADIR=./cuda
MNISTDIR=./mnist
EXTDIR=./mnist/pythonC
UNITTESTDIR=./mnist/test

BUILDDIR=./build

# CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++14 -c -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
# LIBDIRS=-L/usr/local/cuda-7.5/lib64

CUDA_INCDIRS=-I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I$(CUDADIR) -I$(UNITTESTDIR) -I$(MNISTDIR) -I$(EXTDIR)
CUDA_LIBDIRS=-L /usr/local/cuda/targets/x86_64-linux/lib/ -L /usr/local/cuda/lib64/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L   -Bstatic  
LIBS=  -lpthread -lcublas -ldl -lcudart_static -lrt -lcufile -lcuda -L   -Bstatic -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl 

CUDAOBJ = $(BUILDDIR)/cublas_func.o $(BUILDDIR)/gds_add_unit_test.o  $(BUILDDIR)/stddev_unit_test.o $(BUILDDIR)/gds_datastorge_test.o\
          $(BUILDDIR)/storage_gds.o $(BUILDDIR)/dataset_gds.o $(BUILDDIR)/util_func.o $(BUILDDIR)/vectorAdd.o $(BUILDDIR)/reader_numpy_format.o

.PHONY: all build test clean $(CUDAOBJ)

all: build $(BUILDDIR)/libgdsunittests.a 
	CC=g++ LDSHARED='$(shell python $(EXTDIR)/configure.py)' python $(EXTDIR)/setup.py build
	python $(EXTDIR)/setup.py install
	# python $(UNITTESTDIR)/test.py

install: $(BUILDDIR)/libgdsunittests.a
	python $(EXTDIR)/setup.py install


$(BUILDDIR)/libgdsunittests.a:  $(CUDAOBJ)
	rm -f $(BUILDDIR)/libgdsunittests.a
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/gds_add_unit_test.o $(BUILDDIR)/temp_add.o 
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/stddev_unit_test.o  $(BUILDDIR)/temp_dev.o
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/storage_gds.o  $(BUILDDIR)/temp_storage.o
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/dataset_gds.o $(BUILDDIR)/temp_dataset.o
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/util_func.o $(BUILDDIR)/temp_util_func.o 
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/cublas_func.o $(BUILDDIR)/temp_cublas_func.o 
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/vectorAdd.o $(BUILDDIR)/temp_vectorAdd.o 
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/reader_numpy_format.o $(BUILDDIR)/temp_reader_numpy_format.o 
	ar cru $(BUILDDIR)/libgdsunittests.a $(BUILDDIR)/gds_datastorge_test.o $(BUILDDIR)/temp_gds_datastorge_test.o 
	ranlib $(BUILDDIR)/libgdsunittests.a

$(BUILDDIR)/gds_add_unit_test.o:  $(UNITTESTDIR)/gds_add_unit_test.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_add.o $(UNITTESTDIR)/gds_add_unit_test.cu $(GENCODE_FLAGS)
	$(NVCC) -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/gds_add_unit_test.o $(BUILDDIR)/temp_add.o $(CUDA_LIBDIRS) $(LIBS) $(GENCODE_FLAGS)

$(BUILDDIR)/stddev_unit_test.o:  $(UNITTESTDIR)/stddev_unit_test.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_dev.o $(UNITTESTDIR)/stddev_unit_test.cu $(GENCODE_FLAGS)
	$(NVCC) -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/stddev_unit_test.o $(BUILDDIR)/temp_dev.o $(CUDA_LIBDIRS) $(LIBS) $(GENCODE_FLAGS)

$(BUILDDIR)/storage_gds.o:  $(CUDADIR)/storage_gds.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_storage.o $(CUDADIR)/storage_gds.cu $(GENCODE_FLAGS)
	$(NVCC) -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/storage_gds.o $(BUILDDIR)/temp_storage.o $(CUDA_LIBDIRS) $(LIBS) $(GENCODE_FLAGS)

$(BUILDDIR)/dataset_gds.o:  $(MNISTDIR)/dataset_gds.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_dataset.o $(MNISTDIR)/dataset_gds.cu $(GENCODE_FLAGS)
	$(NVCC) -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/dataset_gds.o $(BUILDDIR)/temp_dataset.o $(CUDA_LIBDIRS) $(LIBS) $(GENCODE_FLAGS)

$(BUILDDIR)/util_func.o : $(CUDADIR)/util_func.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_util_func.o $(CUDADIR)/util_func.cu  $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/util_func.o $(BUILDDIR)/temp_util_func.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

$(BUILDDIR)/cublas_func.o : $(CUDADIR)/cublas_func.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_cublas_func.o $(CUDADIR)/cublas_func.cu  $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/cublas_func.o $(BUILDDIR)/temp_cublas_func.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

$(BUILDDIR)/vectorAdd.o : $(CUDADIR)/vectorAdd.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_vectorAdd.o $(CUDADIR)/vectorAdd.cu  $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/vectorAdd.o $(BUILDDIR)/temp_vectorAdd.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

$(BUILDDIR)/reader_numpy_format.o : $(CUDADIR)/reader_numpy_format.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_reader_numpy_format.o $(CUDADIR)/reader_numpy_format.cu  $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/reader_numpy_format.o $(BUILDDIR)/temp_reader_numpy_format.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

$(BUILDDIR)/gds_datastorge_test.o : $(UNITTESTDIR)/gds_datastorge_test.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_gds_datastorge_test.o $(UNITTESTDIR)/gds_datastorge_test.cu  $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/gds_datastorge_test.o $(BUILDDIR)/temp_gds_datastorge_test.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

clean:
	rm -rf *.out *.bin *.exe *.o *.a *.so *.png $(BUILDDIR)/*

sanity:
	python $(UNITTESTDIR)/test.py
