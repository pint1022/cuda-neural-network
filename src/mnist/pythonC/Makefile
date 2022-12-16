CC=g++
NVCC=nvcc
# CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++11 -c -arch=sm_80
# LIBS= -lopenblas -lpthread -lcudart -lcublas
# LIBDIRS=-L/usr/local/cuda-7.5/lib64

CUDA_INCDIRS=-I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I../../cuda -I../test -I../ 
CUDA_LIBDIRS=-L /usr/local/cuda/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L   -Bstatic -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl

.PHONY: all build test clean

all: build libgdsunittests.a gds_func_ops.h
	CC=g++ LDSHARED='$(shell python configure.py)' python setup.py build
	python setup.py install
	# python tests/test.py

install:
	python setup.py install

libgdsunittests.a: gds_add_unit_test.o gds_readimg_unit_test.o stddev_unit_test.o
	rm -f libgdsunittests.a
	ar cru libgdsunittests.a gds_add_unit_test.o gds_readimg_unit_test.o stddev_unit_test.o
	ranlib libgdsunittests.a

stddev_unit_test.o:  ../test/stddev_unit_test.cu
	$(NVCC) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_dev.o ../test/stddev_unit_test.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o stddev_unit_test.o temp_dev.o $(CUDA_LIBDIRS) 

gds_add_unit_test.o:  ../test/gds_add_unit_test.cu
	$(NVCC) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_add.o ../test/gds_add_unit_test.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o gds_add_unit_test.o temp_add.o $(CUDA_LIBDIRS) 

gds_readimg_unit_test.o:  ../test/gds_readimg_unit_test.cu
	$(NVCC) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_readimg.o ../test/gds_readimg_unit_test.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o gds_readimg_unit_test.o temp_readimg.o $(CUDA_LIBDIRS) 

clean:
	rm -rf *.out *.bin *.exe *.o *.a *.so *.png test build

sanity:
	python test.py
