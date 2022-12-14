CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++11 -c -arch=sm_80
LIBS= -lopenblas -lpthread -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda-7.5/lib64

CUDA_INCDIRS=-I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I ../../cuda -I ../test -I../ 
CUDA_LIBDIRS=-L /usr/local/cuda/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L   -Bstatic -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl

.PHONY: all build test clean

all: build
	CC=g++ LDSHARED='$(shell python configure.py)' python setup.py build
	python setup.py install
	# python tests/test.py

install:
	python setup.py install

build:
	$(NVCC) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp.o ../test/gds_add_unit_test.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o gds_add_unit_test.o temp.o $(CUDA_LIBDIRS) 
	rm -f libgdsunittests.a
	ar cru libgdsunittests.a gds_add_unit_test.o temp.o
	ranlib libgdsunittests.a


clean:
	rm -rf *.out *.bin *.exe *.o *.a *.so *.png *.a test build