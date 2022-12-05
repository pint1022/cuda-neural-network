nvcc  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -g -I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I ../cuda strrev_gds.cu -o strrev_gds.co -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L   -Bstatic -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl 
# -lcrypto -lssl
# ../../build/mnist_data/train-images-idx3-ubyte
