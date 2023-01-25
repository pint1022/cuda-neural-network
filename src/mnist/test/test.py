#!/usr/bin/env python3
import os
import numpy as np
import random
import timeit
import pandas as pd
from matplotlib import pyplot as plt
import unittests as gds

def get_dataset():
    datasetDir = os.getenv('ALNAIR_DATASET')
    if not (datasetDir and not datasetDir.isspace()):
        datasetDir = "/home/steven/dev/DataLoaders_DALI/cuda-neural-network/build/mnist_data/train-images-idx3-ubyte"
    return datasetDir


def data_gen(rg):
    lp_time = []
    py_time = []
    np_time = []
    c_time = []

    for l in rg:
        rands = [random.random() for _ in range(0, l)]
        numpy_rands = np.array(rands)
        np_time = np.append(np_time, timeit.timeit(lambda: np.std(numpy_rands), number=1000))
        # print(l, np_time.shape)
        c_time = np.append(c_time, timeit.timeit(lambda: gds.standard_dev(rands), number=1000))
    return np.array([np.transpose(np_time), np.transpose(c_time)])

def test_stddev():
    lens = range(1000, 20000, 1000)
    data = data_gen(rg=lens)

    df = pd.DataFrame(data.transpose(), index=lens, columns=['Numpy', 'C++'])
    plt.figure()
    df.plot()
    plt.legend(loc='best')
    plt.ylabel('Time (Seconds)')
    plt.xlabel('Number of Elements')
    plt.title('1k Runs of Standard Deviation')
    plt.savefig('numpy_vs_c.png')
    plt.show()

def test_system():
    gds.system("ls -l")

def test_add():
    print("test add function")
    print(gds.add(5, 6, 'testfile'))

def test_readimg():
    batch_size = 256
    mnist_data = get_dataset()

    data = gds.gds_read_image_data(mnist_data, batch_size)

def test_numpy():
    batch_size = 256
    mnist_data = get_dataset()
    row, col, data = gds.gds_read_numpy(mnist_data, batch_size)
    print("row: ", row, ", col: ", col)
    print(data)

def test_abc():
    for i in range(2):
        a = np.random.uniform(low=-5, high=5, size=10000)
        b = gds.adc3(a)
        print(np.stack((a, b), axis=-1))

def test_matmul():
    a_row = 1 << 9
    a_col = 1 << 10
    b_row = a_col
    b_col = 1 << 11
    A = np.random.rand(a_row, a_col)
    B = np.random.rand(b_row, b_col)
    C = gds.matmul(A,B, a_row, a_col, b_row, b_col)

def test_matmul_blas():
    a_row = 1 << 9
    a_col = 1 << 10
    b_row = a_col
    b_col = 1 << 11
    A = np.random.rand(a_row, a_col)
    B = np.random.rand(b_row, b_col)
    C = gds.bmatmul(A,B, a_row, a_col, b_row, b_col)

#
# basice testing
#
#
test_numpy()
test_abc()
test_add()
test_matmul_blas()
# test_matmul()




#######################################################
# test_stddev()
# test_system()
#
# read images
#
# test_readimg()

#
#no return data test
# test_mnist_train()