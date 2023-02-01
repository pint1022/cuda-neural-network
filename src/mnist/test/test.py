#!/usr/bin/env python3
import os
import numpy as np
import random
import timeit
import pandas as pd
from matplotlib import pyplot as plt
import unittests as gds
import idx2numpy
from sklearn.metrics import mean_squared_error

EPSILON = (1e-5)

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
    print("\n\nstandard deviance test")
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
    print("\n\nsyste command test")
    gds.system("ls -l")

def test_add():
    print("\n\ntest add function")
    print(gds.add(5, 6, 'testfile'))

def test_readimg():
    print("\n\nread image")
    batch_size = 256
    mnist_data = get_dataset()
    row, col, data = gds.gds_read_image_data(mnist_data, batch_size)
    print(data[4][8:18])
    print(np.ndim(data), np.shape(data))

def test_numpy():
    print("\n\n(PYT): read numpy format image")
    batch_size = 256
    mnist_data = get_dataset()
    row, col, data = gds.gds_read_numpy(mnist_data  + "/train-images-idx3-ubyte", batch_size)
    print("(PYT) row: ", row, ", col: ", col)
    # print(data[4][8:18])
    # print(np.ndim(data), np.shape(data))

def test_abc():
    print("\n\nabc test")
    for i in range(2):
        a = np.random.uniform(low=-5, high=5, size=10000)
        b = gds.adc3(a)
        print(np.stack((a, b), axis=-1))

def test_matmul():
    print("\n\ntiled matmul test")
    a_row = 1 << 9
    a_col = 1 << 10
    b_row = a_col
    b_col = 1 << 11
    A = np.random.rand(a_row, a_col)
    B = np.random.rand(b_row, b_col)
    C = gds.matmul(A,B, a_row, a_col, b_row, b_col)
    # print("returned", C[:9])
    CC = np.matmul(A,B)
    # print("numpy", CC[:9])

def test_matmul_blas():
    print("\n\ncublas matmul test")
    a_row = 1 << 9
    a_col = 1 << 10
    b_row = a_col
    b_col = 1 << 11
    A = np.random.rand(a_row, a_col)
    B = np.random.rand(b_row, b_col)
    C = gds.bmatmul(A,B, a_row, a_col, b_row, b_col)
    # print("returned",  C[:9])
    CC = np.matmul(A,B)
    # print("numpy", CC[:9])

def test_dataset(flag=1):
    print("\n\n(PYT): dataset class test")
    mnist_data = get_dataset()

    # flag: 0: train data; 1: train label; 2: test data; 3: test label;

    row, col, gds_data = gds.test_dataset(mnist_data, flag)
    print("(PYT) row: ", row, ", col: ", col)    
    # print(gds_data[4])
    if flag==0:
        imagefile = mnist_data + "/train-images-idx3-ubyte"
    elif flag == 1:
        imagefile = mnist_data + "/train-labels-idx1-ubyte"
    # print(imagefile)
    cpu_data = idx2numpy.convert_from_file(imagefile)
    print("cpu dim: ", np.ndim(cpu_data), ", gpu dim:", np.ndim(gds_data))
    for x in range(len(gds_data)):
        mse = mean_squared_error(gds_data[x], cpu_data[x])
        if mse > EPSILON:
           print(x, " MSE: ", mse, "\n test failed.")
           break
    print("dataset test: PASS")

    # print(np.ndim(data), np.shape(cpu_data))
#
# basice testing
#
#
# print(gds.vector_add(np.array([1., 2., 3.]), np.array([2., 3., 4.])))
# test_abc()
# test_matmul()
# test_matmul_blas()
# test_numpy()
test_dataset()



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