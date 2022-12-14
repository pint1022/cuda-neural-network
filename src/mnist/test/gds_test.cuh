#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <time.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/host_vector.h>
#include <blas_gds.cuh>
#include <conv_gds.cuh>
#include <dataset_gds.cuh>
#include <dataset.cuh>

int test_operator_add(std::unique_ptr<DataSetGDS> gds, std::unique_ptr<DataSet> pty);
int test_read_image(char * minst_data_path, int length);
int test_read_label(char * minst_data_path, int length);