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

int test_operator_add(std::unique_ptr<DataSetGDS> ds);