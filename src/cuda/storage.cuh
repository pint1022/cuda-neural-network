#pragma once

#include <thrust/device_vector.h>

class Storage {
public:
  Storage(thrust::device_vector<std::size_t> shape);
  Storage(thrust::device_vector<std::size_t> shape,
          thrust::device_vector<float> &&data);

  void fill(float val);
  void xavier(size_t in_size, size_t out_size);

  // data
  thrust::device_vector<float> data;
  thrust::device_vector<std::size_t> shape;
};

__global__ storage_fill(Storage *s, float val);
__global__ storage_xavier(Storage *s, size_t in_size, size_t out_size);