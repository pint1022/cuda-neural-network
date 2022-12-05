#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iterator>
#include <vector>

#define GDS_MEM 1
#define HOST_MEM 0

class GDSStorage {
 public:
  explicit GDSStorage(const std::vector<int> &_shape);
  explicit GDSStorage(const std::vector<int> &_shape, float value);
  explicit GDSStorage(const std::vector<int> &_shape, const std::vector<float> &_data);
  explicit GDSStorage(const std::vector<int> &_shape, int flag);

  // copy/move
  GDSStorage(const GDSStorage &other);
  GDSStorage&operator=(const GDSStorage &other);
  GDSStorage(GDSStorage &&other);
  GDSStorage&operator=(GDSStorage &&other);

  void reshape(const std::vector<int> &_shape);
  void resize(const std::vector<int> &_shape);
  void xavier(size_t in_size, size_t out_size);

  // get
  std::vector<int> &get_shape() { return this->shape; };
  const std::vector<int> &get_shape() const { return this->shape; };
  thrust::device_vector<float> &get_data() { return this->data; };
  const thrust::device_vector<float> &get_data() const { return this->data; };

 private:
  int flag;
  void check_size();  // check data/shape size

  thrust::device_vector<float> data;
  float* device_data;
  std::vector<int> shape;
};