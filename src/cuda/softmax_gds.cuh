#pragma once

#include <blas_gds.cuh>
#include <layer_gds.cuh>

#ifdef DEBUG

void operator_log_softmax(const GDSStorage *input1, int dim, GDSStorage *outputs);

void operator_d_log_softmax(const GDSStorage *output_grads, const GDSStorage *input1,
                            int dim, GDSStorage *inputs_grad);

#endif  // DEBUG

class LogSoftmax : public GDSLayer{
 public:
  explicit LogSoftmax(int dim = 1) : dim(dim) {}
  void forward();
  void backward();

 private:
  int dim;
};