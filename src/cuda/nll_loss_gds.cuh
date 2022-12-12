#pragma once

#include <blas_gds.cuh>
#include <layer_gds.cuh>
#include <unordered_map>

#ifdef DEBUG

void operator_nll_loss(
    const GDSStorage *log_p, const GDSStorage *y, GDSStorage *output,
    std::unordered_map<std::string, std::unique_ptr<GDSStorage>> &temp);

void operator_d_nll_loss(const GDSStorage *y, GDSStorage *inputs_grad);

#endif  // DEBUG

class NLLLoss : public GDSLayer {
 public:
  NLLLoss() { this->output.reset(new GDSStorage({1, 1})); }
  void forward(const GDSStorage *y);
  void backward();

 private:
  const GDSStorage *y;  // backup

  std::unordered_map<std::string, std::unique_ptr<GDSStorage>> temp;
};