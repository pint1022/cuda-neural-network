#pragma once

#include <blas_gds.cuh>
#include <layer_gds.cuh>
#include <unordered_map>

#ifdef DEBUG

// High Performance Convolutional Neural Networks for Document Processing
// https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf

void im2col(const float *data_im, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_col);
void col2im(const float *data_col, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_im);

void operator_conv(const GDSStorage *inputs, GDSStorage *filters, GDSStorage *cols,
                   const int pad_h, const int pad_w, const int stride_h,
                   const int stride_w, GDSStorage *output);
void operator_d_conv(
    GDSStorage *outputs_grad, const GDSStorage *inputs, const GDSStorage *cols,
    GDSStorage *filters, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, GDSStorage *filters_grad, GDSStorage *inputs_grad,
    std::unordered_map<std::string, std::unique_ptr<GDSStorage>> &temp);

void operator_conv_bias(const GDSStorage *inputs, const GDSStorage *bias,
                        GDSStorage *output);
void operator_d_conv_bias(
    const GDSStorage *outputs_grad, GDSStorage *bias_grad,
    std::unordered_map<std::string, std::unique_ptr<GDSStorage>> &temp);

#endif

class Conv : public GDSLayer {
 public:
  explicit Conv(int height, int width, int channel_in, int channel_out,
                int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                int stride_w, bool is_bias);

  void forward();
  void backward();
  std::vector<std::pair<GDSStorage *, GDSStorage *>> parameters();

 private:
  std::unique_ptr<GDSStorage> filters;
  std::unique_ptr<GDSStorage> filters_grad;
  std::unique_ptr<GDSStorage> bias;
  std::unique_ptr<GDSStorage> bias_grad;
  std::unique_ptr<GDSStorage> cols;

  std::unordered_map<std::string, std::unique_ptr<GDSStorage>> temp;

  int height;
  int width;
  int channel_in;
  int channel_out;
  int kernel_h;
  int kernel_w;
  int pad_w;
  int pad_h;
  int stride_w;
  int stride_h;
  bool is_bias;
};
