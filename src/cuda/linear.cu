#include <linear.cuh>
#include <memory>

void operator_linear(const Storage *inputs, const Storage *weights,
                     Storage *output) {
  operator_matmul(inputs, weights, output);
}

void operator_d_linear(const Storage *outputs_grad, const Storage *inputs,
                       const Storage *weights, Storage *weights_grad,
                       Storage *inputs_grad) {
  std::vector<int> weights_transpose_shape(weights->get_shape());
  std::swap(weights_transpose_shape[0], weights_transpose_shape[1]);
  Storage weights_transpose(weights_transpose_shape);
  operator_transpose(weights, 0, 1, &weights_transpose);

  std::vector<int> inputs_transpose_shape(inputs->get_shape());
  std::swap(inputs_transpose_shape[0], inputs_transpose_shape[1]);
  Storage inputs_transpose(inputs_transpose_shape);
  operator_transpose(inputs, 0, 1, &inputs_transpose);

  // Y = X * W
  // dL/dX = dL/dY * W^T
  // dL/dW = X^T * dL/dY
  operator_matmul(outputs_grad, &weights_transpose, inputs_grad);
  operator_matmul(&inputs_transpose, outputs_grad, weights_grad);
}

__global__ void operator_bias_h(const float *inputs, const float *bias,
                                float *output, int width, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int col = index % width;
    output[index] = inputs[index] + bias[col];
  }
}

void operator_linear_bias(const Storage *inputs, const Storage *bias,
                          Storage *output) {
  const float *inputs_ptr = thrust::raw_pointer_cast(inputs->get_data().data());
  const float *bias_ptr = thrust::raw_pointer_cast(bias->get_data().data());
  float *output_ptr = thrust::raw_pointer_cast(output->get_data().data());

  int size = inputs->get_data().size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_bias_h<<<grid_size, BLOCK_SIZE>>>(inputs_ptr, bias_ptr, output_ptr,
                                             bias->get_data().size(), size);

  CUDA_POST_KERNEL_CHECK;
}

void operator_d_linear_bias(const Storage *outputs_grad, Storage *bias_grad) {
  operator_sum(outputs_grad, 0, bias_grad);
}

Linear::Linear(int in_size, int out_size, bool is_bias)
    : in_size(in_size), out_size(out_size), is_bias(is_bias) {
  this->weights.reset(new Storage({in_size, out_size}));
  this->weights->xavier(in_size, out_size);

  if (this->is_bias) {
    this->bias.reset(new Storage({1, out_size}));
    this->bias->xavier(in_size, out_size);
  }
}

std::vector<std::pair<Storage *, Storage *>> Linear::parameters() {
  return {std::make_pair(this->weights.get(), this->weights_grad.get()),
          std::make_pair(this->bias.get(), this->bias_grad.get())};
}

void Linear::forward() {
  const Storage *input = this->pre->get_output();
  std::vector<int> output_shape = {input->get_shape()[0], this->out_size};
  if (this->output.get() == nullptr ||
      this->output->get_shape() != output_shape) {
    this->output.reset(new Storage(output_shape));
  }

  operator_linear(input, this->weights.get(), this->output.get());
  if (this->bias) {
    operator_linear_bias(this->output.get(), this->bias.get(),
                         this->output.get());
  }
}

void Linear::backward() {
  const Storage *input = this->pre->get_output();
  const Storage *output_grad = this->next->get_grad();

  if (this->grad.get() == nullptr ||
      this->grad->get_shape() != input->get_shape()) {
    this->grad.reset(new Storage(input->get_shape()));
  }

  operator_d_linear(output_grad, input, this->weights.get(),
                    this->weights_grad.get(), this->grad.get());
  if (this->bias) {
    operator_d_linear_bias(output_grad, this->bias_grad.get());
  }
}