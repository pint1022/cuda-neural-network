
#include "gds_test.cuh"

void operator_add(const GDSStorage *input1, const GDSStorage *input2,
                  GDSStorage *outputs) {
  CHECK_EQ(sizeof(input1->get_data()), sizeof(input2->get_data()),
           "operator_add: size error");

  thrust::device_ptr<float> input_ptr1 = thrust::device_pointer_cast((float*) input1->get_data());	
  thrust::device_ptr<float> input_ptr2 = thrust::device_pointer_cast((float*) input2->get_data());	
  thrust::device_ptr<float> output_ptr = thrust::device_pointer_cast((float*) outputs->get_data());	

  thrust::transform(input_ptr1, input_ptr1 + sizeof(input1),
                    input_ptr2, output_ptr,
                    thrust::plus<float>());
}

//
// test vector add operation on device
//
void test_operator_add(char * minst_data_path) {
    std::unique_ptr<DataSetGDS> ds;
    ds.reset(new DataSetGDS(minst_data_path, true));
}