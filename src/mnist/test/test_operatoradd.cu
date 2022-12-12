
#include "gds_test.cuh"

void operator_add(const GDSStorage *input1, const GDSStorage *input2,
                  GDSStorage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_add: size error");

  thrust::device_ptr<float> input_ptr1 = input1->get_data();	
  thrust::device_ptr<float> input_ptr2 = input2->get_data();	
  thrust::device_ptr<float> output_ptr = outputs->get_data();	

  thrust::transform(input_ptr1, input_ptr1 + sizeof(input1->data_size )
                    input_ptr2, output_ptr,
                    thrust::plus<float>());
}

//
// test vector add operation on device
//
void test_operator_add(std::unique_ptr<DataSetGDS> ds) {
	
}