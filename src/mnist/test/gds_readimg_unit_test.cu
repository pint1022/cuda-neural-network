
#include "gds_test.cuh"


//
// test image reader on device
//
void test_read_image(char * minst_data_path, int length) {
    std::unique_ptr<DataSetGDS> gds;
    char * gds_image;
    
    gds.read_images(minst_data_path, gds_image);

    int end = std::min(length,
                       sizeof(gds_image));

    int end = std::min(length,  sizeof(gds_image));
    thrust::device_vector<char> host_label(gds_image, end);    
}