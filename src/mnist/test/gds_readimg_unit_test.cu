
#include "gds_test.cuh"

//
// test image reader on device
//
float* read_image(char * minst_data_path, int length) {
    std::unique_ptr<DataSetGDS> gds;
    char * gds_image;
    
    gds.reset(new DataSetGDS(minst_data_path, true));
    gds_image = gds->get_train_data();

    int end = std::min(length, gds->get_train_datasize());
    thrust::host_vector<char> host_img(gds_image, gds_image+end);

    return ((float*)&host_img[0]);
}


