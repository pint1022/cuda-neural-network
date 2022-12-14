
#include "gds_test.cuh"

// test label reader on device
//
void test_read_label(char * minst_data_path, int length) {
    std::unique_ptr<DataSetGDS> gds;
    char * gds_label;
    
    gds.read_labels(minst_data_path, gds_label);

    int end = std::min(length,
                       sizeof(gds_label));
    thrust::device_vector<char> host_label(gds_label, end);
}