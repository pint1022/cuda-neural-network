
#include "gds_test.cuh"

//
// test vector add operation on device
//
void test_storage(char * minst_data_path) {
    std::unique_ptr<DataSetGDS> ds;
    ds.reset(new DataSetGDS(minst_data_path, true));
}