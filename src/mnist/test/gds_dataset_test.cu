
#include "gds_test.cuh"

//
// test vector add operation on device
//
char* test_dataset(char * minst_data_path, int* batch, int *row, int *col) {
    std::unique_ptr<DataSetGDS> ds;
    ds.reset(new DataSetGDS(minst_data_path, true));
    char *buf = (char*)malloc(ds->get_train_datasize());

    cudaMemcpy(buf, ds->get_train_data(), ds->get_train_datasize(), cudaMemcpyDeviceToHost);
    *row = ds->get_height();
    *col = ds->get_width();
    *batch = ds->get_number_of_image();

    return(buf);
}