
#include "gds_test.cuh"

//
// test vector add operation on device
// flag: 0, data; 1: label; 2: test data; 3: test label;
//
#define TRAIN_DATA  0
#define TRAIN_LABEL 1
#define TEST_DATA   2
#define TEST_LABEL  3

char* test_dataset(char * minst_data_path, int* batch, int *row, int *col, int flag) {
    std::unique_ptr<DataSetGDS> ds;
    ds.reset(new DataSetGDS(minst_data_path, true));
    char *buf;
    
    switch(flag){
        case TRAIN_LABEL:
            buf = (char*)malloc(ds->get_train_datasize());
            cudaMemcpy(buf, ds->get_train_label(), ds->get_train_datasize(), cudaMemcpyDeviceToHost);
            *row = ds->get_height();
            *col = ds->get_width();
            *batch = ds->get_number_of_image();
        break;
        case TEST_DATA:
            std::cout<< "NOT IMPLEMENTED.";
        break;
        case TEST_LABEL:
            std::cout<< "NOT IMPLEMENTED.";
        break;
        default:
            buf = (char*)malloc(ds->get_train_datasize());
            cudaMemcpy(buf, ds->get_train_data(), ds->get_train_datasize(), cudaMemcpyDeviceToHost);
            *row = ds->get_height();
            *col = ds->get_width();
            *batch = ds->get_number_of_image();
        break;
    }

    return(buf);
}