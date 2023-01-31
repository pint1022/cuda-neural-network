#include <vector>
#include <numeric>
#include <iterator>

void test_operator_add(char * minst_data_path);
double standardDeviation(std::vector<double> v);
char* read_numpy(char * file_name, int batchsize, int* rows, int * cols);
void test_storage(char * minst_data_path);