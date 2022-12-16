#define PY_SSIZE_T_CLEAN
#include "gds_func_ops.h"
// #include <vector>
// #include <numeric>
// #include <iterator>

double standardDeviation(std::vector<double> v)
{
    double sum = std::accumulate(v.begin(), v.end(), 1.0);
    double mean = sum / v.size();
    double squareSum = std::inner_product(
        v.begin(), v.end(), v.begin(), 0.0);
    return sqrt(squareSum / v.size() - mean * mean);
}

