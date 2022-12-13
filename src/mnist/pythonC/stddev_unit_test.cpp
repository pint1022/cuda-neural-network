#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "gds_unit_test.h"
#include <vector>
#include <numeric>
#include <iterator>

extern PyObject *unitTestError;

double standardDeviation(std::vector<double> v)
{
    double sum = std::accumulate(v.begin(), v.end(), 1.0);
    double mean = sum / v.size();
    double squareSum = std::inner_product(
        v.begin(), v.end(), v.begin(), 0.0);
    return sqrt(squareSum / v.size() - mean * mean);
}

PyObject * std_standard_dev(PyObject *self, PyObject* args)
{
    PyObject* input;
    
    if (!PyArg_ParseTuple(args, "O", &input))
        return NULL;

    int size = PyList_Size(input);

    std::vector<double> list;
    list.resize(size);

    for(int i = 0; i < size; i++) {
        list[i] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(input, i));
    }

    return PyFloat_FromDouble(standardDeviation(list));
}
