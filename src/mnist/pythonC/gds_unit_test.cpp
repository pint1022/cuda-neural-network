#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <numeric>
#include <iterator>
// #include "gds_unit_test.h"
#include "gds_func_ops.h"

PyObject *unitTestError;

static PyObject *
gds_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        PyErr_SetString(unitTestError, "System command failed");
        return NULL;
    }
    return PyLong_FromLong(sts);
}

static PyObject * 
add(PyObject *self, PyObject *args) 
{
	int num1, num2;
    char * filename;
	char eq[20];

	if(!PyArg_ParseTuple(args, "iis", &num1, &num2, &filename))
		return NULL;

	sprintf(eq, "%d + %d", num1, num2);

    test_operator_add(filename);

	return Py_BuildValue("is", num1 + num2, eq);
}

static PyObject * std_standard_dev(PyObject *self, PyObject* args)
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

// static PyObject* test_read_image(PyObject* self, PyObject* args) {
//     char * datafile;
//     int length;
//     float * output;

//     if (!PyArg_ParseTuple(args, "si", &datafile, &length))
//         return NULL;


//     printf("datafile: %s, length: %d\n", datafile, length);


//     output =  read_image(datafile, length);
//     int dims[1];
//     dims[0] = length;
//     return PyArray_SimpleNewFromData(1, dims, PyArray_TYPE(output), output);

// }