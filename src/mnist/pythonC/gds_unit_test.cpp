#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <numeric>
#include <iterator>
// #include "gds_unit_test.h"
#include "gds_func_ops.h"
#include <numpy/arrayobject.h>

static PyObject *unitTestError;

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

static PyObject * 
std_standard_dev(PyObject *self, PyObject* args)
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

static PyObject* 
test_read_image(PyObject* self, PyObject* args) {
    char * datafile;
    int length;
    float * output;

    if (!PyArg_ParseTuple(args, "si", &datafile, &length))
        return NULL;


    printf("datafile: %s, length: %d\n", datafile, length);


    output =  read_image(datafile, length);
    npy_intp dims[1];
    dims[0] = length;
    return PyArray_SimpleNewFromData(1, dims, PyArray_TYPE(output), output);

}

static PyObject* 
test_read_image_data(PyObject* self, PyObject* args) {
    char * datafile;
    int length;
    char * output;
    PyArrayObject *outArray = NULL;    

    if (!PyArg_ParseTuple(args, "si", &datafile, &length))
        return NULL;


    int col, row;
    int ret =  read_image_data(datafile, length, &output, &row, &col);
    npy_intp dims[3]; //B R W 
    
    printf("image_data - batchsize: %d, rows: %d, cols: %d\n", length,  row , col);

    if (ret > 0) {
        dims[0] = ret;
        dims[1] = row;
        dims[2] = col;
    }
    outArray = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_INT, output);
    PyArray_ENABLEFLAGS(outArray, NPY_ARRAY_OWNDATA);    
    // outArray->flags |= NPY_ARRAY_OWNDATA;
    return PyArray_Return(outArray); 

}

static PyObject* 
test_read_numpy(PyObject* self, PyObject* args) {
    char * datafile;
    char * output;
    int col, row, length;
    PyArrayObject *outArray = NULL;    

    if (!PyArg_ParseTuple(args, "si", &datafile, &length))
        return NULL;


    char* ret =  read_numpy(datafile, length,  &row, &col);
    // int ret =  read_image_data(datafile, length, output, &row, &col);

    npy_intp dims[1]; //B R W 
    
    printf("batchsize: %d, rows: %d, cols: %d\n", length,  row , col);
    if (ret != NULL) {
       dims[0] = length*row*col;
    //    dims[1] = row;
    //    dims[2] = col;        
       printf("got data\n");
       outArray = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_INT, output);
       outArray->flags |= NPY_ARRAY_OWNDATA;
       return PyArray_Return(outArray);        
    //    free(ret);
    } else
       printf("ERROR: null return\n");

    // return PyArray_Return(outArray);        
    // return Py_BuildValue("ii", row, col);
    return Py_BuildValue("iiO", row, col, outArray);


}

// static PyObject *
// test_read_narray(PyObject *self, PyObject *args) {
//   PyArrayObject *inArray = NULL, *outArray = NULL;
//   double *pinp = NULL, *pout = NULL;
//   npy_intp nelem;
//   int dims[1], i, j;

//   /* Get arguments:  */
//   if (!PyArg_ParseTuple(args, "O:adc3", &inArray))
//     return NULL;

//   nelem = PyArray_DIM(inArray,0); /* size of the input array */
//   pout = (double *) malloc(nelem*sizeof(double));
//   pinp = (double *) PyArray_DATA(inArray);

//   /*   ADC action   */
//   for (i = 0; i < nelem; i++) {
//     if (pinp[i] >= -0.5) {
//     if      (pinp[i] < 0.5)   pout[i] = 0;
//     else if (pinp[i] < 1.5)   pout[i] = 1;
//     else if (pinp[i] < 2.5)   pout[i] = 2;
//     else if (pinp[i] < 3.5)   pout[i] = 3;
//     else                      pout[i] = 4;
//     }
//     else {
//     if      (pinp[i] >= -1.5) pout[i] = -1;
//     else if (pinp[i] >= -2.5) pout[i] = -2;
//     else if (pinp[i] >= -3.5) pout[i] = -3;
//     else                      pout[i] = -4;
//     }
//   }

//   dims[0] = nelem;

//   outArray = (PyArrayObject *)
//                PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, pout);
//   //Py_INCREF(outArray);
//   return PyArray_Return(outArray); 
// } 
