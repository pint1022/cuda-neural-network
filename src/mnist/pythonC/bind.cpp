#include <iostream>
#include <Python.h>
#include "gds_func_ops.h"
#include <numpy/arrayobject.h>

#include "util_func.cuh"
#include "cublas_func.cuh"

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
    // PyArray_ENABLEFLAGS(outArray, NPY_ARRAY_OWNDATA);    
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

    npy_intp dims[3]; //B R W 
    
    printf("batchsize: %d, rows: %d, cols: %d\n", length,  row , col);
    if (ret != NULL) {
       dims[0] = length*row*col;
       dims[1] = row;
       dims[2] = col;        
       printf("got data\n");
       outArray = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_INT, output);
       outArray->flags |= NPY_ARRAY_OWNDATA;
    //    return  Py_BuildValue("iiO", row, col, outArray);
    //    return PyArray_Return(outArray);        
    //    free(ret);
    } else {
       printf("ERROR: null return\n");        
    }

    // return PyArray_Return(outArray);        
    // return Py_BuildValue("ii", row, col);
    return Py_BuildValue("iiO", row, col, outArray);


}

static PyObject *
adc3(PyObject *self, PyObject *args) {
  PyArrayObject *inArray = NULL, *outArray = NULL;
  double *pinp = NULL, *pout = NULL;
  npy_intp nelem;
  npy_intp dims[1];
  int i, j;

  /* Get arguments:  */
  if (!PyArg_ParseTuple(args, "O:adc3", &inArray))
    return NULL;

  nelem = PyArray_DIM(inArray,0); /* size of the input array */
  pout = (double *) malloc(nelem*sizeof(double));
  pinp = (double *) PyArray_DATA(inArray);

  /*   ADC action   */
  for (i = 0; i < nelem; i++) {
    if (pinp[i] >= -0.5) {
    if      (pinp[i] < 0.5)   pout[i] = 0;
    else if (pinp[i] < 1.5)   pout[i] = 1;
    else if (pinp[i] < 2.5)   pout[i] = 2;
    else if (pinp[i] < 3.5)   pout[i] = 3;
    else                      pout[i] = 4;
    }
    else {
    if      (pinp[i] >= -1.5) pout[i] = -1;
    else if (pinp[i] >= -2.5) pout[i] = -2;
    else if (pinp[i] >= -3.5) pout[i] = -3;
    else                      pout[i] = -4;
    }
  }

  dims[0] = nelem;

  outArray = (PyArrayObject *)
               PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, pout);
  // PyArray_ENABLEFLAGS(outArray, NPY_ARRAY_OWNDATA);    
  //Py_INCREF(outArray);
  return PyArray_Return(outArray); 
} 

static PyObject *
matmul(PyObject *self, PyObject *args) {
  PyArrayObject *AArray = NULL,  *BArray = NULL, *CArray = NULL;
  double *pA = NULL, *pB = NULL, *pC = NULL;
  npy_intp nelem;
  npy_intp dims[2];
  int i, j;
  int A_row, A_col, B_row, B_col;

  /* Get arguments:  */
  if (!PyArg_ParseTuple(args, "OOiiii:matmul", &AArray, &BArray, &A_row, &A_col, &B_row, &B_col))
    return NULL;

  nelem = PyArray_DIM(AArray,0); /* size of the input array */
  int c_size = A_row * B_col;
  pC = (double *) malloc(c_size*sizeof(double));
  pA = (double *) PyArray_DATA(AArray);
  pB = (double *) PyArray_DATA(BArray);

  std::cout << "A:" << A_row << "x" << A_col << ", C size: " << c_size << std::endl;

//
// flag
// 0: tiled
// 1: thrust
//
  float gpu_time_ms;

  // gpu_time_ms = 
  perform_matmul(pA, pB, pC, A_row, A_col, B_row, B_col, 0);
  dims[0] = A_row;
  dims[1] = B_col;

  CArray = (PyArrayObject *)
               PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, pC);
  return PyArray_Return(CArray); 
} 

static PyObject *
b_matmul(PyObject *self, PyObject *args) {
  PyArrayObject *AArray = NULL,  *BArray = NULL, *CArray = NULL;
  double *pA = NULL, *pB = NULL, *pC = NULL;
  npy_intp nelem;
  npy_intp dims[2];
  int i, j;
  int A_row, A_col, B_row, B_col;

  /* Get arguments:  */
  if (!PyArg_ParseTuple(args, "O!O!iiii:matmul", &PyArray_Type, &AArray, &PyArray_Type, &BArray, &A_row, &A_col, &B_row, &B_col))
    return NULL;
  if (AArray -> nd != 2 || BArray -> nd != 2 || AArray->descr->type_num != PyArray_DOUBLE || BArray->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
      return NULL;
  }
  nelem = PyArray_DIM(AArray,0); /* size of the input array */
  int c_size = A_row * B_col;
  pC = (double *) malloc(c_size*sizeof(double));
  pA = (double *) PyArray_DATA(AArray);
  pB = (double *) PyArray_DATA(BArray);

//
// flag
// 0: tiled
// 1: thrust
//
  float gpu_time_ms = time_matmul(pA, pB, pC, A_row, A_col, B_row, B_col, 1);
  std::cout << "cuBlas A:" << A_row << "x" << A_col << ", C size: " << c_size << std::endl;

  dims[0] = A_row;
  dims[1] = B_col;

  CArray = (PyArrayObject *)
               PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, pC);
  // free(pC);
//   PyArray_ENABLEFLAGS(CArray, NPY_ARRAY_OWNDATA);    
  //Py_INCREF(outArray);
  return PyArray_Return(CArray); 
} 

static PyObject* 
vector_add(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;
    double * output;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    if (array1 -> nd != 1 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    printf("running vector_add on dim1: %d, stride1: %d, dim2: %d, stride2: %d\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }

    output = myVectorAdd((double *) array1 -> data, (double *) array2 -> data, n1);

    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}


char system_docs[] = "call shell command by 'system'.  import gds_unittest as gds gds.system(\'ls -l\') ";
char addfunc_docs[] = "Add two numbers function.";
char stddevfunc_docs[] = "Return the standard deviation of a list.";
char gds_readimage_docs[] = "Read a batch of data from imagefile(mnist).";
char gds_readimagedata_docs[] = "Read a batch of data from imagefile(mnist). It returns the rows and columns of image";
char gds_read_narray_docs[] = "n-bit Analog-to-Digital Converter (ADC)";

char gds_matmul_docs[] = "matrix matmul op";
char gds_bmatmul_docs[] = "matrix matmul op in cublas";
char time_bmatmul_docs[] = "timing the matrix matmul op in cublas";
char time_matmul_docs[] = "timing the matrix matmul op";


static PyObject *initError;

static PyMethodDef unittest_funcs[] = {
	{	"system",	(PyCFunction)gds_system, METH_VARARGS,	system_docs},
	{	"add", (PyCFunction)add, METH_VARARGS,	addfunc_docs},
	{	"standard_dev",	(PyCFunction)std_standard_dev,	METH_VARARGS,	stddevfunc_docs},
	{	"gds_read_image",	(PyCFunction)test_read_image,	METH_VARARGS,	gds_readimage_docs},
	{	"gds_read_image_data", (PyCFunction)test_read_image_data, METH_VARARGS,	gds_readimage_docs},
	{	"gds_read_numpy",	(PyCFunction)test_read_numpy,	METH_VARARGS,	gds_readimage_docs},
	{	"adc3",	(PyCFunction)adc3,METH_VARARGS,	gds_read_narray_docs},
	{	"matmul", (PyCFunction)matmul,	METH_VARARGS,	gds_matmul_docs},
	{	"bmatmul", (PyCFunction)b_matmul,METH_VARARGS,	gds_bmatmul_docs},
  { "vector_add", (PyCFunction)vector_add, METH_VARARGS, "add two vectors with CUDA"},
	{	NULL}
};

char gds_unittest_docs[] = "This is a set of unit tests for gas-ai.";

static PyModuleDef unittests_mod = {
	PyModuleDef_HEAD_INIT,
	"unittests",
	gds_unittest_docs,
	-1,
	unittest_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit_unittests(void)
{
    PyObject *m;

    m = PyModule_Create(&unittests_mod);
    if (m == NULL)
        return NULL;

    import_array();  // for NumPy
    initError = PyErr_NewException("GAS unit test.error", NULL, NULL);
    Py_XINCREF(initError);
    if (PyModule_AddObject(m, "error", initError) < 0) {
        Py_XDECREF(initError);
        Py_CLEAR(initError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


// PyMODINIT_FUNC initmwa()  {
//     Py_InitModule("mwa", mwa_methods);
//     import_array();  // for NumPy
// }