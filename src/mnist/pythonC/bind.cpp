// #include "gds_unit_test.h"
#include <Python.h>
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
  PyArray_ENABLEFLAGS(outArray, NPY_ARRAY_OWNDATA);    
  //Py_INCREF(outArray);
  return PyArray_Return(outArray); 
} 

// static PyMethodDef SpamMethods[] = {
//     {"system",  gds_system, METH_VARARGS,
//      "Execute a shell command."},
//     {"add",  add, METH_VARARGS,
//      "Execute a shell command."},
//     {NULL, NULL, 0, NULL}        /* Sentinel */
// };
// static struct PyModuleDef spammodule = {
//     PyModuleDef_HEAD_INIT,
//     "spam",   /* name of module */
//     NULL, /* module documentation, may be NULL */
//     -1,       /* size of per-interpreter state of the module,
//                  or -1 if the module keeps state in global variables. */
//     SpamMethods
// };

char system_docs[] = "call shell command by 'system'.  import gds_unittest as gds gds.system(\'ls -l\') ";
char addfunc_docs[] = "Add two numbers function.";
char stddevfunc_docs[] = "Return the standard deviation of a list.";
char gds_readimage_docs[] = "Read a batch of data from imagefile(mnist).";
char gds_readimagedata_docs[] = "Read a batch of data from imagefile(mnist). It returns the rows and columns of image";
char gds_read_narray_docs[] = "n-bit Analog-to-Digital Converter (ADC)";

static PyObject *initError;

static PyMethodDef unittest_funcs[] = {
	{	"system",
		(PyCFunction)gds_system,
		METH_VARARGS,
		system_docs},
	{	"add",
		(PyCFunction)add,
		METH_VARARGS,
		addfunc_docs},
	{	"standard_dev",
		(PyCFunction)std_standard_dev,
		METH_VARARGS,
		stddevfunc_docs},
	{	"gds_read_image",
		(PyCFunction)test_read_image,
		METH_VARARGS,
		gds_readimage_docs},
	{	"gds_read_image_data",
		(PyCFunction)test_read_image_data,
		METH_VARARGS,
		gds_readimage_docs},
	{	"gds_read_numpy",
		(PyCFunction)test_read_numpy,
		METH_VARARGS,
		gds_readimage_docs},
	{	"adc3",
		(PyCFunction)adc3,
		METH_VARARGS,
		gds_read_narray_docs},
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