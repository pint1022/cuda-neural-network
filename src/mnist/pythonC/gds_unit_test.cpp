#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "gds_unit_test.h"
#include <vector>
#include <numeric>
#include <iterator>

PyObject *unitTestError;

PyObject *
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

PyObject * 
add(PyObject *self, PyObject *args) 
{
	int num1, num2;
	char eq[20];

	if(!PyArg_ParseTuple(args, "ii", &num1, &num2))
		return NULL;

	sprintf(eq, "%d + %d", num1, num2);

	return Py_BuildValue("is", num1 + num2, eq);
}
