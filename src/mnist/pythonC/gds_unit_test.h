#ifndef __GDS_UNIT_TEST_H__
#define __GDS_UNIT_TEST_H__

#include <Python.h>

static PyObject * add(PyObject *, PyObject *);
static PyObject * gds_system(PyObject *self, PyObject *args);
static PyObject * std_standard_dev(PyObject *self, PyObject* args);
static PyObject * test_read_image(PyObject* self, PyObject* args);
static PyObject* test_read_image_data(PyObject* self, PyObject* args);
static PyObject* test_read_numpy(PyObject* self, PyObject* args);
// static PyObject *test_read_narray(PyObject *self, PyObject *args);

#endif