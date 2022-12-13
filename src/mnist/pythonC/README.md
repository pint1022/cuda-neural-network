# How to create new unit tests:  
1.  create a source code in c++, filenaming convention: funcname_unit_test.cpp  
2.  add function declaration to gds_unit_test.h, for example:  
    PyObject * add(PyObject *, PyObject *);
3.  add the method meta to PyMethodDef unittest_funcs array in bind.cpp  

    