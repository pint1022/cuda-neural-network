# How to create new unit tests:  
1.  create a source code in c++, filenaming convention: funcname_unit_test.cpp  under 'test' folder.
2.  add function declaration to gds_unit_test.h, for example:  
    PyObject * add(PyObject *, PyObject *);
3.  add the function definition in gds_unit_test.cpp
4.  add the method meta to PyMethodDef unittest_funcs array in bind.cpp  

    
