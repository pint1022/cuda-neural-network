from distutils.core import setup, Extension

func_files = [
       'gds_unit_test.cpp', 
       'stddev_unit_test.cpp',
       'bind.cpp'
       ]
module1 = Extension('unittests',
                    sources =func_files)

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a unit test package for GDS-framework',
       ext_modules = [module1])