from distutils.core import setup, Extension

module1 = Extension('unittests',
                    sources = ['gds_unit_test.c', 'bind.c'])
setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a unit test package for GDS-framework',
       ext_modules = [module1])