from distutils.core import setup, Extension

func_files = [
       'gds_unit_test.cpp', 
       'stddev_unit_test.cpp',
       'bind.cpp'
       ]

module1 = Extension('unittests',
                    sources =func_files,
                    libraries=["gdsunittests"],
              library_dirs = ["."])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a unit test package for GDS-framework',
       ext_modules = [module1])

# setup(name = 'myModule', version = '1.0',  \
#    ext_modules = [
#       Extension('myModule', ['myModule.c'], 
#       include_dirs=[np.get_include(), os.path.join(CUDA_PATH, "include")],
#       libraries=["vectoradd", "cudart"],
#       library_dirs = [".", os.path.join(CUDA_PATH, "lib64")]
# )])
