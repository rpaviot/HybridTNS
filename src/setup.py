from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import os
import glob
import sysconfig
import numpy
import os

os.environ["CC"] = 'gcc-11'


libraries= ['gsl','gslcblas']
lib_path = "/usr/local/lib"
include_path = "/usr/local/include"
cython_path="/usr/local/lib/python3.8/site-packages"

def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)
    
    
             

compile_args = ["-O3", "-Wall", "-fopenmp" ,"-ffast-math"]
path = "./"
             
ext_modules = [Extension("gcc_integral",
                         [path+"gcc_integral.pyx"],
                         libraries=libraries,
                         library_dirs=[lib_path],
                         cython_include_dirs=[cython_path],
                         extra_compile_args =compile_args,
                         extra_link_args=['-lm','-fopenmp'],
                         include_dirs=[include_path,numpy.get_include()]),
            Extension("main",
                 [path+"main.pyx"],
                         libraries=libraries,
                         library_dirs=[lib_path],
                         cython_include_dirs=[cython_path],
                         extra_compile_args =compile_args,
                         extra_link_args=['-lm','-fopenmp'],
                         include_dirs=[include_path,numpy.get_include()]),
            Extension("TNScorr",
                     [path+"TNScorr.pyx"],
                         libraries=libraries,
                         library_dirs=[lib_path],
                         cython_include_dirs=[cython_path],
                         extra_compile_args =compile_args,
                         extra_link_args=['-lm','-fopenmp'],
                         include_dirs=[include_path,numpy.get_include()]),
            Extension("Dsigma",
                     [path+"Dsigma.pyx"],
                         libraries=libraries,
                         library_dirs=[lib_path],
                         cython_include_dirs=[cython_path],
                         extra_compile_args =compile_args,
                         extra_link_args=['-lm','-fopenmp'],
                         include_dirs=[include_path,numpy.get_include()]),
            Extension("cspline",
                 [path+"cspline.pyx"],
                         libraries=libraries,
                         library_dirs=[lib_path],
                         cython_include_dirs=[cython_path],
                         extra_compile_args =compile_args,
                         extra_link_args=['-lm','-fopenmp'],
                         include_dirs=[include_path,numpy.get_include()]),

]

packages = ["gcc_integral.pxd","cspline.pxd","main.pxd","TNScorr.pxd","Dsigma.pxd"]
setup(name = 'hybridTNS',package_data={"":packages}, cmdclass = {"build_ext": BuildExtWithoutPlatformSuffix},ext_modules = ext_modules)

