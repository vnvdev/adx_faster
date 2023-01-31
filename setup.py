from setuptools import setup
from Cython.Build import cythonize
import numpy as np
#python setup.py build_ext --inplace
setup(
    name='ADX Cython',
    ext_modules=cythonize("adx.pyx"),
    include_dirs=[np.get_include()],
)

