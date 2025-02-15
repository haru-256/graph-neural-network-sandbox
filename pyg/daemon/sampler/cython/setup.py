import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        "cython_fn.pyx",
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
