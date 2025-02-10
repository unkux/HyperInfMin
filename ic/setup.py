from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "ic_x",
        sources=["ic_x.pyx", "ic.c"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=["-O3"],
    )
]

setup(
    name="ic_x",
    ext_modules=cythonize(ext_modules),
)
