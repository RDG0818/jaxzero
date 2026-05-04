import os
import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

here = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    name="cytree",
    sources=["cytree.pyx"],
    include_dirs=[here, np.get_include()],
    language="c++",
    extra_compile_args=["-O2", "-std=c++17", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)


class BuildExt(build_ext):
    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        super().build_extensions()


setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=cythonize([ext], language_level=3, include_path=[here]),
)
