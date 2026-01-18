"""Setup script for Cython extension compilation."""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "histlookup.lookup_cy",
        sources=["src/histlookup/lookup_cy.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
