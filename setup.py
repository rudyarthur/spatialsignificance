from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("spatialsignificance", ["src/spatialsignificance.pyx"], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],include_dirs=[numpy.get_include()]),
]
setup(
   packages=find_packages(where="src"), 
   package_dir={"": "src"},
   package_data = {"": ["*.pyx"] },
   ext_modules = cythonize(extensions, annotate=True, 
   compiler_directives={'language_level' : "3", "cdivision":True, "boundscheck":False, "wraparound":False, "nonecheck":False}
   ),
   python_requires=">=3.9",
   install_requires=["numpy>=2.0.0", "Cython>=3.0.10" ],
   setup_requires=["numpy>=2.0.0", "Cython>=3.0.10" ],
   include_package_data=True,  
   zip_safe=False,
)
