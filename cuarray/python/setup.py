
from setuptools import find_packages
from setuptools import setup
import os

path = os.path.relpath(os.path.dirname(__file__))

setup(
    author='Andy Stokely',
    email='amstokely@ucsd.edu',
    name='cuarray',
    install_requires=[],
    platforms=['Linux',
               'Unix', ],
    python_requires="<=3.9",
    py_modules=[path + "/cuarray/cuarray"],
    packages=find_packages() + [''],
    zip_safe=False,
    package_data={
        '': [
            path + '/cuarray/_python_cuarray.so'
        ]
    },
)
