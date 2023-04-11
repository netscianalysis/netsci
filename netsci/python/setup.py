
from setuptools import find_packages
from setuptools import setup
import os

path = os.path.relpath(os.path.dirname(__file__))

setup(
    author='Andy Stokely',
    email='amstokely@ucsd.edu',
    name='netsci',
    install_requires=[],
    platforms=['Linux',
               'Unix', ],
    python_requires="<=3.9",
    py_modules=[path + "/netsci/netsci"],
    packages=find_packages() + [''],
    zip_safe=False,
    package_data={
        '': [
            path + '/netsci/_python_netsci.so'
        ]
    },
)
