import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

print("Running", __file__)

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""
Create a random float32, 1-dimension numpy array, 
with 10 elements
"""

np_array = np.random.rand(10).astype(np.float32)

"""Copy the numpy array to the CuArray instance"""
float_cuarray.fromNumpy1D(np_array)

"""Print the CuArray and numpy array to compare."""
for _ in range(10):
    print(float_cuarray[0][_], np_array[_])
