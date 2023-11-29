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
Create a random float32, 2-dimension numpy array
with 10 rows and 10 columns.
"""
np_array = np.random.random((10, 10)).astype(np.float32)

"""Copy the numpy array to the CuArray instance"""
float_cuarray.fromNumpy2D(np_array)

"""Print the CuArray and numpy array to compare."""
for i in range(10):
    for j in range(10):
        print(float_cuarray[i][j], np_array[i][j])