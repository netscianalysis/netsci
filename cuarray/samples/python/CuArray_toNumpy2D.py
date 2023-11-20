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
np_array1 = np.random.random((10, 10)).astype(np.float32)

"""Copy the numpy array to the CuArray instance"""
float_cuarray.fromNumpy2D(np_array1)

"""Convert the CuArray instance to a numpy array"""
np_array2 = float_cuarray.toNumpy2D()

"""Print the CuArray and both numpy arrays to compare."""
for i in range(10):
    for j in range(10):
        print(
            float_cuarray[i][j],
            np_array1[i][j],
            np_array2[i][j]
        )