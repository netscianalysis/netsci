import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

print("Running", __file__)

"""
Create a new float CuArray instance with 
10 rows and 10 columns
"""
float_cuarray = FloatCuArray()

"""
Create a random float32 numpy array with 10 rows
and 10 columns
"""
numpy_array = np.random.rand(10, 10).astype(np.float32)

"""Save the numpy array to a .npy file"""
np.save("tmp.npy", numpy_array)

"""
Load the .npy file into the float CuArray instance
"""
float_cuarray.load("tmp.npy")

"""Print the CuArray and the numpy array"""
for i in range(10):
    for j in range(10):
        print(float_cuarray[i][j], numpy_array[i, j])