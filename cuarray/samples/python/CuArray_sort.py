import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

print("Running", __file__)

"""
Create a new float CuArray instance 
"""
float_cuarray = FloatCuArray()

"""
Create a random float32 numpy array with 10 rows
and 10 columns
"""
numpy_array = np.random.rand(10, 10).astype(np.float32)

"""Load the numpy array into the CuArray"""
float_cuarray.fromNumpy2D(numpy_array)

"""
Perform an out of place descending sort on the 
8th column of float_cuarray
"""
sorted_cuarray = float_cuarray.sort(7)

"""
Print the 8th row of the original 
CuArray and sorted_cuarray
"""
print(sorted_cuarray)
print(float_cuarray[7])