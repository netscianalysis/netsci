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
float_cuarray.init(10, 10)

"""Fill the array with random values"""

for i in range(10):
    for j in range(10):
        val = np.random.random()
        float_cuarray.set(val, i, j)

"""Print the array"""
print(float_cuarray)