import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray int and float templates 
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
Perform a descending sort on 
the 8th row of float_cuarray
"""
sorted_cuarray = float_cuarray.sort(7)

"""
Get the indices that sort the 8th row of float_cuarray
"""
argsort_cuarray = float_cuarray.argsort(7)

"""
Print the sorted 8th row of float_cuarray using 
sorted_cuarray and argsort_cuarray indices
"""
for _ in range(10):
    sort_idx = argsort_cuarray[0][_]
    print(
        sorted_cuarray[0][_],
        float_cuarray[7][sort_idx]
    )