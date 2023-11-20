"""
Always precede CuArray with the data type
Here we are importing float templates.
"""
from cuarray import FloatCuArray

print("Running", __file__)

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Initialize the float CuArray with 10 rows and 10 columns"""
float_cuarray.init(10, 10)

"""
Print the CuArray, 
which has a __repr__ method implemented in the SWIG interface
"""
print(float_cuarray)