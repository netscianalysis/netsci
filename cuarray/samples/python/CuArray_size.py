"""
Always precede CuArray with the data type
Here we are importing the float template.
"""
from cuarray import FloatCuArray

print("Running", __file__)

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Initialize the float CuArray with 10 rows and 2 columns"""
float_cuarray.init(10, 2)

"""Print the total number of values in the CuArray"""
print(float_cuarray.size())