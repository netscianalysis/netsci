"""
Always precede CuArray with the data type
Here we are importing CuArray int and float templates.
"""
from cuarray import FloatCuArray, IntCuArray

print("Running", __file__)

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Create a new int CuArray instance"""
int_cuarray = IntCuArray()
