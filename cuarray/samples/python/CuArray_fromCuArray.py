"""
Always precede CuArray with the data type
Here we are importing float templates.
"""
from cuarray import FloatCuArray

import numpy as np

print("Running", __file__)

"""Create two new float CuArray instances"""
float_cuarray1 = FloatCuArray()
float_cuarray2 = FloatCuArray()

"""Initialize float_cuarray1 with 10 rows and 10 columns"""
float_cuarray1.init(10, 10)

"""Fill float_cuarray1 with random values"""
for i in range(float_cuarray1.m()):
    for j in range(float_cuarray1.n()):
        val = np.random.random()
        float_cuarray1[i][j] = val

"""Copy the data from float_cuarray1 into float_cuarray2"""
float_cuarray2.fromCuArray(float_cuarray1, 0, 9, 10, 10)

"""
Print both CuArrays. Also this performs a deep copy for 
memory safety.
"""
for i in range(float_cuarray1.m()):
    for j in range(float_cuarray1.n()):
        print(float_cuarray1[i][j], float_cuarray2[i][j])