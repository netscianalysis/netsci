import numpy as np
from netcalc import mutualInformation, GPU_PLATFORM
from cuarray import FloatCuArray, IntCuArray

"""
Initialize the CuArrays for storing random variable data (X), mutual information results (I), 
and indices of random variables (ab).
"""
X = FloatCuArray()
I = FloatCuArray()

"""
'ab' is a 2xN CuArray where each row represents a pair of indices referring to the rows in 'X'.
These index pairs determine which pairs of random variables from 'X' are used in each mutual information calculation.
"""
ab = IntCuArray()

"""
Define the parameters for the mutual information calculation.
"""
"""Number of points in each random variable"""
n = 1000

"""k-nearest neighbors coefficient"""
k = 4

"""Dimension of the joint random variable space"""
xd = 2

"""Dimension of each random variable"""
d = 1

"""Calculation platform (GPU or CPU)"""
platform = GPU_PLATFORM

"""
Create a dictionary of calculation parameters 
"""
params = dict(
    X=X,
    I=I,
    ab=ab,
    k=k,
    n=n,
    xd=xd,
    d=d,
    platform=platform,
)

"""
Define the input domain for generating random variables.
"""
domain = np.linspace(0, 2 * np.pi, 1000)

"""
Initialize the random variable data in X:
- The first random variable is a sine function with added Gaussian noise.
- The second random variable is a cosine function with added Gaussian noise.
"""
X.init(2, 1000)
for i in range(1000):
    X[0][i] = np.sin(domain[i]) + np.random.normal(0, 0.1)  # Sine with noise
    X[1][i] = np.cos(domain[i]) + np.random.normal(0, 0.1)  # Cosine with noise

"""
Initialize the 'ab' array with pairs of indices for mutual information calculation.
Each row, such as [0, 1], represents a pair of random variables in 'X' to be analyzed.
"""
np_ab = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.int32)
ab.fromNumpy2D(np_ab)


"""
Perform mutual information calculation and print results.
"""
mutualInformation(**params)
print(I)

"""
Calculate and print the Pearson correlation coefficient for the data in 'X'.
The Pearson correlation is expected to be close to zero since the random variables
(sine and cosine functions with added noise) are orthogonal. This orthogonality
leads to a minimal linear correlation, which is what Pearson correlation measures.
"""
r = np.corrcoef(X.toNumpy2D())
print(r)
