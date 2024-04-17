"""
Show how two columns of data points may have their
mutual information computed using Netsci
"""

import numpy as np

import netcalc
import cuarray

# k-nearest neighbors number
k = 1

# Dimensionality of data
d = 1

# Two dimensions of distributions
xd = 2

# Number of data points in 2D distribution
N = 100

# Define all pair correlations that will be computed
num_nodes = 2
num_node_pairs = num_nodes**2
ab = cuarray.IntCuArray()
ab.init(num_node_pairs, 2)
for i in range(num_nodes):
    for j in range(num_nodes):
        node_pair_index = i*num_nodes + j
        ab[node_pair_index][0] = i
        ab[node_pair_index][1] = j

# Load the data into a numpy array
data_list = []
with open("sample.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            # Skip the data header
            continue
        
        line = line.strip().split()
        col1_datum = line[0]
        col2_datum = line[1]
        data_list.append([col1_datum, col2_datum])

assert len(data_list) == N

two_columns_of_data = np.array(data_list, dtype=np.float32).T

# The input array
X = cuarray.FloatCuArray()
X.fromNumpy2D(two_columns_of_data)

# The output array
I = cuarray.FloatCuArray()

netcalc.mutualInformation(X, I, ab, k, N, xd, d, netcalc.GPU_PLATFORM)

mutual_information = I[0][1]

print("predicted mutual information for sample.dat:", mutual_information)

