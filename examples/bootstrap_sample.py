"""
Demonstrate how one may compute error margins for mutual
information estimates of our data file sample.dat using
a bootstrapping analysis.
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

# Number of bootstrap samples of the data to compute
NUM_BOOTSTRAP_SAMPLES = 10

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

bootstrap_MI_array = np.zeros(NUM_BOOTSTRAP_SAMPLES)
for b in range(NUM_BOOTSTRAP_SAMPLES):
    # The input array
    X = cuarray.FloatCuArray()
    
    # Shuffle the data with replacement
    shuffled_two_columns_of_data = np.zeros(two_columns_of_data.shape, dtype=np.float32)
    indices = np.random.choice(np.arange(N), size=N, replace=True)
    shuffled_two_columns_of_data[0,:] = two_columns_of_data[0, indices]
    shuffled_two_columns_of_data[1,:] = two_columns_of_data[1, indices]
    X.fromNumpy2D(shuffled_two_columns_of_data)

    # The output array
    I = cuarray.FloatCuArray()

    netcalc.mutualInformation(X, I, ab, k, N, xd, d, netcalc.GPU_PLATFORM)

    bootstrap_MI_array[b] = I[0][1]

bootstrap_error = np.std(bootstrap_MI_array)
print("predicted mutual information for sample.dat:", mutual_information, "+/-", bootstrap_error)

