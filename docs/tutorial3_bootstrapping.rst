Tutorial 3: Error Estimation with Bootstrapping
===============================================

In this tutorial, we will estimate the error margins of a
MI calculation by employing a bootstrapping method. A 
Jupyter notebook of this tutorial can be found in the repository 
https://github.com/netscianalysis/ in the tutorials/ folder.

Make sure to activate the NetSci conda environment::

  conda activate netsci

Bootstrapping Analysis
----------------------

The following Python code will sample data from two independent
Gaussian distributions, with mean zero and standard deviation of 1, 
creating a file named 'sample.dat' (This is the same method as in 
tutorial 1: Simple Gaussian MI)::

    import numpy as np

    # 100 rows
    N = 100

    # 2 columns
    M = 2

    gaussian_2D = np.zeros((M, N)).astype(np.float32)
    for i in range(M):
        gaussian_2D[i,:] = np.random.normal(size=N)
        
    with open("sample.dat", "w") as f:
        f.write("column1\tcolumn2\n")
        for i in range(N):
            f.write(str(gaussian_2D[0,i])+"\t"+str(gaussian_2D[1,i])+"\n")
            
Inside sample.dat, there are two columns of data, each containing 100 rows
(not including the header)::

    column1	column2
    -0.7315501	-0.2413269
    -1.012331	0.03488477
    -0.5544968	-1.5803745
    -0.86383635	0.91249526
    2.1049206	-0.46482915
    0.84087217	1.1912068
    2.6172605	1.1999675
    -0.52112085	0.1250357
    -0.9300766	0.73021877
    1.1873163	2.590006
    -1.0747769	-0.05441373
    ...
    

Now, let us compute the MI, and also run a bootstrapping analysis for
an error estimate::

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

