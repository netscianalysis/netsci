Tutorial 1: Simple Gaussian Mutual Information
==============================================

In this tutorial, we will generate some random data from a
Gaussian distribution to see how to compute the mutual
information (MI) for a set of data. A Jupyter notebook of
this tutorial can be found in the repository 
https://github.com/netscianalysis/ in the tutorials/ folder.

Make sure to activate the NetSci conda environment::

  conda activate netsci

Two Independent Gaussians
-------------------------

The following Python code will sample data from two independent
Gaussian distributions, with mean zero and standard deviation of 1, 
creating a file named 'sample.dat'::

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
    

Now, let us use NetSci to compute the MI between these two distributions::

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
    
Let us consider the inputs to this script later. First, let's examine the output::

    predicted mutual information for sample.dat: -0.015671253204345703
    
The exact MI for two independent Gaussian distributions can be found exactly:

.. math::
   
   I_{exact}(X,Y)=-\frac{1}{2}\log{(1-r^2)}

Where :math:`I_{exact}(X,Y)` is the MI of the two Gaussian distributions :math:`X` 
and :math:`Y`, and :math:`r` is the covariance of the two distributions, which in
this case, is zero. Therefore, for :math:`r=0`, :math:`I_{exact}(X,Y)=0`, which is
fairly close to the value we obtained in this case.

Let us consider the inputs to the script above:

k
  This is the "k" of "k-nearest-neighbors" - that is - how many neighbors to each point
  are taken in Kraskov's algorithm. Kraskov recommends a value of "k" between 2 and 4, 
  although situations where k=6 have also been used. In general, a lower value of "k"
  reduces bias, but increases noise, and a higher value of "k" reduces noise, but 
  increases bias for non-independent distributions. If testing for independence, bias
  is not such an issue, so k can be as large as N/2, where N is the number of data points.
  
d
  Dimensionality of the data. Since these are 1D Gaussians, the dimensionality is equal to
  one. In contrast, for example, the positions of atoms in a protein would be three-dimensional
  data.
  
xd
  The number of distributions per MI calculation. At this time, only the MI of two
  distributions can be calculated at a time in NetSci, so xd=2.
  
N
  The number of data points sampled from each distribution.
  
ab
  An array that represents which pairs of distributions to compute. This can be adjusted
  to exclude the computation of certain pairs of distributions, if desired.
  
X
  The input data, converted to a form that NetSci can use directly.
  
I
  The output MI, with entries for each pair of distributions specified for an MI calculation.
  
netcalc.GPU_PLATFORM
  Instruct NetSci to use the GPU device to perform the MI calculation. Alternatively, one may
  use the GPU by passing netcalc.CPU_PLATFORM as this argument.
  
Two Dependent Gaussians
-----------------------

Let us modify our data to include a strong dependency between the Gaussian distributions:

::

    import numpy as np
    
    # 100 rows
    N = 100

    # 2 columns
    M = 2
    
    covariance_value = 0.9

    def make_covariance_matrix(covariance_value):
        cov = np.array([[1.0, covariance_value], [covariance_value, 1.0]])
        return cov
    
    gaussian_2D_mean = np.zeros(2)
    gaussian_2D_cov = make_covariance_matrix(covariance_value)
    gaussian_2D = np.random.multivariate_normal(
        mean=gaussian_2D_mean,
        cov=gaussian_2D_cov,
        size=N,
    ).T.astype(np.float32)
    
    with open("sample.dat", "w") as f:
        f.write("column1\tcolumn2\n")
        for i in range(N):
            f.write(str(gaussian_2D[0,i])+"\t"+str(gaussian_2D[1,i])+"\n")
            
This time, when one runs the MI calculation, the predicted MI is much larger::

    predicted mutual information for sample.dat: 1.1769
    
This is due to the large covariance value between the two Gaussian distributions.

In Closing
----------

This tutorial demonstrates the MI calculation on a set of data that was
generated artificially by sampling from a Gaussian distribution. One could
repeat the same MI analysis on data that was generated from any other source
and benefit from the large speedup enabled by GPU acceleration.


