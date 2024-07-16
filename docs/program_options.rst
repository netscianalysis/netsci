Program Options for NetSci
==========================

This page contains brief descriptions of common functions and inputs used
in NetSci.

netcalc.mutualInformation
-------------------------

A call to mutualInformation will be formatted as follows::

    netcalc.mutualInformation(X, I, ab, k, N, xd, d, platform)
    
Each of the arguments are described below:

X
  The input data, converted to a form that NetSci can use directly. 
  Must be of type cuarray.FloatCuArray().
  
I
  The output MI, which is passed by reference for the function to fill out, 
  with entries for each pair of distributions specified for an MI calculation.
  Must be of type cuarray.FloatCuArray().

ab
  An array that represents which pairs of distributions to compute. This can be adjusted
  to exclude the computation of certain pairs of distributions, if desired.
  Must be of type cuarray.IntCuArray().
  
k
  This is the "k" of "k-nearest-neighbors" - that is - how many neighbors to each point
  are taken in Kraskov's algorithm. Kraskov recommends a value of "k" between 2 and 4, 
  although situations where k=6 have also been used. In general, a lower value of "k"
  reduces bias, but increases noise, and a higher value of "k" reduces noise, but 
  increases bias for non-independent distributions. If testing for independence, bias
  is not such an issue, so k can be as large as N/2, according to Kraskov et al., 
  where N is the number of data points.
  Must be of type int, greater than zero.

N
  The number of data points sampled from each distribution.
  Must be of type int greater than zero.

xd
  The number of distributions per MI calculation. At this time, only the MI of two
  distributions can be calculated at a time in NetSci, so xd=2.
  Must be of type int, and only xd=2 is currently supported in Netsci.
  
d
  Dimensionality of the data. In the case of 1D Gaussians, the dimensionality is equal to
  one. In contrast, for example, the positions of atoms in a protein would be three-dimensional
  data.
  Must be of type int, greater than zero.

platform
  Which platform to use. If the GPU platform is desired, set this value to netcalc.GPU_PLATFORM,
  otherwise, if the CPU platform is desired, set this value to netcalc.CPU_PLATFORM.

netcalc.generalizedCorrelation
------------------------------

A call to generalizedCorrelation will be formatted as follows::

    netcalc.generalizedCorrelation(X, R, ab, k, n, d, xd, platform)

The parameters ab, k, d, xd, and platform are the same as in the previous section 
"netcalc.mutualInformation". The new, or different, parameters are:

X
  The input data, But unlike with the mutualInformation function above, this "X" is 
  obtained from a netchem.Graph() object, specificall from the nodeCoordinates() method.
  
R
  The output correlation matrix, which is passed by reference for the function to 
  fill out, with entries for each pair of distributions specified for a generalized
  correlation calculation.
  Must be of type cuarray.FloatCuArray().
  
n
  The number of frames for the loaded trajectory.
