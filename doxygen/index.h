/*!
\mainpage NetSci: A Toolkit for High Performance Scientific Network Analysis Computation

\tableofcontents

\section overview_sec Overview

<hr>

NetSci is a specialized toolkit designed for advanced network analysis in computational sciences. Utilizing the
capabilities of modern GPUs, it offers a powerful and efficient solution for processing computationally demanding
network analysis metrics while delivering state-of-the-art performance.

\section install_sec Installation

<hr>

NetSci is designed with a focus on ease of installation and long-term stability, ensuring compatibility with Linux
systems featuring CUDA-capable GPUs (compute capability 3.5 and above). It leverages well-supported core C++ and Python
libraries to maintain simplicity and reliability.

1. **Download Miniconda Installation Script**:
   \code
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   \endcode

2. **Execute the Installation Script**:
   \code
   bash Miniconda3-latest-Linux-x86_64.sh
   \endcode

3. **Update Environment Settings**:
   \code
   source ~/.bashrc
   \endcode

4. **Install Git with Conda**:
   \code
   conda install -c conda-forge git
   \endcode

5. **Clone the NetSci Repository**:
   \code
   git clone https://github.com/netscianalysis/netsci.git
   \endcode

6. **Navigate to the NetSci Root Directory**:
   \code
   cd netsci
   \endcode

7. **Create NetSci Conda Environment**:
   \code
   conda env create -f netsci.yml
   \endcode

8. **Activate NetSci Conda Environment**:
   \code
   conda activate netsci
   \endcode

9. **Create CMake Build Directory**:
   \code
   mkdir build
   \endcode

10. **Set NetSci Root Directory Variable**:
    \code
    NETSCI_ROOT=$(pwd)
    \endcode

11. **Navigate to the CMake Build Directory**:
    \code
    cd ${NETSCI_ROOT}/build
    \endcode

12. **Compile CUDA Architecture Script**:
    \code
    nvcc ${NETSCI_ROOT}/build_scripts/cuda_architecture.cu -o cuda_architecture
    \endcode

13. **Set CUDA Architecture Variable**:
    \code
    CUDA_ARCHITECTURE=$(./cuda_architecture)
    \endcode

14. **Configure the Build with CMake**:
    \code
    cmake .. -DCONDA_DIR=$CONDA_PREFIX -DCUDA_ARCHITECTURE=${CUDA_ARCHITECTURE}
    \endcode

15. **Build NetSci**:
    \code
    cmake --build . -j
    \endcode

16. **Build NetSci Python Interface**:
    \code
    make python
    \endcode

17. **Test C++ and CUDA Backend**:
    \code
    ctest
    \endcode

18. **Run Python Interface Tests**:
    \code
    cd ${NETSCI_ROOT}
    pytest
    \endcode

<hr>

 * \section theory Theory
 * Mutual information is used to measure how much two random variables
 * are correlated, including both linear and non-linear relationships.
 * Imagine we have a set of data pairs \f$(x_i, y_i)\f$, where each pair
 * is an independent realization of random variables \f$(X, Y)\f$.
 * These variables follow a distribution \f$\mu(x, y)\f$.
 * Shannon entropy, denoted as \f$H(X)\f$, is calculated using:
 * \f[
 * H(X) = -\int\mu(x)\log\mu(x)dx
 * \f]
 * where the logarithm's base determines the information's unit
 * (bits, nats, etc.). We use the natural logarithm in our context.
 * Mutual information, \f$I(X, Y)\f$, is defined as:
 * \f[
 * I(X, Y) = H(X) + H(Y) - H(X, Y)
 * \f]
 * This value indicates how strongly \f$X\f$ and \f$Y\f$ are connected.
 * If they are completely independent, \f$I(X, Y)\f$ equals zero.
 * Often, we don't know \f$\mu\f$ exactly and need to estimate it.
 * Assuming \f$\mu\f$ is uniform, we approximate \f$H(X)\f$ with:
 * \f[
 *    \widehat{H}(X) = -\frac{1}{N}\sum_{i=1}^N\widehat{\log(\mu(x_i))}
 * \f]
 * We use a k-nearest neighbor estimator for this purpose.
 * To calculate the probability distributions necessary for these
 * estimations, we consider the distances to a data point's nearest
 * neighbors in both X and Y dimensions, and compute probabilities based
 * on these distances.
 * <hr>
 *
 * \section Algorithms
 *
 * \subsection mutual_information_gpu Parallel Mutual Information
 * | Variable | Description |
 * |----------|-------------|
 * | `Xa`     | Array containing the data points for the first random variable in the mutual information calculation. |
 * | `Xb`     | Array containing the data points for the second random variable in the mutual information calculation. |
 * | `k`      | Integer specifying the number of nearest neighbors to consider for each data point. |
 * | `n`      | Integer representing the total number of data points in each of the random variables `Xa` and `Xb`. |
 * | `nXa`    | Array for storing the count of data points in `Xa` within a radius of epsilon_Xa / 2 for each point. |
 * | `nXb`    | Array for storing the count of data points in `Xb` within a radius of epsilon_Xb / 2 for each point. |
 * | `s_argMin` | Shared memory array used to store the indices of the nearest neighbors during the calculation. |
 * | `s_min`  | Shared memory array used to store the minimum distances calculated in the search for nearest neighbors. |
 * | `s_epsXa` | Shared memory array to store the maximum distance (epsilon) within `Xa` for the k-th nearest neighbors. |
 * | `s_epsXb` | Shared memory array to store the maximum distance (epsilon) within `Xb` for the k-th nearest neighbors. |
 * | `s_nXa`  | Shared memory array used to temporarily store neighbor counts for `Xa` within each CUDA block. |
 * | `s_nXb`  | Shared memory array used to temporarily store neighbor counts for `Xb` within each CUDA block. |
 *
 * \subsubsection step1 Step 1: Initialization
 * - \b Input: Arrays \c Xa, \c Xb, integers \c k, \c n, output arrays \c nXa, \c nXb
 * - \b Output: Updated \c nXa, \c nXb with neighbor counts for each point in \c Xa, \c Xb
 * - Initialize shared memory arrays: \c s_argMin[1024], \c s_min[1024], \c s_epsXa[1], \c s_epsXb[1], \c s_nXa[1024], \c s_nXb[1024]
 *
 * \subsubsection step2 Step 2: Parallel Processing
 * For each data point \c i processed in parallel CUDA blocks:
 *   - Load \c Xa[i] and \c Xb[i] into registers \c r_Xai, \c r_Xbi
 *   - Set local thread index \c localThreadIndex = threadIdx.x
 *   - Initialize \c s_nXa[localThreadIndex], \c s_nXb[localThreadIndex] to zero
 *   - If \c localThreadIndex == 0, set \c s_epsXa[0], \c s_epsXb[0] to zero
 *
 * \subsubsection step3 Step 3: Load Data in Chunks
 * - Iterate over \c Xa, \c Xb in chunks, loading into \c r_Xa, \c r_Xb
 * - Synchronize threads using \c __syncthreads()
 *
 * \subsubsection step4 Step 4: Find k Nearest Neighbors
 * - Initialize \c localMin = RAND_MAX, \c localArgMin = 0
 * - Iterate over chunks, updating \c localMin, \c localArgMin based on distance \c dX between \c r_Xai, \c r_Xbi and chunk data
 * - Update shared memory \c s_min, \c s_argMin
 * - Perform parallel reduction to find global minimum distance and corresponding index
 * - Update \c s_epsXa, \c s_epsXb and mark processed points in \c r_Xa, \c r_Xb as needed
 * - Synchronize threads using \c __syncthreads()
 *
 * \subsubsection step5 Step 5: Increment Neighbor Counts
 * - Iterate over chunks, incrementing \c s_nXa, \c s_nXb based on distance conditions to \c r_Xai, \c r_Xbi
 * - Synchronize threads using \c __syncthreads()
 * - Perform parallel reduction on \c s_nXa, \c s_nXb
 *
 * \subsubsection step6 Step 6: Update Global Counts
 * - If \c localThreadIndex == 0, update \c nXa[i], \c nXb[i] with reduced counts from \c s_nXa[0], \c s_nXb[0]
 * - Synchronize threads using \c __syncthreads()
 *
 * \section tutorials Tutorials
 * \subsection mutual_information_gpu Mutual Information
 * ```{.py}
 * from pathlib import Path
 *
 * import numpy as np
 *
 * from netcalc import mutualInformation
 * from cuarray import floatCuArray, IntCuArray
 * from netchem import Network, data_files
 *
 * dcd, pdb = data_files("pyro")
 *
 * trajectory_file = str(dcd)
 * topology_file = str(pdb)
 * first_frame = 0
 * last_frame = 999
 * stride = 1
 *
 * network = Network()
 * network.init(
 *    trajectory_file,
 *    topology_file,
 *    first_frame,
 *    last_frame,
 *    stride
 * )
 *
 * num_nodes = network.numNodes()
 * num_frames = network.numNodes()
 * k = 4
 *
 *
 * ```
 */