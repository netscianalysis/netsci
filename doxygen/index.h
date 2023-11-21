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
 *
 * \subsection introduction Introduction
 *
 * In the realm of network analysis, understanding the intricate relationships and dependencies between elements in a dataset is crucial. The 'netsci' library, developed using CUDA for GPU-accelerated performance, provides a sophisticated toolkit for such analysis. At the heart of this library lies a deep integration of core theoretical concepts from information theory and statistical analysis, tailored to extract meaningful insights from complex network data.
 * This library harnesses the computational power of modern GPUs to
 * perform high-speed calculations, enabling the analysis of
 * large-scale networks that are often encountered in fields like
 * social network analysis, biological network analysis, and communication networks. The underlying theory, rooted in principles of mutual information, Shannon entropy, and generalized correlation, provides a robust framework for understanding the dynamics and structure of networks.
 * The utilization of CUDA not only accelerates computations but also
 * allows for handling intricate calculations involving high-dimensional data, making 'netsci' an ideal choice for researchers and practitioners dealing with complex network systems. The integration of these theoretical concepts with state-of-the-art computational techniques opens new avenues for network analysis, offering insights that were previously challenging to obtain due to computational constraints.
 * In the following sections, we delve into the core theoretical
 * constructs that form the foundation of the 'netsci' library. Starting with Mutual Information, we explore how this measure serves as a cornerstone in understanding and quantifying the relationships between variables in a network. The subsequent sections will further elucidate how these theoretical principles are translated into practical, high-performance tools for network analysis.
 *
 * \subsection mutual_information Mutual Information
 * Mutual information serves as a way to measure correlation, both
 * linear and non-linear, between two random variables. Given a bivariate set of data
 * \f$z_i=(x_i,y_i),i=1,...,N\f$, which can be of any dimension,
 *   we assume that each of the \f$N\f$ elements of the data are
 *   independent
 * and identically distributed realizations of the random variables \f$Z=(X,Y)\f$, and that
 * they are distributed according to \f$\mu(x,y)\f$, a proper smooth function. The marginal densities are
 * \f$\mu(x)=\int \mu(x,y)dy\f$ and \f$\mu(y)=\int \mu(x,y)dx\f$.
 *
 * The Shannon entropy can be defined as
 * \f[
 * H(X)=-\int\mu(x)\log\mu(x)dx
 * \f]
 *
 * where the base of the logarithm depends on the units desired for
 * the information, whether bits (\f$\log_2\f$), nats (\f$\log_e\f$),
 * decimal digits (\f$\log_{10}\f$), or otherwise. In this work, we use the natural logarithm.
 * The mutual information \f$I(X,Y)\f$ is defined as:
 *
 * \f[
 * I(X,Y)=H(X)+H(Y)-H(X,Y)
 * \f]
 *
 * The value of \f$I(X,Y)\f$ measures the strength of the connection between the variables \f$X\f$ and \f$Y\f$;
 * if the two variables were completely independent, then \f$I(X,Y)\f$ would be zero.
 *
 * In most cases, the density distribution \f$\mu\f$ is not known exactly, and must be estimated.
 * Under the condition that \f$\mu\f$ is a uniform distribution, we
 * may approximate \f$H(X)\f$ by a discrete sum,
 * \f[
 *    \widehat{H}(X) = -\frac{1}{N}\sum_{i=1}^N\widehat{\log(\mu(x_i))}
 * \f]
 * Now, an estimate for \f$\widehat{\log(\mu(x_i))}\f$ must be
 * defined. In this paper, we use a k-nearest neighbor estimator,
 * which generalizes well to high-dimensions.
 * In order to rank neighbors of a data point \f$z_i\f$ by nearness,
 * we use the max norm,
 * \f[
 * ||z-z'|| = \max \{ ||x-x'||,||y-y'||\}
 * \f]
 * where we choose to use a similar max norm for \f$||x-x'||\f$ and
 * \f$||y-y'||\f$,
 * although this is not required - a Euclidean norm could be used,
 * for instance.
 * For each data point \f$z_i\f$, let \f$\epsilon_x(i)/2\f$ and
 * \f$\epsilon_y(i)/2\f$ represent
 * the distances from \f$z_i\f$ to its \f$k\f$th nearest neighbor
 * projected onto the X and Y subspaces, respectively.
 * The value \f$p_i\f$ is the integrated density within a distance
 * \f$\epsilon/2\f$ of the point
 * \f$x_i\f$, \f$p_i(\epsilon)=\int_{||\xi-x_i||<\epsilon/2}\mu(\xi)
 * d\xi\f$.
 * Consider the probability distribution
 * \f[
 * P_k(\epsilon_x, \epsilon_y)=P_k^{(b)}(\epsilon_x, \epsilon_y)
 * +P_k^{(c)}(\epsilon_x, \epsilon_y)
 * \f]
 * Specifically, \f$ P_k^{(b)}(\epsilon_x, \epsilon_y) \f$ represents
 * the probability distribution that there
 * are \f$ k-1 \f$ data points within the rectangle \f$
 * x_i\pm\epsilon_x(i)/2 \f$ and \f$ y_i\pm\epsilon_y(i)/2 \f$, a rectangle
 * defined by the \f$ k \f$th nearest neighbor within in the \f$ x
 * \f$ subspace, \f$ N-k-1 \f$ points that are
 * outside a different rectangle defined by \f$ x_i\pm(\epsilon_x(i)
 * +d\epsilon_x)/2 \f$ and
 * \f$ y_i\pm(\epsilon_y(i)+d\epsilon_y)/2 \f$, and one data point in
 * the space between the two rectangles.
 * The probability distribution \f$ P_k^{(c)}(\epsilon_x, \epsilon_y)
 * \f$ is similar, though the rectangles
 * are defined by the \f$ k \f$th nearest neighbor within the \f$ y
 * \f$ subspace, which may be the same, or a different,
 * point used to define \f$ P_k^{(b)}(\epsilon_x, \epsilon_y) \f$.
 * These quantities are then
 * \f[
 *    P_k^{(b)}(\epsilon_x, \epsilon_y) = \begin{pmatrix}
 *    N - 1 \\
 *    k
 *    \end{pmatrix}\left(\frac{d^2[q_i^k]}{d\epsilon_x
 *    d\epsilon_y}\right)\left(1-p_i\right)^{N-1-k}
 * \f]
 * and
 * \f[
 *    P_k^{(c)}(\epsilon_x, \epsilon_y) = (k-1)\begin{pmatrix}
 *    N - 1 \\
 *    k
 *    \end{pmatrix}\left(\frac{d^2[q_i^k]}{d\epsilon_x
 *    d\epsilon_y}\right)\left(1-p_i\right)^{N-1-k}
 * \f]
 * Similar to \f$ p_i \f$, the value \f$ q_i(\epsilon_x, \epsilon_y)
 * \f$ is the integrated density
 * within a tiny rectangle of size \f$ \epsilon_x \times \epsilon_y
 * \f$ centered at \f$ (x_i, y_i) \f$.
 * As mentioned before, \f$ p_i \f$ is the integrated density within
 * a tiny square of side length \f$ \epsilon \f$ - tiny enough
 * that we may assume that \f$ \mu(x) \f$ is constant within:
 * \f[
 * p_i(\epsilon)\approx c_d \epsilon^d \mu (x_i)
 * \f]
 * where \f$ d \f$ is the dimension of \f$ x \f$, and \f$ c_d \f$ is
 * the volume of the \f$ d \f$-dimensional unit ball.
 * For the maximum norm used in this study, we simply use \f$ c_d=1
 * \f$. In this case,
 * \f[
 * I(X_1,X_2) = \psi(k) - 1/k - \langle \psi(n_x) + \psi(n_y) \rangle
 * + \psi(N)
 * \f]
 * where \f$ \psi(x) \f$ is the digamma function, and \f$ n_x(i) \f$
 * and \f$ n_y(i) \f$ are the number of
 * points with distance less than or equal to \f$ \epsilon_x(i)/2 \f$
 * and \f$ \epsilon_y(i)/2 \f$, respectively.
 * \subsection generalized_correlation Generalized Correlation
 * The Pearson product-moment correlation is defined as
 * \f[
 *    r(X, Y) = \frac{(X - \mu_{X})(Y - \mu_{Y})}{\sigma_{X}\sigma_{Y}},
 * \f]
 *  where \f$ \mu_{X} \f$ is the average of random variable \f$ X \f$,
 * \f$ \mu_{Y} \f$ is the mean of random variable
 * \f$ Y \f$, \f$ \sigma_{X} \f$ is the standard deviation of \f$ X \f$ and \f$ \sigma_{Y} \f$ is the standard deviation of
 * \f$ Y \f$. The Pearson correlation defines the best linear fit between data sets sampled from \f$ X \f$ and \f$ Y \f$.
 * As mentioned before, the Pearson correlation suffers a number of insufficiencies when used for data that is related
 * nonlinearly, and sets of related data that oscillate in non-parallel directions. The Mutual Information (MI) can address
 * these shortcomings. In order to provide an equivalent quantity to \f$ r \f$ in eq. \ref{pearson_correlation}, we define the
 * generalized correlation coefficient \f$ r_{\textrm{MI}} \f$:
 * \f[
 *    r_{\textrm{MI}}(X, Y) = \left(1-e^{-\frac{2I(X,Y)}{d}}\right)
 *    ^{\frac{1}{2}}
 * \f]
 * Where \f$ d \f$ is the dimensionality of the data.
 *
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
 */