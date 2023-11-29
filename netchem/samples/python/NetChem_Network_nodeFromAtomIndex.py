from netchem import Network, data_files

"""
If operating on a CuArray, you must import the 
appropriate CuArray class. In this case, we operate on
nodeCoordinates, which is a FloatCuArray.
"""
from cuarray import FloatCuArray

"""
1000 frame Molecular Dynamics trajectory of a pyrophosphatase system.
The dcd is a binary coordinate file, and the pdb is a single frame 
topology file.
"""
dcd, pdb = data_files('pyro')

"""
Construct a network using the entire trajectory.
If you want to use only a subset of the trajectory, you can specify
the first frame, last frame, and stride.
"""
network = Network()
network.init(
    trajectoryFile=str(dcd), # Convert the dcd Path object to a string
    topologyFile=str(pdb), # Convert the pdb Path object to a string
    firstFrame=0,
    lastFrame=999,
    stride=1
)

"""
Use the nodeFromAtomIndex method to get the node index of the atom.
The below example returns the Node object that the fifth atom (index 4) belongs to.
See the Node class documentation for information on the Node class properties 
and methods.
"""
node = network.nodeFromAtomIndex(4)
