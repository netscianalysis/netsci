from pathlib import Path

from netchem import Network, data_files, Node

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
Network objects can be serialized and saved to a JSON file 
using the save method. The save method takes a single argument,
which is the absolute path to the JSON file that the Network
object will be saved to. If using the pathlib module,
make sure to convert the Path object to a string prior
to calling the save method. Another import note is that
the save method does not automatically save the node coordinates,
which have to be saved separately using the CuArray 
save method. 
"""
cwd = Path.cwd()
network_json = str(cwd / 'network.json')

network_node_coordinates_npy = str(cwd / 'network_node_coordinates.npy')
network.save(network_json)
"""Save the network node coordinates to a .npy file."""
network.nodeCoordinates().save(network_node_coordinates_npy)

