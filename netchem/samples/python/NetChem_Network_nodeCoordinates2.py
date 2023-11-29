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

"""Current working directory"""
cwd = Path.cwd()

"""Save the network node coordinates to a .npy file."""
network_node_coordinates_npy = str(cwd / 'network_node_coordinates.npy')
network.nodeCoordinates().save(network_node_coordinates_npy)

"""
Create a new Network object and load the serialized network node
coordinates from the .npy file.
"""
network_copy = Network()
network_copy.nodeCoordinates(network_node_coordinates_npy)

"""
Print the two network Node objects' coordinates side by side to verify that they are the same.
"""
for node in network.nodes():
    for frame in range(network.numFrames()):
        x_idx = frame
        y_idx = frame + network_copy.numFrames()
        z_idx = frame + 2*network_copy.numFrames()
        x = network_copy.nodeCoordinates()[node.index()][x_idx]
        y = network_copy.nodeCoordinates()[node.index()][y_idx]
        z = network_copy.nodeCoordinates()[node.index()][z_idx]
        print(f'Node {node.index()} in frame {frame} coordinates: ({x}, {y}, {z})', end=' ')
        x = network.nodeCoordinates()[node.index()][x_idx]
        y = network.nodeCoordinates()[node.index()][y_idx]
        z = network.nodeCoordinates()[node.index()][z_idx]
        print(f'Node {node.index()} in frame {frame} coordinates: ({x}, {y}, {z})')
    print("-"*80)
