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

"""
The load method can be used to deserialize, or load a Network object from
a Network serialization JSON file. Like the save method,
the load method takes a single argument, which is the absolute path
to the JSON file that the Network object will be loaded from.
The network's node coordinates must be loaded separate from the 
.npy file they were saved to using the Network nodeCoordinates method.
"""
network_copy = Network()
network_copy.load(str(network_json))
"""Load the network node coordinates from the serialized coordinates .npy file."""
network_copy.nodeCoordinates(network_node_coordinates_npy)

"""
Print the two network Node objects side by side to verify that they are the same.
"""
for node_copy, node in zip(network_copy.nodes(), network.nodes()):
    print(node)
    print(node_copy)
    for frame in range(network_copy.numFrames()):
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
