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
The nodeCoordinates method returns a Mx3*N float CuArray,
where M is the number of nodes in the network and N is the number
of frames. For each node, the coordinates are stored in a linear,
row-major format, so to access the x, y, and z coordinates of the 
i-th node in the j-th frame, you would use the following syntax:

    x = nodeCoordinates[i][j]
    y = nodeCoordinates[i][j+N]
    z = nodeCoordinates[i][j+2*N]
"""
nodeCoordinates = network.nodeCoordinates()

"""
Get the x, y, and z coordinates of node 5 in frame 50.
"""
frame_index = 50
node_index = 4
y_offset = network.numFrames()
z_offset = 2*network.numFrames()
x = nodeCoordinates[node_index][frame_index]
y = nodeCoordinates[node_index][frame_index+y_offset]
z = nodeCoordinates[node_index][frame_index+z_offset]
print(f'Node 5 in frame 50 coordinates: ({x}, {y}, {z})')
print(f'Node 5 in frame 50 coordinates: ({x}, {y}, {z})')

