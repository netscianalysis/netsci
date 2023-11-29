from netchem import Network, data_files

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
Use the Network nodes method to get a NodeVector object, which
is essentially a list of Node objects, and can be converted to a
Python list via list(NodeVector). The NodeVector is iterable and
individual Node objects can be accessed via indexing.
"""
nodes = network.nodes()

"""
Iterate over nodes and print the index and number of 
atoms in each of each node. See the Node class examples
and documentation for more information about Node methods.
"""
for node in nodes:
    print(node.index(), node.numAtoms())