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
For this example, we will use the first node in the network.
"""
node = network.nodes()[0]

"""
To get a vector of all Atom objects in the node, use the atoms() method.
Unlike the atoms method in the Network class, this method returns a vector
of Atom objects instead of an Atoms object. This is for memory efficiency.
"""
atoms = node.atoms()

"""
The Atom object vector is iterable, indexable, and can be converted 
to a built-in Python list/tuple using the explicit list and tuple constructors.
"""
atoms_list = list(atoms)
atoms_tuple = tuple(atoms)

"""
Iterate over all atoms in the node and print their information. 
For more information on the Atom class, see the Atom class documentation.
"""
for atom in atoms:
    print(atom)
    print("-" * 80)
