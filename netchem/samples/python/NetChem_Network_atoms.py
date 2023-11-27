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
The Network atoms method returns an Atoms object, which is a container
class for Atom objects. The reason Atom objects are stored in a custom
class instead of a build in container is to keep minimize cache misses,
as the Atoms class ensures that all Atom objects are stored contiguously.
It also allows for more control over memory management. The Atoms class
is iterable and individual Atom objects can be accessed via indexing. For
more information about the Atoms class, see the Atoms class examples and
documentation.
"""
atoms = network.atoms()

"""
Iterate over atoms and print the index and name of each atom.
"""
for atom in atoms:
    print(atom.index(), atom.name())
