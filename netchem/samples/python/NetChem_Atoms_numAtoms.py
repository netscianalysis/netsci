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
Get the Network's Atoms class object using the Network atoms method.
The Atoms class is a container like class for Atom objects.
The reason for using a custom container class to store Atom objects
instead of using an Atom vector is to have more control over memory
management.
"""
atoms = network.atoms()

"""
Use the Atoms numAtoms method to get the number of atoms in the Atoms
object.
"""
num_atoms = atoms.numAtoms()
print(num_atoms)