/**
 * \file node.h
 * \brief Defines the Node class representing a node in a graph.
 */

#ifndef CURVA_NODE_H
#define CURVA_NODE_H

#include "atoms.h"
#include "cuarray.h"

/**
 * \class Node
 * \brief Represents a node in a graph.
 */
class Node {
public:
    /**
     * \brief Default constructor for Node.
     */
    Node();

    /**
     * \brief Destructor for Node.
     */
    ~Node();

    /**
     * \brief Constructor for Node with specified number of frames and index.
     *
     * \param numFrames Number of frames.
     * \param index_ Index of the node.
     */
    Node(unsigned int numFrames, unsigned int index_);

    /**
     * \brief Add an Atom to the Node.
     *
     * Adds the specified Atom to the Node along with its corresponding coordinates.
     *
     * \param atom Pointer to the Atom object.
     * \param coordinates Pointer to the coordinates array.
     * \param nodeCoordinates Pointer to the node coordinates array.
     */
    void addAtom(Atom* atom, CuArray<float>* coordinates, CuArray<float>* nodeCoordinates);

    /**
     * \brief Get the tag of the Node.
     *
     * Returns the tag of the Node, which represents its unique identifier.
     *
     * \return The tag of the Node.
     *
     * @PythonExample{NetChem_Node_tag.py}
     */
    std::string tag();

    /**
     * \brief Get the number of Atoms in the Node.
     *
     * Returns the number of Atoms contained in the Node.
     *
     * \return The number of Atoms in the Node.
     *
     * @PythonExample{NetChem_Node_numAtoms.py}
     */
    unsigned int numAtoms() const;

    /**
     * \brief Get the index of the Node.
     *
     * Returns the index of the Node.
     *
     * \return The index of the Node.
     *
     * @PythonExample{NetChem_Node_index.py}
     */
    unsigned int index() const;

    /**
     * \brief Get the total mass of the Node.
     *
     * Returns the total mass of the Node, calculated as the sum of the masses
     * of all the Atoms in the Node.
     *
     * \return The total mass of the Node.
     *
     * @PythonExample{NetChem_Node_totalMass.py}
     */
    float totalMass() const;

    /**
     * \brief Get the hash value of the Node.
     *
     * Returns the hash value of the Node, which is a unique identifier
     * based on its tag and index.
     *
     * \return The hash value of the Node.
     */
    unsigned int hash() const;

    /**
     * \brief Get a vector of pointers to the Atoms in the Node.
     *
     * Returns a vector of pointers to the Atoms contained in the Node.
     *
     * \return A vector of pointers to the Atoms in the Node.
     *
     * @PythonExample{NetChem_Node_atoms.py}
     */
    std::vector<Atom*> atoms() const;

private:
    friend nlohmann::adl_serializer<Node*>;

    friend class Network;

    unsigned int _numAtoms; // Number of Atoms in the Node.
    std::vector<int> atomIndices_; // Indices of the Atoms in the Node.
    unsigned int _index; // Index of the Node.
    float _totalMass; // Total mass of the Node.
    std::string _tag; // Tag of the Node.
    std::vector<Atom*> atoms_; // Pointers to the Atoms in the Node.
    unsigned int _hash = 0; // Hash value of the Node.
    unsigned int _numFrames; // Number of frames.
};

#endif // CURVA_NODE_H
