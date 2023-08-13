//
// Created by astokely on 5/2/23.
//

#ifndef NETSCI_NETWORK_H
#define NETSCI_NETWORK_H

#include <map>
#include "node.h"


class Network {
public:
    /**
     * \brief Default constructor for Network.
     *
     * Constructs an empty Network object.
     */
    Network();

    /**
     * \brief Destructor for Network.
     */
    ~Network();

    /**
     * \brief Initialize the Network with trajectory and topology files.
     *
     * Initializes the Network by loading trajectory and topology files.
     *
     * \param trajectoryFile Path to the trajectory file.
     * \param topologyFile Path to the topology file.
     * \param firstFrame Index of the first frame to consider.
     * \param lastFrame Index of the last frame to consider.
     * \param stride Stride between frames.
     */
    void init(
            const std::string &trajectoryFile,
            const std::string &topologyFile,
            int firstFrame,
            int lastFrame,
            int stride=1
    );

    /**
     * \brief Get the number of nodes in the Network.
     *
     * Returns the number of nodes in the Network.
     *
     * \return The number of nodes.
     */
    int numNodes() const;

    /**
     * \brief Get the node coordinates as a CuArray.
     *
     * Returns a pointer to the CuArray object containing the node coordinates.
     *
     * \return A pointer to the CuArray containing the node coordinates.
     */
    CuArray<float>* nodeCoordinates();

    /**
     * \brief Get a reference to the vector of nodes in the Network.
     *
     * Returns a reference to the vector of nodes in the Network.
     *
     * \return A reference to the vector of nodes.
     */
    std::vector<Node*>& nodes();

    /**
     * \brief Get the number of frames in the Network.
     *
     * Returns the number of frames in the Network.
     *
     * \return The number of frames.
     */
    int numFrames() const;

    /**
     * \brief Get the node corresponding to the Atom with the given index.
     *
     * Returns a pointer to the Node object that the Atom with the
     * specified index is part of
     *
     * \param atomIndex The index of the Atom.
     * \return A pointer to the Node corresponding to the Atom index.
     */
    Node* nodeFromAtomIndex(int atomIndex);

    /**
     * \brief Get the Atoms object associated with the Network.
     *
     * Returns a pointer to the Atoms object associated with the Network.
     *
     * \return A pointer to the Atoms object.
     */
    Atoms* atoms() const;

    /**
     * \brief Parse a PDB file to populate the Network.
     *
     * Parses the specified PDB file to populate the Network with Atom and Node objects.
     *
     * \param fname Path to the PDB file.
     */
    void parsePdb(const std::string& fname);

    /**
     * \brief Parse a DCD file to populate the Network.
     *
     * Parses the specified DCD file to populate the Network with node coordinates.
     *
     * \param fname Path to the node coordinates file.
     * \param firstFrame Index of the first frame to consider.
     * \param lastFrame Index of the last frame to consider.
     * \param stride Stride between frames.
     */
    void parseDcd(
            const std::string &nodeCoordinates,
            int firstFrame,
            int lastFrame,
            int stride
    );

    /**
     * \brief Save the Network as a JSON file.
     *
     * Saves the Network as a JSON file.
     *
     * \param jsonFile Path to the JSON file.
     */
    void save(const std::string& jsonFile);

    /**
     * \brief Load a Network from a JSON file.
     *
     * Loads a Network from the specified JSON file.
     *
     * \param jsonFile Path to the JSON file.
     */
    void load(const std::string& jsonFile);

    /**
     * \brief Set the node coordinates from a file.
     *
     * Sets the node coordinates from the specified node coordinates file.
     *
     * \param nodeCoordinatesFile Path to the node coordinates file.
     */
    void nodeCoordinates(const std::string& nodeCoordinatesFile);

private:
    friend nlohmann::adl_serializer<Network*>;
    std::vector<Node*> nodeAtomIndexVector_; // Vector of nodes.
    // The position of each node corresponds to the index of an Atom
    // in the Node.
    std::vector<Node*> nodes_; // Vector of nodes sorted by node index.
    int numNodes_; // Number of nodes.
    int numFrames_; // Number of frames.
    CuArray<float>* nodeCoordinates_; // Coordinates of nodes.
    Atoms* atoms_; // Atoms object.
};

#endif //NETSCI_NETWORK_H
