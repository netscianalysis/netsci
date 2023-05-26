//
// Created by astokely on 5/4/23.
//

#ifndef NETSCI_ATOM_H
#define NETSCI_ATOM_H

#include <string>
#include "cuarray.h"
#include "nlohmann/json.hpp"

class Atom {
public:
    /**
     * \brief Default constructor for Atom.
     *
     * Constructs an empty Atom object.
     */
    Atom();

    /**
 * \brief Constructor for Atom with PDB line.
 *
 * Constructs an Atom object using the provided PDB line. The constructor parses the PDB line to extract
 * the relevant atom information, such as index, name, element, residue name, residue ID, chain ID, segment ID,
 * temperature factor, occupancy, serial number, atom tag, mass, and hash.
 *
 * The PDB line should follow the standard PDB format for ATOM records as described in Section 9 of the PDB file format documentation:
 * \see https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
 *
 * \param pdbLine The PDB line containing atom information in the standard PDB format.
 */
    explicit Atom(const std::string &pdbLine);

    /**
     * \brief Constructor for Atom with PDB line and atom index.
     *
     * Constructs an Atom object using the provided PDB line and atom index.
     *
     * \param pdbLine    The PDB line containing atom information.
     * \param atomIndex  The atom index.
     */
    Atom(
            const std::string &pdbLine,
            int atomIndex
    );

    /**
     * \brief Get the atom index.
     *
     * Returns the atom index.
     *
     * \return The atom index.
     */
    int index() const;

    /**
     * \brief Get the atom name.
     *
     * Returns the atom name.
     *
     * \return The atom name.
     */
    std::string name();

    /**
     * \brief Get the atom element.
     *
     * Returns the atom element.
     *
     * \return The atom element.
     */
    std::string element();

    /**
     * \brief Get the residue name.
     *
     * Returns the residue name.
     *
     * \return The residue name.
     */
    std::string residueName();

    /**
     * \brief Get the residue ID.
     *
     * Returns the residue ID.
     *
     * \return The residue ID.
     */
    int residueId() const;

    /**
     * \brief Get the chain ID.
     *
     * Returns the chain ID.
     *
     * \return The chain ID.
     */
    std::string chainId();

    /**
     * \brief Get the segment ID.
     *
     * Returns the segment ID.
     *
     * \return The segment ID.
     */
    std::string segmentId();

    /**
     * \brief Get the temperature factor.
     *
     * Returns the temperature factor.
     *
     * \return The temperature factor.
     */
    float temperatureFactor() const;

    /**
     * \brief Get the occupancy.
     *
     * Returns the occupancy.
     *
     * \return The occupancy.
     */
    float occupancy() const;

    /**
     * \brief Get the serial number.
     *
     * Returns the serial number, which is one greater than the atom index.
     *
     * \return The serial number.
     */
    int serial() const;

    /**
     * \brief Get the atom tag.
     *
     * Returns the atom tag, which is the concatenation of the residue name,
     * residue ID, chain ID, and segment ID.
     *
     * \return The atom tag.
     */
    std::string tag();

    /**
     * \brief Get the mass of the atom.
     *
     * Returns the mass of the atom.
     *
     * \return The mass of the atom.
     */
    float mass() const;

    /**
     * \brief Get the hash of the atom.
     *
     * Returns the hash of the atom, which is calculated from the atom tag
     * concatenated with the atom index.
     *
     * \return The hash of the atom.
     */
    unsigned int hash() const;

    /**
     * \brief Get the x-coordinate of the atom.
     *
     * Returns the x-coordinate of the atom at the specified frame.
     *
     * \param coordinates  The CuArray containing the coordinates.
     * \param frame        The frame index.
     * \param numFrames    The total number of frames.
     * \return The x-coordinate of the atom.
     */
    float x(
            CuArray<float> *coordinates,
            int frame,
            int numFrames
    ) const;

    /**
     * \brief Get the y-coordinate of the atom.
     *
     * Returns the y-coordinate of the atom at the specified frame.
     *
     * \param coordinates  The CuArray containing the coordinates.
     * \param frame        The frame index.
     * \param numFrames    The total number of frames.
     * \return The y-coordinate of the atom.
     */
    float y(
            CuArray<float> *coordinates,
            int frame,
            int numFrames
    ) const;

    /**
     * \brief Get the z-coordinate of the atom.
     *
     * Returns the z-coordinate of the atom at the specified frame.
     *
     * \param coordinates  The CuArray containing the coordinates.
     * \param frame        The frame index.
     * \param numFrames    The total number of frames.
     * \return The z-coordinate of the atom.
     */
    float z(
            CuArray<float> *coordinates,
            int frame,
            int numFrames
    ) const;

    /**
     * \brief Load atom information from a JSON file.
     *
     * Loads atom information from the specified JSON file.
     *
     * \param jsonFile The name of the JSON file to load.
     */
    void load(const std::string &jsonFile);

private:
    friend nlohmann::adl_serializer<Atom>;
    friend nlohmann::adl_serializer<Atom *>;

    int _index;                  // Atom index. Atom indexing starts at 0
    std::string _name;           // Atom name (e.g., CA)
    std::string _element;        // Atom element (e.g., C)
    std::string _residueName;    // Residue name (e.g., ALA)
    int _residueId;              // Residue ID (e.g., 1). Residue indexing starts at 1
    std::string _chainId;        // Chain ID (e.g., A)
    std::string _segmentId;      // Segment ID (e.g., A)
    float _temperatureFactor;    // Temperature factor
    float _occupancy;            // Occupancy
    int _serial;                 // Serial number, which is 1 greater than the atom index
    std::string _tag;            // Atom tag, concatenation of residue name, residue ID, chain ID, and segment ID
    float _mass;                 // Mass of the atom
    unsigned int _hash;          // Hash of the atom tag concatenated with the atom index
};

#endif // NETSCI_ATOM_H

