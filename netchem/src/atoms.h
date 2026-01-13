//
// Created by astokely on 5/4/23.
//

#ifndef NETSCI_ATOMS_H
#define NETSCI_ATOMS_H


#include "atom.h"
#include <vector>
#include <set>
#include "nlohmann/json.hpp"


class Atoms {
public:
    /**
     * \brief Default constructor for Atoms.
     *
     * Constructs an empty Atoms object.
     */
    Atoms();

    /**
     * \brief Add an Atom to the Atoms collection.
     *
     * Adds the specified Atom to the collection of Atoms.
     *
     * \param atom Pointer to the Atom to add.
     */
    void addAtom(Atom* atom);

    /**
     * \brief Get the number of Atoms in the collection.
     *
     * Returns the number of Atoms in the collection.
     *
     * \return The number of Atoms.
     *
     * @PythonExample{NetChem_Atoms_numAtoms.py}
     */
    int numAtoms() const;

    /**
     * \brief Get the Atom with the specified index.
     *
     * Returns a pointer to the Atom with the specified index.
     *
     * \param atomIndex The index of the Atom.
     * \return A pointer to the Atom with the specified index.
     *
     * @PythonExample{NetChem_Atoms_at.py}
     */
    Atom* at(int atomIndex);

    /**
     * \brief Get the number of unique Atom tags.
     *
     * Returns the number of unique Atom tags.
     * Atoms with the same tag belong to the same Node.
     *
     * \return The number of unique Atom tags.
     */
    int numUniqueTags() const;

    /**
     * \brief Get a reference to the vector of Atoms.
     *
     * Returns a reference to the vector of Atoms.
     *
     * \return A reference to the vector of Atoms.
     *
     * @PythonExample{NetChem_Atoms_atoms.py}
     */
    std::vector<Atom*>& atoms();

private:
    friend nlohmann::adl_serializer<Atoms*>;
    std::vector<Atom*> atoms_;
    std::set<std::string> uniqueTags_;
};


#endif //NETSCI_ATOMS_H

