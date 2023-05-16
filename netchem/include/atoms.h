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
    Atoms();

    void addAtom(Atom *atom);

    int numAtoms() const;

    Atom *at(int atomIndex);

    int numUniqueTags() const;

    std::vector<Atom*> &atoms();

private:
    friend nlohmann::adl_serializer<Atoms*>;
    std::vector<Atom*> atoms_;
    std::set<std::string> uniqueTags_;
};


#endif //NETSCI_ATOMS_H
