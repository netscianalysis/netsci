/*--------------------------------------------------------------------*
 *                         CuRva                                      *
 *--------------------------------------------------------------------*
 * This is part of the GPU-accelerated, random variable analysis      *
 * library CuRva.                                                     *
 * Copyright (C) 2022 Andy Stokely                                    *
 *                                                                    *
 * This program is free software: you can redistribute it             *
 * and/or modify it under the terms of the GNU General Public License *
 * as published by the Free Software Foundation, either version 3 of  *
 * the License, or (at your option) any later version.                *
 *                                                                    *
 * This program is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
 * GNU General Public License for more details.                       *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this program.                                           *
 * If not, see <https://www.gnu.org/licenses/>                        *
 * -------------------------------------------------------------------*/

#ifndef CURVA_NODE_H
#define CURVA_NODE_H

#include "atoms.h"
#include "cuarray.h"

class Node {
public:
    Node();

    ~Node();

    Node(unsigned int numFrames, unsigned int index_);

    void addAtom(Atom *atom, CuArray<float> *coordinates, CuArray<float> *nodeCoordinates);

    std::string tag();

    unsigned int numAtoms() const;

    unsigned int index() const;

    float totalMass() const;

    unsigned int hash() const;

    std::vector<Atom *> atoms() const;

private:

    friend nlohmann::adl_serializer<Node *>;

    friend class Graph;

    unsigned int _numAtoms;

    std::vector<int> atomIndices_;

    unsigned int _index;

    float _totalMass;

    std::string _tag;

    std::vector<Atom *> atoms_;

    unsigned int _hash = 0;

    unsigned int _numFrames;
};

#endif // CURVA_NODE_H
