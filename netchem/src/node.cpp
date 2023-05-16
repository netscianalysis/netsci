/*--------------------------------------------------------------------*
 *                         HiPSci                                      *
 *--------------------------------------------------------------------*
	HiPSci is a collection of data structures and algorithms for
	high-performance scientific computing.                                              *
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

#include <utility>
#include <boost/range/irange.hpp>
#include <iostream>
#include <fstream>
#include "node.h"
#include "utils.h"
#include "serializer.h"

Node::Node() = default;

Node::~Node() {
}

Node::Node(unsigned int numFrames, unsigned int index_) {
    this->_index = index_;
    this->_totalMass = 0.0;
    this->_numFrames = numFrames;
}

void Node::addAtom(Atom *atom, CuArray<float> *coordinates, CuArray<float> *nodeCoordinates) {
    atoms_.push_back(atom);
    this->_tag = atom->tag();
    this->_numAtoms = atoms_.size();
    int trajectoryNumAtoms = coordinates->n()
                             / (
                                     3
                                     * this->_numFrames
                             );
    float mass = atom->mass();
    this->_totalMass += mass;
    for (
        auto frame: boost::irange(
            this->_numFrames
    )) {
        int xCoordinatesIndex = static_cast<int>(atom->index()
                                                 * this->_numFrames
                                                 + frame);
        int yCoordinatesIndex = static_cast<int>(atom->index()
                                                 * this->_numFrames
                                                 + frame
                                                 + this->_numFrames
                                                   * trajectoryNumAtoms);
        int zCoordinatesIndex = static_cast<int>(atom->index()
                                                 * this->_numFrames
                                                 + frame
                                                 + 2
                                                   * this->_numFrames
                                                   * trajectoryNumAtoms);
        float nodeX = nodeCoordinates->get(
                this->_index, frame
        ) + (mass * coordinates->get(
                0, xCoordinatesIndex
        ));
        float nodeY = nodeCoordinates->get(
                this->_index, this->_numFrames + frame
        ) + (mass * coordinates->get(
                0, yCoordinatesIndex
        ));
        float nodeZ = nodeCoordinates->get(
                this->_index, 2 * this->_numFrames + frame
        ) + (mass * coordinates->get(
                0, zCoordinatesIndex
        ));
        nodeCoordinates->set(
                nodeX,
                this->_index, frame
        );
        nodeCoordinates->set(
                nodeY,
                this->_index, this->_numFrames + frame
        );
        nodeCoordinates->set(
                nodeZ,
                this->_index, 2 * this->_numFrames + frame
        );
    }
    this->_hash = utils::hashString(
            this->_tag
    );
}

std::string Node::tag() {
    return _tag;
}

unsigned int Node::numAtoms() const {
    return _numAtoms;
}

unsigned int Node::index() const {
    return _index;
}

float Node::totalMass() const {
    return this->_totalMass;
}

unsigned int Node::hash() const {
    return this->_hash;
}

std::vector<Atom *> Node::atoms() const {
    return this->atoms_;
}

