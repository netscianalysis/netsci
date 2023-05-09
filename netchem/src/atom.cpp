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


#include <numeric>
#include <sstream>
#include <algorithm>
#include "atom.h"
#include "chemical_properties.h"
#include "utils.h"


/*
 *COLUMNS        DATA TYPE       CONTENTS
--------------------------------------------------------------------------------
 1 -  6        Record name     "ATOM  "
 7 - 11        Integer         Atom serial number.
13 - 16        Atom            Atom name.
17             Character       Alternate location indicator.
18 - 20        Residue name    Residue name.
22             Character       Chain identifier.
23 - 26        Integer         Residue sequence number.
27             AChar           Code for insertion of residues.
31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
55 - 60        Real(6.2)       Occupancy.
61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
73 - 76        LString(4)      Segment identifier, left-justified.
77 - 78        LString(2)      Element symbol, right-justified.
 */

Atom::Atom() {
}

Atom::Atom(const std::string &pdbLine) {
    ChemicalProperties chemicalProperties;
    _serial = utils::strToInt(
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            6,
                            5
                    )
            )
    );
    _index = _serial
             - 1;
    _name = utils::removeWhiteSpace(
            pdbLine.substr(
                    12,
                    3
            )
    );
    _residueName = utils::removeWhiteSpace(
            pdbLine.substr(
                    17,
                    3
            )
    );
    _chainId = utils::removeWhiteSpace(
            pdbLine.substr(
                    21,
                    1
            ));
    _residueId = utils::strToInt(
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            22,
                            4
                    )
            )
    );
    _occupancy = utils::strToDouble(
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            54,
                            6
                    )
            )
    );
    _temperatureFactor = utils::strToDouble(
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            60,
                            6
                    )
            )
    );
    _segmentId = utils::removeWhiteSpace(
            pdbLine.substr(
                    72,
                    4
            )
    );
    utils::removeWhiteSpace(_segmentId);
    _element = utils::removeWhiteSpace(
            pdbLine.substr(
                    76,
                    2
            )
    );
    std::transform(
            std::begin(this->_element),
            std::end(this->_element),
            std::begin(this->_element),
            []
                    (char const &c) {
                return std::tolower(c);
            }
    );
    this->_element[0] = std::toupper(this->_element[0]);
    this->_tag = (
            this->_residueName
            + "_"
            +
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            22,
                            4
                    ))
            + "_"
            +
            this->_chainId
            + "_"
            +
            this->_segmentId
    );
    this->_mass = 1000.0
                  * chemicalProperties.atomicWeight(this->_element);
    std::stringstream hashStringStream;
    hashStringStream
            << this->_index
            << "_"
            << this->_tag;
    this->_hash = utils::hashString(
            hashStringStream.str()
    );
}

Atom::Atom(
        const std::string &pdbLine,
        int atomIndex
) {
    ChemicalProperties chemicalProperties;
    _serial = atomIndex
              + 1;
    _index = atomIndex;
    _name = utils::removeWhiteSpace(
            pdbLine.substr(
                    12,
                    3
            )
    );
    _residueName = utils::removeWhiteSpace(
            pdbLine.substr(
                    17,
                    3
            )
    );
    _chainId = utils::removeWhiteSpace(
            pdbLine.substr(
                    21,
                    1
            ));
    _residueId = utils::strToInt(
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            22,
                            4
                    )
            )
    );
    _occupancy = utils::strToDouble(
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            54,
                            6
                    )
            )
    );
    _temperatureFactor = utils::strToDouble(
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            54,
                            6
                    )
            )
    );
    _segmentId = utils::removeWhiteSpace(
            pdbLine.substr(
                    72,
                    4
            )
    );
    utils::removeWhiteSpace(_segmentId);
    _element = utils::removeWhiteSpace(
            pdbLine.substr(
                    76,
                    2
            )
    );
    std::transform(
            std::begin(this->_element),
            std::end(this->_element),
            std::begin(this->_element),
            []
                    (char const &c) {
                return std::tolower(c);
            }
    );
    this->_element[0] = std::toupper(this->_element[0]);
    this->_tag = (
            this->_residueName
            + "_"
            +
            utils::removeWhiteSpace(
                    pdbLine.substr(
                            22,
                            4
                    ))
            + "_"
            +
            this->_chainId
            + "_"
            +
            this->_segmentId
    );
    this->_mass = 1000.0
                  * chemicalProperties.atomicWeight(this->_element);
    std::stringstream hashStringStream;
    hashStringStream
            << this->_index
            << "_"
            << this->_tag;
    this->_hash = utils::hashString(
            hashStringStream.str()
    );
}


int Atom::index() const {
    return _index;
}

std::string Atom::name() {
    return _name;
}

std::string Atom::element() {
    return _element;
}

std::string Atom::residueName() {
    return _residueName;
}

int Atom::residueId() const {
    return _residueId;
}

std::string Atom::chainId() {
    return _chainId;
}

std::string Atom::segmentId() {
    return _segmentId;
}

float Atom::temperatureFactor() const {
    return _temperatureFactor;
}

float Atom::occupancy() const {
    return _occupancy;
}

int Atom::serial() const {
    return _serial;
}


std::string Atom::tag() {
    return _tag;
}

unsigned int Atom::hash() const {
    return this->_hash;
}

float Atom::x(
        CuArray<float> *coordinates,
        int frame,
        int numFrames
) const {
    return coordinates->get(0, this->_index * numFrames + frame);
}

float Atom::y(
        CuArray<float> *coordinates,
        int frame,
        int numFrames
) const {
    int numAtoms = coordinates->n()
                   / (
                           3
                           * numFrames
                   );
    return coordinates->get(0, numFrames * numAtoms + this->_index * numFrames + frame);
}

float Atom::z(
        CuArray<float> *coordinates,
        int frame,
        int numFrames
) const {
    int numAtoms = coordinates->n()
                   / (
                           3
                           * numFrames
                   );
    return coordinates->get(0, numFrames * numAtoms * 2 + this->_index * numFrames + frame);
}

