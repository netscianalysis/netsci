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

#include "atoms.h"

Atoms::Atoms () = default;

void Atoms::addAtom(Atom *atom) {
    this->uniqueTags_.insert(atom->tag());
    atoms_.push_back(atom);
}

int Atoms::numAtoms () const {
	return static_cast<int>(atoms_.size());
}

Atom *Atoms::at(int atomIndex) {
	return atoms_.at(atomIndex);
}

int Atoms::numUniqueTags () const {
    return static_cast<int>(uniqueTags_.size());
}

std::vector<Atom*> &Atoms::atoms() {
    return this->atoms_;
}

