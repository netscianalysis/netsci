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

#ifndef CURVA_SERIALIZER_H
#define CURVA_SERIALIZER_H

#include "json/include/nlohmann/json.hpp"
#include "atom.h"
#include "node.h"
#include "network.h"

namespace nlohmann {


    template<>
    struct adl_serializer<Atom *> {
#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

        static Atom *from_json(
                const json &j
        );

#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

        static void to_json(
                json &j,
                Atom *atom
        );
    };

    template<>
    struct adl_serializer<Atoms *> {
#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

        static Atoms *from_json(
                const json &j
        );

#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

        static void to_json(
                json &j,
                Atoms *atoms
        );
    };

    template<>
    struct adl_serializer<Node *> {


#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

        static void to_json(
                json &j,
                Node *node
        );

        static Node *from_json(
                const json &j
        );
    };
    template<>
    struct adl_serializer<Graph *> {


#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

        static void to_json(
                json &j,
                Graph *graph
        );

        static Graph *from_json(
                const json &j
        );
    };

}
#endif //CURVA_SERIALIZER_H

