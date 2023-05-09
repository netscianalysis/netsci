//
// Created by astokely on 5/3/23.
//
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

#include "utils.h"
#include <numeric>
#include  <algorithm>

void utils::determineLastFrame (
        int *lastFrame,
        int numFrames
) {
    if (*lastFrame
        == -1) {
        *lastFrame = numFrames
                     - 1;
    }
}

void utils::generateIndicesArray (
        unsigned int **indicesArray,
        int size
) {
    *indicesArray = new unsigned int[size];
    std::iota(
            *indicesArray,
            *indicesArray
            + size,
            0
    );
}

bool utils::isRecordAtom (std::string &pdbLine) {
    size_t isRecordAtom = pdbLine.find("ATOM");
    if (isRecordAtom
        != std::string::npos) {
        return true;
    } else {
        return false;
    }
}

bool utils::isEndOfFrame (std::string &pdbLine) {
    size_t isEndOfFrame = pdbLine.find("END");
    if (isEndOfFrame
        != std::string::npos) {
        return true;
    } else {
        return false;
    }
}

std::string utils::removeWhiteSpace (std::string str) {
    str.erase(
            std::remove_if(
                    str.begin(),
                    str.end(),
                    isspace
            ),
            str.end()
    );
    return str;
}

double utils::strToDouble (const std::string &str) {
    return std::stod(str);
}

int utils::strToInt (const std::string &str) {
    return std::stoi(str);
}

unsigned int utils::hashString (
        const std::string &str
) {
    const char   *strPtr = str.data();
    unsigned int h       = HASH_FIRSTH;
    while (*strPtr) {
        h = (
                    h
                    * HASH_A
            )
            ^ (
                    strPtr[0]
                    * HASH_B
            );
        strPtr++;
    }
    return h;
}

int utils::substringInString (
        const std::string &str,
        const std::string
        &substr
) {
    if (str.find(substr)
        != std::string::npos) {
        return 1;
    }
    return 0;
}