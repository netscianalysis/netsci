//
// Created by astokely on 5/3/23.
//

#ifndef NETSCI_UTILS_H
#define NETSCI_UTILS_H


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

#ifndef CURVA_UTILS_H
#define CURVA_UTILS_H
#define HASH_A 54059 /* a prime */
#define HASH_B 76963 /* another prime */
#define HASH_FIRSTH 37 /* also prime */


#include <string>

namespace utils {
    /*!
    @function{determineLastFrame} @type{void}
    @brief Determine the last frame to be read from the DCD file. If lastFrame is -1, then
    the last frame read is the last frame in the file.

    @param lastFrame The last frame to be read from the DCD file.
    @type{int*}

    @param numFrames The total number of frames in the DCD file.
    @type{int}
    */
    void determineLastFrame (
            int *lastFrame,
            int numFrames
    );

    /*!
    @function{generateIndicesArray} @type{void}
    @brief Generate an array of indices from 0 to size-1.

    @param indicesArray The array the indices will be stored in.
    @type{unsigned int**}

    @param size The size of the array.
    @type{int}
    */
    void generateIndicesArray (
            unsigned int **indicesArray,
            int size
    );

    /*!
    @function{isRecordAtom} @type{bool}
    @brief Determine if a line from a PDB file is an ATOM record.

    @param pdbLine The line from the PDB file.
    @type{std::string&}

    @return True if the line is an ATOM record, false otherwise.
    */
    bool isRecordAtom (std::string &pdbLine);

    /*!
    @function{isEndOfFrame} @type{bool}
    @brief Determine if a line from a PDB file is the last line of a frame.

    @param pdbLine The line from the PDB file.
    @type{std::string&}

    @return True if the line is the last line of a frame, false otherwise.
    */
    bool isEndOfFrame (std::string &pdbLine);

    /*!
    @function{strToDouble} @type{double}
    @brief Convert a string to a double.

    @param str The string to be converted.
    @type{const std::string&}

    @return The double value of the string.
    */
    double strToDouble (const std::string &str);

    /*!
    @function{strToInt} @type{int}
    @brief Convert a string to an int.

    @param str The string to be converted.
    @type{const std::string&}

    @return The int value of the string.
    */
    int strToInt (const std::string &str);

    /*!
    @function{removeWhiteSpace} @type{std::string}
    @brief Remove all whitespace from a string.

    @param str The string to be modified.
    @type{std::string}

    @return The string with all whitespace removed.
    */
    std::string removeWhiteSpace (std::string str);


    /*!
    @function{hashString} @type{unsigned int}
    @brief Hash a string using the FNV-1a algorithm.

    @param str The string to be hashed.
    @type{const std::string&}

    @return The hash value of the string.
    */
    unsigned int hashString (const std::string &str);

    /*!
    @function{substringInString} @type{int}

    @brief Determine if a substring is in a string.

    @param str The string to be searched.
    @type{const std::string&}

    @param substr The substring to be searched for.
    @type{const std::string&}

    @return 1 if the substring is in the string, 0 otherwise.
    */
    int substringInString (
            const std::string &str,
            const std::string
            &substr
    );
}
#endif //CURVA_UTILS_H


#endif //NETSCI_UTILS_H
