//
// Created by astokely on 5/4/23.
//

#ifndef NETSCI_ATOM_H
#define NETSCI_ATOM_H

#include <string>
#include "cuarray.h"

class Atom {
public:
    Atom ();

    explicit Atom (
            const std::string &pdbLine
    );

    Atom (
            const std::string &pdbLine,
            int atomIndex
    );

    int index () const;

    std::string name ();

    std::string element ();

    std::string residueName ();

    int residueId () const;

    std::string chainId ();

    std::string segmentId ();

    float temperatureFactor () const;

    float occupancy () const;

    int serial () const;

    std::string tag ();

    float mass () const {
        return _mass;
    }

    unsigned int hash () const;

    float x (
            CuArray<float> *coordinates,
            int frame,
            int numFrames
    ) const;

    float y (
            CuArray<float> *coordinates,
            int frame,
            int numFrames
    ) const;

    float z (
            CuArray<float> *coordinates,
            int frame,
            int numFrames
    ) const;

private:
    int _index;

    std::string _name;

    std::string _element;

    std::string _residueName;

    int _residueId;

    std::string _chainId;

    std::string _segmentId;

    float _temperatureFactor;

    float _occupancy;

    int _serial;

    std::string _tag;

    float _mass;

    unsigned int _hash;
};

#endif //NETSCI_ATOM_H
