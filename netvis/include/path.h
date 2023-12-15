#ifndef NETSCI_NETVIS_H
#define NETSCI_NETVIS_H

#include <string>
#include "cuarray.h"
#include "network.h"

struct IntVector {
    std::vector<int> intVector;

    int operator[](int i) const { return intVector[i]; }

    int &operator[](int i) { return intVector[i]; }
};

struct IntVectorVector {
    std::vector<std::vector<int> > intVectorVector;

    std::vector<int>
    operator[](int i) const { return intVectorVector[i]; }

    std::vector<int> &operator[](int i) { return intVectorVector[i]; }
};

struct StringVector {
    std::vector<std::string> stringVector;

    std::string operator[](int i) const { return stringVector[i]; }

    std::string &operator[](int i) { return stringVector[i]; }
};

class Path {
public:
    Path();

    ~Path();

    void init(
            Network *network,
            const std::string &pathFileName,
            int molId,
            const std::string &nodeStyle,
            const std::string &nodeMaterial,
            const std::string &nodeColor,
            const std::string &pathStyle,
            const std::string &pathMaterial,
            const std::string &pathColor,
            float radius,
            float length
    );


    std::string showPath();

    std::string showNodes();

    std::string hidePath();

    std::string hideNodes();

    std::string pathFileName() const;

    int molId() const;

    std::string nodeStyle() const;

    std::string nodeMaterial() const;

    std::string nodeColor() const;

    std::string pathStyle() const;

    std::string pathMaterial() const;

    std::string pathColor() const;

    float radius() const;

    float length() const;

    IntVector residueIds() const;

    IntVectorVector atomIndices() const;

    IntVector nodeIndices() const;

    StringVector residueNames() const;


private:
    CuArray<int> *path_;
    std::string pathFileName_;
    int molId_;
    Network *network_;
    std::string vmdSelection_;
    std::string nodeStyle_;
    std::string nodeMaterial_;
    std::string nodeColor_;
    std::string pathStyle_;
    std::string pathMaterial_;
    std::string pathColor_;
    float radius_;
    float length_;
    IntVector residueIds_;
    IntVectorVector atomIndices_;
    IntVector nodeIndices_;
    StringVector residueNames_;
};

#endif //NETSCI_NETVIS_H
