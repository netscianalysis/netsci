//
// Created by astokely on 5/2/23.
//

#ifndef NETSCI_GRAPH_H
#define NETSCI_GRAPH_H

#include <map>
#include "node.h"


class Graph {
public:
    Graph();

    ~Graph();

    void init(
            const std::string& trajectoryFile,
            const std::string& topologyFile,
            int firstFrame,
            int lastFrame
    );

    int numNodes() const;

    CuArray<float> *nodeCoordinates();

    std::vector<Node *> &nodes();

    int numFrames() const;

    Node *nodeFromAtomIndex(int atomIndex);

    Atoms *atoms() const;

    void parsePdb(const std::string &fname);

    void parseDcd(const std::string &nodeCoordinates, int firstFrame, int lastFrame);

private:
    std::vector<Node*> nodeAtomIndexVector_;
    std::vector<Node *> nodes_;
    int numNodes_;
    int numFrames_;
    CuArray<float> *nodeCoordinates_;
    Atoms *atoms_;
};

#endif //NETSCI_GRAPH_H
