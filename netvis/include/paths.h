//
// Created by astokely on 12/6/23.
//

#ifndef NETSCI_PATHS_H
#define NETSCI_PATHS_H

#include "path.h"

struct PathIndexPathMap {
    std::map<int, Path *> paths;
};


class Paths {
public:
    Paths();

    ~Paths();

    void init(
            Network *network,
            int molId
    );

    std::string addPath(
            const std::string &pathFileName,
            const std::string &nodeStyle,
            const std::string &nodeMaterial,
            const std::string &nodeColor,
            const std::string &pathStyle,
            const std::string &pathMaterial,
            const std::string &pathColor,
            float radius,
            float length,
            int pathIndex,
            std::string *TCL_COMMAND
    );

    std::string removePath(
            int pathIndex,
            std::string *TCL_COMMAND
    );

    std::string updatePath(
            const std::string &nodeStyle,
            const std::string &nodeMaterial,
            const std::string &nodeColor,
            const std::string &pathStyle,
            const std::string &pathMaterial,
            const std::string &pathColor,
            float radius,
            int pathIndex,
            std::string *TCL_COMMAND
    );


    std::string showPath(
            int pathIndex,
            std::string *TCL_COMMAND
    );

    std::string showNodes(
            int pathIndex,
            std::string *TCL_COMMAND
    );

    std::string hidePath(
            int pathIndex,
            std::string *TCL_COMMAND
    );

    std::string hideNodes(
            int pathIndex,
            std::string *TCL_COMMAND
    );

    int numPaths() const;

    Path *path(int pathIndex);

    PathIndexPathMap paths() const;

private:
    int molId_;
    Network *network_;
    PathIndexPathMap pathIndexPathMap_;
    std::vector<int> activePaths_;
    std::vector<int> activeNodes_;
    PathIndexPathMap paths_;
};

#endif //NETSCI_PATHS_H
