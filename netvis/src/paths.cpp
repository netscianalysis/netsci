#include <iostream>
#include "paths.h"

Paths::Paths() = default;

Paths::~Paths() {
    for (auto &path: this->paths_.paths) {
        delete &path;
    }
}

void Paths::init(
        Network *network,
        int molId
) {
    this->network_ = network;
    this->molId_ = molId;
}

std::string Paths::addPath(
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
) {
    std::string tclCommand;
    if (std::find(
            this->activePaths_.begin(),
            this->activePaths_.end(),
            pathIndex
    ) != this->activePaths_.end()) {
        tclCommand = this->paths_.paths[pathIndex]->hidePath();
    }
    if (std::find(
            this->activeNodes_.begin(),
            this->activeNodes_.end(),
            pathIndex
    ) != this->activeNodes_.end()) {
        tclCommand += this->paths_.paths[pathIndex]->hideNodes();
    }
    auto path = new Path();
    path->init(
            this->network_,
            pathFileName,
            this->molId_,
            nodeStyle,
            nodeMaterial,
            nodeColor,
            pathStyle,
            pathMaterial,
            pathColor,
            radius,
            length
    );
    this->paths_.paths[pathIndex] = path;
    *TCL_COMMAND = tclCommand;
    return tclCommand;
}

std::string Paths::showPath(
        int pathIndex,
        std::string *TCL_COMMAND
) {
    std::string tclCommand;
    if (pathIndex == -1) {
        for (auto path: this->paths_.paths) {
            tclCommand += this->showPath(path.first,
                                         TCL_COMMAND);
        }
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }
    if (this->paths_.paths.find(pathIndex) != this->paths_.paths.end()) {
        if (std::find(
                this->activePaths_.begin(),
                this->activePaths_.end(),
                pathIndex
        ) != this->activePaths_.end()) {
            tclCommand = this->paths_.paths[pathIndex]->showPath();
            *TCL_COMMAND = tclCommand;
            return tclCommand;
        }
        this->activePaths_.push_back(pathIndex);
        tclCommand = this->paths_.paths[pathIndex]->showPath();
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }
    tclCommand = ";";
    *TCL_COMMAND = tclCommand;
    return tclCommand;
}

std::string Paths::showNodes(
        int pathIndex,
        std::string *TCL_COMMAND
) {
    std::string tclCommand;
    if (pathIndex == -1) {
        for (auto path: this->paths_.paths) {
            tclCommand += this->showNodes(path.first,
                                          TCL_COMMAND);
        }
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }
    if (this->paths_.paths.find(pathIndex) != this->paths_.paths.end()) {
        if (std::find(
                this->activeNodes_.begin(),
                this->activeNodes_.end(),
                pathIndex
        ) != this->activeNodes_.end()) {
            tclCommand = this->paths_.paths[pathIndex]->showNodes();
            *TCL_COMMAND = tclCommand;
            return tclCommand;
        }
        this->activeNodes_.push_back(pathIndex);
        tclCommand = this->paths_.paths[pathIndex]->showNodes();
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }
    tclCommand = ";";
    *TCL_COMMAND = tclCommand;
    return tclCommand;
}

std::string Paths::hidePath(
        int pathIndex,
        std::string *TCL_COMMAND
) {
    std::string tclCommand;
    if (pathIndex == -1) {
        for (auto path: this->paths_.paths) {
            tclCommand += this->hidePath(path.first,
                                         TCL_COMMAND);
        }
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }
    for (auto &pathIndex_: this->activePaths_) {
        tclCommand += this->paths_.paths[pathIndex_]->hidePath();
    }
    auto pathIndexIterator = std::find(
            this->activePaths_.begin(),
            this->activePaths_.end(),
            pathIndex
    );
    if (pathIndexIterator != this->activePaths_.end()) {
        pathIndex = (int) (pathIndexIterator -
                           this->activePaths_.begin());
        this->activePaths_.erase(
                this->activePaths_.begin() + pathIndex
        );
    }
    for (auto pathIndex_: this->activePaths_) {
        tclCommand += this->paths_.paths[pathIndex_]->showPath();
    }
    *TCL_COMMAND = tclCommand;
    return tclCommand;
}

std::string Paths::hideNodes(
        int pathIndex,
        std::string *TCL_COMMAND
) {
    std::string tclCommand;
    if (pathIndex == -1) {
        for (auto path: this->paths_.paths) {
            tclCommand += this->hideNodes(path.first,
                                          TCL_COMMAND);
        }
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }

    for (auto &pathIndex_: this->activeNodes_) {
        tclCommand += this->paths_.paths[pathIndex_]->hideNodes();
    }
    auto pathIndexIterator = std::find(
            this->activeNodes_.begin(),
            this->activeNodes_.end(),
            pathIndex
    );
    if (pathIndexIterator != this->activeNodes_.end()) {
        pathIndex = (int) (pathIndexIterator -
                           this->activeNodes_.begin());
        this->activeNodes_.erase(
                this->activeNodes_.begin() + pathIndex
        );
    }
    for (auto pathIndex_: this->activeNodes_) {
        tclCommand += this->paths_.paths[pathIndex_]->showNodes();
    }
    *TCL_COMMAND = tclCommand;
    return tclCommand;
}

std::string Paths::removePath(
        int pathIndex,
        std::string *TCL_COMMAND
) {
    std::string tclCommand;
    if (pathIndex == -1) {
        std::vector<int> pathIndices;
        pathIndices.reserve(this->paths_.paths.size());
        for (auto path: this->paths_.paths) {
            pathIndices.push_back(path.first);
        }
        for (auto pathIndex_: pathIndices) {
            tclCommand += this->removePath(pathIndex_,
                                           TCL_COMMAND);
        }
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }
    if (this->paths_.paths.find(
            pathIndex
    ) != this->paths_.paths.end()) {
        auto pathIndexIterator = std::find(
                this->activePaths_.begin(),
                this->activePaths_.end(),
                pathIndex
        );
        if (pathIndexIterator != this->activePaths_.end()) {
            int activePathIndex = (int) (pathIndexIterator -
                                         this->activePaths_.begin());
            this->activePaths_.erase(
                    this->activePaths_.begin() + activePathIndex
            );
            tclCommand += this->paths_.paths[pathIndex]->hidePath();

        }
        pathIndexIterator = std::find(
                this->activeNodes_.begin(),
                this->activeNodes_.end(),
                pathIndex
        );
        if (pathIndexIterator != this->activeNodes_.end()) {
            int activeNodeIndex = (int) (pathIndexIterator -
                                         this->activeNodes_.begin());
            this->activeNodes_.erase(
                    this->activeNodes_.begin() + activeNodeIndex
            );
            tclCommand += this->paths_.paths[pathIndex]->hideNodes();
        }
        delete this->paths_.paths[pathIndex];
        this->paths_.paths.erase(pathIndex);
        *TCL_COMMAND = tclCommand;
        return tclCommand;
    }
    tclCommand = ";";
    *TCL_COMMAND = tclCommand;
    return tclCommand;
}

std::string Paths::updatePath(
        const std::string &nodeStyle,
        const std::string &nodeMaterial,
        const std::string &nodeColor,
        const std::string &pathStyle,
        const std::string &pathMaterial,
        const std::string &pathColor,
        float radius,
        int pathIndex,
        std::string *TCL_COMMAND
) {
    std::string tclCommand;
    if (this->paths_.paths.find(pathIndex) != this->paths_.paths.end()) {
        std::string pathFileName = this->paths_.paths[pathIndex]->pathFileName();
        float length = this->paths_.paths[pathIndex]->length();
        int showPath = 0;
        int showNodes = 0;
        if (std::find(
                this->activePaths_.begin(),
                this->activePaths_.end(),
                pathIndex
        ) != this->activePaths_.end()) {
            showPath = 1;
        }
        if (std::find(
                this->activeNodes_.begin(),
                this->activeNodes_.end(),
                pathIndex
        ) != this->activeNodes_.end()) {
            showNodes = 1;
        }
        tclCommand += this->removePath(pathIndex,
                                       TCL_COMMAND);
        tclCommand += this->addPath(
                pathFileName,
                nodeStyle,
                nodeMaterial,
                nodeColor,
                pathStyle,
                pathMaterial,
                pathColor,
                radius,
                length,
                pathIndex,
                TCL_COMMAND
        );
        if (showPath) {
            tclCommand += this->showPath(pathIndex,
                                         TCL_COMMAND);
        }
        if (showNodes) {
            tclCommand += this->showNodes(pathIndex,
                                          TCL_COMMAND);
        }
        *TCL_COMMAND += tclCommand;
        return tclCommand;
    }
    tclCommand = ";";
    *TCL_COMMAND = tclCommand;
    return tclCommand;
}

int Paths::numPaths() const {
    return (int) this->paths_.paths.size();
}

Path *Paths::path(int pathIndex) {
    return this->paths_.paths[pathIndex];
}

PathIndexPathMap Paths::paths() const {
    return this->paths_;
}




