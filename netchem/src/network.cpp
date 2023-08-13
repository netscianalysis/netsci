//
// Created by astokely on 5/2/23.
//
#include <boost/range/irange.hpp>
#include <iostream>
#include <fstream>
#include "network.h"
#include "utils.h"
#include "dcd/dcd.h"
#include "serializer.h"

Network::Network() {
    this->numNodes_ = 0;
    this->numFrames_ = 0;
    this->nodeCoordinates_ = new CuArray<float>();
    this->atoms_ = new Atoms;
};

Network::~Network() {
    for (auto node: this->nodes_) {
        delete node;
    }
    delete nodeCoordinates_;
    this->nodes_.clear();
    this->nodeAtomIndexVector_.clear();
    delete atoms_;
}

void Network::init(
        const std::string &trajectoryFile,
        const std::string &topologyFile,
        int firstFrame,
        int lastFrame,
        int stride
) {
    this->numFrames_ = ((
                                lastFrame
                                - firstFrame
                        )
                        + 1) / stride;
    this->atoms_ = new Atoms();
    this->parsePdb(topologyFile);
    std::map<std::string, Node *> nodeTagNodeMap;
    this->nodeAtomIndexVector_.resize(this->atoms_->numAtoms());
    this->nodes_.resize(this->atoms_->numUniqueTags());
    for (auto atom: this->atoms_->atoms()) {
        auto emplace_pair = nodeTagNodeMap.emplace(
                atom->tag(),
                new Node(this->numFrames_,
                         nodeTagNodeMap.size())
        );
        emplace_pair.first->second->_tag = atom->tag();
        emplace_pair.first->second->atoms_.push_back(atom);
        emplace_pair.first->second->_numAtoms =
                emplace_pair.first->second->atoms_.size();
        emplace_pair.first->second->_totalMass += atom->mass();
        this->nodes_[emplace_pair.first->second->index()] = emplace_pair.first->second;
        this->nodeAtomIndexVector_[atom->index()] = emplace_pair.first->second;
    }
    this->numNodes_ = nodeTagNodeMap.size();
    this->nodeCoordinates_ = new CuArray<float>();
    this->nodeCoordinates_->init(
            this->atoms_->numUniqueTags(),
            3 * this->numFrames_
    );
    parseDcd(
            trajectoryFile,
            firstFrame,
            lastFrame,
            stride);
}

int Network::numNodes() const {
    return this->numNodes_;
}

CuArray<float> *Network::nodeCoordinates() {
    return this->nodeCoordinates_;
}

std::vector<Node *> &Network::nodes() {
    return this->nodes_;
}

int Network::numFrames() const {
    return this->numFrames_;
}

Node *Network::nodeFromAtomIndex(int atomIndex) {
    return this->nodeAtomIndexVector_.at(atomIndex);
}

Atoms *Network::atoms() const {
    return this->atoms_;
}

void Network::parseDcd(
        const std::string &fname,
        int firstFrame,
        int lastFrame,
        int stride
) {
    int numAtoms;
    int totalNumFrames = 0;
    dcdhandle *dcd = open_dcd_read(
            &fname[0],
            &numAtoms,
            &totalNumFrames
    );
    utils::determineLastFrame(
            &lastFrame,
            totalNumFrames
    );
    molfile_timestep_t ts;
    unsigned int *atomIndicesArray = nullptr;
    utils::generateIndicesArray(
            &atomIndicesArray,
            numAtoms
    );
    auto getAtomCoordinatesFromDcdLambda = [](
            int numAtoms,
            const float *x,
            const float *y,
            const float *z,
            int atomIndex,
            int frame,
            int numFrames,
            const std::vector<Node *> &nodeAtomIndexVector,
            Atoms *atoms,
            CuArray<float> *nodeCoordinates
    ) {
        int nodeIndex = nodeAtomIndexVector.at(atomIndex)->index();
        float nodeX = nodeCoordinates->get(
                nodeIndex,
                frame
        ) + x[atomIndex] * atoms->at(atomIndex)->mass();
        float nodeY = nodeCoordinates->get(
                nodeIndex,
                frame + numFrames
        ) + y[atomIndex] * atoms->at(atomIndex)->mass();
        float nodeZ = nodeCoordinates->get(
                nodeIndex,
                frame + 2 * numFrames
        ) + z[atomIndex] * atoms->at(atomIndex)->mass();
        nodeCoordinates->set(
                nodeX,
                nodeIndex,
                frame
        );
        nodeCoordinates->set(
                nodeY,
                nodeIndex,
                frame + numFrames
        );
        nodeCoordinates->set(nodeZ,
                             nodeIndex,
                             frame + 2 * numFrames);
        if (frame == numFrames - 1 &&
            nodeAtomIndexVector.at(atomIndex)->atoms_.back()->index() ==
            atomIndex) {
            for (auto frameIndex: boost::irange(numFrames)) {
                nodeX = nodeCoordinates->get(
                        nodeIndex,
                        frameIndex
                ) / nodeAtomIndexVector.at(atomIndex)->_totalMass;
                nodeY = nodeCoordinates->get(
                        nodeIndex,
                        frameIndex + numFrames
                ) / nodeAtomIndexVector.at(atomIndex)->_totalMass;
                nodeZ = nodeCoordinates->get(
                        nodeIndex,
                        frameIndex + 2 * numFrames
                ) / nodeAtomIndexVector.at(atomIndex)->_totalMass;

                nodeCoordinates->set(
                        nodeX,
                        nodeIndex,
                        frameIndex
                );
                nodeCoordinates->set(
                        nodeY,
                        nodeIndex,
                        frameIndex + numFrames
                );
                nodeCoordinates->set(
                        nodeZ,
                        nodeIndex,
                        frameIndex + 2 * numFrames
                );
            }
        }
    };
    while (dcd->setsread
           <= lastFrame) {
        read_next_timestep(
                dcd,
                numAtoms,
                &ts
        );
        if (dcd->setsread
            > firstFrame) {
            std::for_each(
                    atomIndicesArray,
                    atomIndicesArray
                    + numAtoms,
                    [
                            getAtomCoordinatesFromDcdLambda,
                            numAtoms,
                            capture0 = dcd->x,
                            capture1 = dcd->y,
                            capture2 = dcd->z,
                            capture3 = (
                                    (dcd->setsread
                                     - 1
                                     - firstFrame) / stride
                            ),
                            capture4 = this->numFrames_,
                            capture5 = this->nodeAtomIndexVector_,
                            capture6 = this->atoms_,
                            capture7 = this->nodeCoordinates_
                    ](
                            auto &&PH1
                    ) {
                        return getAtomCoordinatesFromDcdLambda(
                                numAtoms,
                                capture0,
                                capture1,
                                capture2,
                                std::forward<decltype(PH1)>(PH1),
                                capture3,
                                capture4,
                                capture5,
                                capture6,
                                capture7
                        );
                    }
            );
            if (stride > 1) {
                for (int s = 0; s < stride - 1; s++) {
                    read_next_timestep(
                            dcd,
                            numAtoms,
                            &ts
                    );
                }
            }
        }
    }
    close_file_read(dcd);
    delete[] atomIndicesArray;
}

void Network::parsePdb(
        const std::string &fname
) {
    std::string line;
    std::ifstream pdb(fname);
    int atomIndex = 0;
    while (std::getline(
            pdb,
            line
    )) {
        if (utils::isRecordAtom(line)) {
            auto atom = new Atom(
                    line,
                    atomIndex
            );
            this->atoms_->addAtom(
                    atom
            );
            atomIndex++;
        }
    }
}

void Network::save(const std::string &jsonFile) {
    nlohmann::json j;
    j["numNodes_"] = this->numNodes_;
    j["numFrames_"] = this->numFrames_;
    j["nodes_"] = this->nodes_;
    j["nodeAtomIndexVector_"] = this->nodeAtomIndexVector_;
    j["atoms_"] = this->atoms_;
    std::ofstream o(jsonFile);
    o << std::setw(4) << j << std::endl;
}

void Network::load(const std::string &jsonFile) {
    nlohmann::json j;
    std::ifstream i(jsonFile);
    i >> j;
    i.close();
    this->numFrames_ = j.at("numFrames_").get<int>();
    this->numNodes_ = j.at("numNodes_").get<int>();
    this->nodes_ = j.at("nodes_").get<std::vector<Node *>>();
    this->nodeAtomIndexVector_ = j.at("nodeAtomIndexVector_").get<std::vector<Node *>>();
    this->atoms_ = j.at("atoms_").get<Atoms *>();
}

void Network::nodeCoordinates(
        const std::string &nodeCoordinatesFile
) {
    delete this->nodeCoordinates_;
    this->nodeCoordinates_ = new CuArray<float>(
    );
    this->nodeCoordinates_->load(
            nodeCoordinatesFile
    );
}
