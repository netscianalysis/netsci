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

#include "serializer.h"
#include "cnpy.h"
#include "node.h"

Atom *
nlohmann::adl_serializer
        <Atom *>::from_json(
        const json &j
) {
    Atom atom{};
    atom._index = j.at("_index").get<int>();
    atom._name = j.at("_name").get<std::string>();
    atom._element = j.at("_element").get<std::string>();
    atom._residueName = j.at("_residueName").get<std::string>();
    atom._residueId = j.at("_residueId").get<int>();
    atom._chainId = j.at("_chainId").get<std::string>();
    atom._segmentId = j.at("_segmentId").get<std::string>();
    atom._temperatureFactor = j.at("_temperatureFactor").get<double>();
    atom._occupancy = j.at("_occupancy").get<double>();
    atom._serial = j.at("_serial").get<int>();
    atom._tag = j.at("_tag").get<std::string>();
    atom._mass = j.at("_mass").get<double>();
    atom._hash = j.at("_hash").get<unsigned int>();
    return new Atom(atom);
}

void nlohmann::adl_serializer
        <Atom *>::to_json(
        json &j,
        Atom *atom
) {
    j = json{
            {"_index",             atom->_index},
            {"_name",              atom->_name},
            {"_element",           atom->_element},
            {"_residueName",       atom->_residueName},
            {"_residueId",         atom->_residueId},
            {"_chainId",           atom->_chainId},
            {"_segmentId",         atom->_segmentId},
            {"_temperatureFactor", atom->_temperatureFactor},
            {"_occupancy",         atom->_occupancy},
            {"_serial",            atom->_serial},
            {"_tag",               atom->_tag},
            {"_mass",              atom->_mass},
            {"_hash",              atom->_hash}
    };
}

void nlohmann::adl_serializer<Atoms *>::to_json(
        json &j,
        Atoms *atoms
        ) {
    j = json{
            {"uniqueTags_", atoms->uniqueTags_},
            {"_atoms",      atoms->atoms_}
    };

}

Atoms * nlohmann::adl_serializer<Atoms *>::from_json(
        const json &j
        ) {
    Atoms atoms;
    auto uniqueTags = j.at("uniqueTags_").get<std::vector<std::string>>();
    for (auto &tag : uniqueTags) {
        atoms.uniqueTags_.insert(tag);
    }
    atoms.atoms_ = j.at("_atoms").get<std::vector<Atom *>>();
    return new Atoms(atoms);
}

void nlohmann::adl_serializer
        <Node *>::to_json(
        json &j,
        Node *node
) {
    j = {
            {"_numAtoms",    node->_numAtoms},
            {"_index",       node->_index},
            {"_totalMass",   node->_totalMass},
            {"_tag",         node->_tag},
            {"_hash",        node->_hash},
            {"atomIndices_", node->atomIndices_},
            {"atoms_",        node->atoms_},
    };
}

Node *
nlohmann::adl_serializer
        <Node *>::from_json(
        const json &j
) {
    Node node;
    node._numAtoms = j.at("_numAtoms").get<int>();
    node._index = j.at("_index").get<int>();
    node._totalMass = j.at("_totalMass").get<float>();
    node._tag = j.at("_tag").get<std::string>();
    node._hash = j.at("_hash").get<unsigned int>();
    node.atoms_ = j.at("atoms_").get<std::vector<Atom *>>();
    node.atomIndices_ = j.at("atomIndices_").get<std::vector<int>>();
    return new Node(node);
}

void nlohmann::adl_serializer
        <Graph *>::to_json(
        json &j,
        Graph *graph
) {
    /*
     *  std::vector<Node*> nodeAtomIndexVector_;
    std::vector<Node *> nodes_;
    int numNodes_;
    int numFrames_;
    CuArray<float> *nodeCoordinates_;
    Atoms *atoms_;
     */
    j = {
            {"numNodes_",            graph->numNodes_},
            {"numFrames_",           graph->numFrames_},
            {"nodes_",               graph->nodes_},
            {"nodeAtomIndexVector_", graph->nodeAtomIndexVector_},
    };
}

Graph *
nlohmann::adl_serializer
        <Graph *>::from_json(
        const json &j
) {
    Graph graph;
    graph.numFrames_ = j.at("numFrames_").get<int>();
    graph.numNodes_ = j.at("numNodes_").get<int>();
    graph.nodes_ = j.at("nodes_").get<std::vector<Node *>>();
    graph.nodeAtomIndexVector_ = j.at("nodeAtomIndexVector_").get<std::vector<Node *>>();
    return new Graph(graph);
}



