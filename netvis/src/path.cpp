#include "path.h"
#include <sstream>

Path::Path() {
    this->path_ = new CuArray<int>();
}

Path::~Path() {
    delete this->path_;
}

void Path::init(
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
) {
    this->network_ = network;
    this->path_->load(pathFileName);
    this->nodeStyle_ = nodeStyle;
    this->nodeMaterial_ = nodeMaterial;
    this->nodeColor_ = nodeColor;
    this->pathStyle_ = pathStyle;
    this->pathMaterial_ = pathMaterial;
    this->pathColor_ = pathColor;
    this->molId_ = molId;
    this->radius_ = radius;
    this->pathFileName_ = pathFileName;
    this->length_ = length;
    this->residueIds_.intVector.resize(this->path_->n());
    this->atomIndices_.intVectorVector.resize(this->path_->n());
    this->nodeIndices_.intVector.resize(this->path_->n());
    this->residueNames_.stringVector.resize(this->path_->n());
    for (int i = 0; i < this->path_->n(); i++) {
        this->residueIds_[i] = this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()[0]->residueId();
        for (auto atom: this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()) {
            this->atomIndices_[i].push_back(atom->index());
        }
        this->nodeIndices_[i] = this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->index();
        this->residueNames_[i] = this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()[0]->residueName();
    }
}

std::string Path::showNodes() {
    this->vmdSelection_.clear();
    std::stringstream s3;
    std::stringstream s4;
    s3
            << "index ";
    for (int i = 0; i < this->path_->n() - 1; i++) {
        int atom1Index = this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()[0]->index();
        for (auto atom: this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()) {
            if (atom->name() == "CA") {
                atom1Index = atom->index();
            }
        }
        int atom2Index = this->network_->nodes()[
                this->path_->get(0,
                                 i + 1)
        ]->atoms()[0]->index();
        for (auto atom: this->network_->nodes()[
                this->path_->get(0,
                                 i + 1)
        ]->atoms()) {
            if (atom->name() == "CA") {
                atom2Index = atom->index();
            }
        }
        s3
                << atom1Index
                << " "
                << atom2Index
                << " ";
    }
    std::string s3String = s3.str().substr(0,
                                           s3.str().size() - 1);
    s4
            << "set repid [molinfo "
            << this->molId_
            << " get numreps];";
    s4
            << "mol addrep "
            << this->molId_
            << ";";
    s4
            << "mol modselect $repid "
            << this->molId_
            << " \"same residue as "
            << s3String
            << "\";";
    s4
            << "mol modstyle $repid "
            << this->molId_
            << " "
            << this->nodeStyle_
            << " ;";
    s4
            << "mol modmaterial $repid "
            << this->molId_
            << " "
            << this->nodeMaterial_
            << ";";
    s4
            << "mol modcolor $repid "
            << this->molId_
            << " "
            << this->nodeColor_
            << ";";
    std::string s4String = s4.str();
    std::string tclCommand = s4String;
    this->vmdSelection_ = s3String;
    return tclCommand;
}

std::string Path::showPath() {
    this->vmdSelection_.clear();
    std::stringstream s1;
    std::stringstream s2;
    std::stringstream s3;
    s1
            << "set repid [molinfo "
            << this->molId_
            << " get numreps];";
    s1
            << "mol addrep "
            << this->molId_
            << ";";
    s1
            << "mol modstyle $repid "
            << this->molId_
            << " "
            << this->pathStyle_
            << " "
            << this->radius_
            << " 1.2 100.0"
            << ";";
    s1
            << "mol modmaterial $repid "
            << this->molId_
            << " "
            << this->pathMaterial_
            << ";";
    s1
            << "mol modcolor $repid "
            << this->molId_
            << " \""
            << this->pathColor_
            << "\";";
    s2
            << "mol modselect $repid "
            << this->molId_
            << " \"";

    s3
            << "index ";
    for (int i = 0; i < this->path_->n() - 1; i++) {
        int atom1Index = this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()[0]->index();
        for (auto atom: this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()) {
            if (atom->name() == "CA") {
                atom1Index = atom->index();
            }
        }
        int atom2Index = this->network_->nodes()[
                this->path_->get(0,
                                 i + 1)
        ]->atoms()[0]->index();
        for (auto atom: this->network_->nodes()[
                this->path_->get(0,
                                 i + 1)
        ]->atoms()) {
            if (atom->name() == "CA") {
                atom2Index = atom->index();
            }
        }
        s3
                << atom1Index
                << " "
                << atom2Index
                << " ";
        s1
                << "topo addbond "
                << atom1Index
                << " "
                << atom2Index
                << " -molid "
                << this->molId_
                << ";";
    }
    std::string s1String = s1.str();
    std::string s2String = s2.str();
    std::string s3String = s3.str().substr(0,
                                           s3.str().size() - 1);
    std::string tclCommand =
            s1String + s2String + s3String + "\";";
    this->vmdSelection_ = s3String;
    return tclCommand;
}

std::string Path::hideNodes() {
    std::stringstream tclCommand;
    tclCommand
            << "set repid [molinfo "
            << this->molId_
            << " get numreps];"
            << "for {set i [expr $repid-1]} {$i >= 0} {incr i -1} {"
            << "lassign [molinfo "
            << this->molId_
            << " get \"{selection $i}\"] selection;"
            << "if {$selection eq \"same residue as "
            << this->vmdSelection_
            << "\"} {mol delrep $i "
            << this->molId_
            << ";}"
            << "};";
    return tclCommand.str();
}

std::string Path::hidePath() {
    std::stringstream tclCommand;
    tclCommand
            << "set repid [molinfo "
            << this->molId_
            << " get numreps];"
            << "for {set i [expr $repid-1]} {$i >= 0} {incr i -1} {"
            << "lassign [molinfo "
            << this->molId_
            << " get \"{selection $i}\"] selection;"
            << "if {$selection eq \""
            << this->vmdSelection_
            << "\"} {mol delrep $i "
            << this->molId_
            << ";}"
            << "};";
    for (int i = 0; i < this->path_->n() - 1; i++) {
        int atom1Index = this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()[0]->index();
        for (auto atom: this->network_->nodes()[
                this->path_->get(0,
                                 i)
        ]->atoms()) {
            if (atom->name() == "CA") {
                atom1Index = atom->index();
            }
        }
        int atom2Index = this->network_->nodes()[
                this->path_->get(0,
                                 i + 1)
        ]->atoms()[0]->index();
        for (auto atom: this->network_->nodes()[
                this->path_->get(0,
                                 i + 1)
        ]->atoms()) {
            if (atom->name() == "CA") {
                atom2Index = atom->index();
            }
        }
        tclCommand
                << "topo delbond "
                << atom1Index
                << " "
                << atom2Index
                << " -molid "
                << this->molId_
                << ";";
    }
    return tclCommand.str();
}

std::string Path::pathFileName() const {
    return this->pathFileName_;
}

int Path::molId() const {
    return this->molId_;
}

std::string Path::nodeStyle() const {
    return this->nodeStyle_;
}

std::string Path::nodeMaterial() const {
    return this->nodeMaterial_;
}

std::string Path::nodeColor() const {
    return this->nodeColor_;
}

std::string Path::pathStyle() const {
    return this->pathStyle_;
}

std::string Path::pathMaterial() const {
    return this->pathMaterial_;
}

std::string Path::pathColor() const {
    return this->pathColor_;
}

float Path::radius() const {
    return this->radius_;
}

float Path::length() const {
    return this->length_;
}

IntVector Path::residueIds() const {
    return this->residueIds_;
}

IntVectorVector Path::atomIndices() const {
    return this->atomIndices_;
}

IntVector Path::nodeIndices() const {
    return this->nodeIndices_;
}

StringVector Path::residueNames() const {
    return this->residueNames_;
}
