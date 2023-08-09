%{
#include <algorithm>
#include <sstream>
%}

%inline %{
class StopAtomsIterator{};
class AtomsIterator{
        public:
        AtomsIterator(
                std::vector<Atom*>::iterator _cur,
                int numAtoms_
        ) : cur(_cur), numAtoms(numAtoms_) {}
        AtomsIterator* __iter__()
        {
            return this;
        }
        std::vector<Atom*>::iterator cur;
        int numAtoms;
        int stopAtNextIteration = 0;
};
%}
%include "exception.i"

%exception AtomsIterator::__next__{
        try {
            $action // calls %extend function next() below
        }
        catch (StopAtomsIterator) {
            PyErr_SetString(PyExc_StopIteration, "End of iterator");
            return NULL;
        }
}
%extend AtomsIterator{
        Atom * __next__() {
            if (!$self->stopAtNextIteration) {
                if ((*($self->cur))->index() == $self->numAtoms - 1) {
                    $self->stopAtNextIteration = 1;
                }
                return *($self->cur++);
            }
            throw StopAtomsIterator();
        }
}

%extend Atoms{
        AtomsIterator __iter__() {
            return AtomsIterator($self->atoms().begin(), $self->numAtoms());
        }
};

%inline %{
class StopNetworkIterator{};
class NetworkIterator{
        public:
        NetworkIterator(
                std::vector<Node*>::iterator _cur,
        int numNodes_
        ) : cur(_cur), numNodes(numNodes_) {}
        NetworkIterator* __iter__()
        {
            return this;
        }
        std::vector<Node*>::iterator cur;
        int numNodes;
        int stopAtNextIteration = 0;
};
%}
%include "exception.i"

%exception NetworkIterator::__next__{
        try {
            $action // calls %extend function next() below
        }
        catch (StopNetworkIterator) {
            PyErr_SetString(PyExc_StopIteration, "End of iterator");
            return NULL;
        }
}
%extend NetworkIterator{
        Node * __next__() {
            if (!$self->stopAtNextIteration) {
                if ((*($self->cur))->index() == $self->numNodes - 1) {
                    $self->stopAtNextIteration = 1;
                }
                return *($self->cur++);
            }
            throw StopNetworkIterator();
        }
}

%extend Network{
        NetworkIterator __iter__() {
            return NetworkIterator($self->nodes().begin(), $self->nodes().size());
        }
};

%extend Atoms{
    Atom *__getitem__(int atomIndex) {
        return $self->atoms().at(atomIndex);
    }
};

%extend Atom{
    std::string __repr__() {
        std::stringstream repr_ss;
        repr_ss << "Index: " << $self->index() << std::endl;
        repr_ss << "Serial: " << $self->serial() << std::endl;
        repr_ss << "Name: " << $self->name() << std::endl;
        repr_ss << "Element: " << $self->element() << std::endl;
        repr_ss << "Residue ID: " << $self->residueId() << std::endl;
        repr_ss << "Residue Name: " << $self->residueName() <<
        std::endl;
        repr_ss << "Chain ID: " << $self->chainId() << std::endl;
        repr_ss << "Segment ID: " << $self->segmentId() << std::endl;
        repr_ss << "Mass: " << $self->mass();
        return repr_ss.str();
    }
};

%extend Node{
        std::string __repr__() {
            std::stringstream repr_ss;
            repr_ss << "Index: " << $self->index() << std::endl;
            repr_ss << "Tag: " << $self->tag() << std::endl;
            repr_ss << "Total Mass: " << $self->totalMass() <<
            std::endl;
            repr_ss << "Number of Atoms: " << $self->numAtoms();
            return repr_ss.str();
        }
};
