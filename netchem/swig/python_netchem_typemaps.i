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
class StopGraphIterator{};
class GraphIterator{
        public:
        GraphIterator(
                std::vector<Node*>::iterator _cur,
        int numNodes_
        ) : cur(_cur), numNodes(numNodes_) {}
        GraphIterator* __iter__()
        {
            return this;
        }
        std::vector<Node*>::iterator cur;
        int numNodes;
        int stopAtNextIteration = 0;
};
%}
%include "exception.i"

%exception GraphIterator::__next__{
        try {
            $action // calls %extend function next() below
        }
        catch (StopGraphIterator) {
            PyErr_SetString(PyExc_StopIteration, "End of iterator");
            return NULL;
        }
}
%extend GraphIterator{
        Node * __next__() {
            if (!$self->stopAtNextIteration) {
                if ((*($self->cur))->index() == $self->numNodes - 1) {
                    $self->stopAtNextIteration = 1;
                }
                return *($self->cur++);
            }
            throw StopGraphIterator();
        }
}

%extend Graph{
        GraphIterator __iter__() {
            return GraphIterator($self->nodes().begin(), $self->nodes().size());
        }
};
