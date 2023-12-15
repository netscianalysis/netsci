%template(FloatFloatVector) std::vector<std::vector<float> >;

%extend CuArray {
        std::vector<std::vector<float> > toList() {
            std::vector<std::vector<float> > cuArrayData($self->m(), std::vector<float>($self->n()));
            for (int i = 0; i < $self->m(); i++) {
                for (int j = 0; j < $self->n(); j++) {
                    cuArrayData[i][j] = $self->get(i, j);
                }
            }
            return cuArrayData;
        }
};

%typemap(out) std::vector<std::vector<float> > {
        for (int i = 0; i < $1.size(); i++) {
            Tcl_Obj *list = Tcl_NewListObj(0, NULL);
            for (int j = 0; j < $1[i].size(); j++) {
                double tmp = (double) $1[i][j];
                Tcl_Obj *obj = Tcl_NewDoubleObj(tmp);
                Tcl_ListObjAppendElement(interp, list, obj);
            }
            Tcl_ListObjAppendElement(interp, $result, list);
        }
    }