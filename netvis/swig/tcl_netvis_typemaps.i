%{
#include <string>
#include <map>
#include <iostream>
%}

%typemap(in, numinputs=0) std::string *TCL_COMMAND (std::string temp) {
    $1 = &temp;
}

%typemap(argout) std::string *TCL_COMMAND {
        int result = Tcl_Eval(interp, $1->c_str());
}


%typemap(out) PathIndexPathMap {
    Tcl_Obj *dict = Tcl_NewDictObj();
    for (auto const &pair : $1.paths) {
        Tcl_Obj *key = Tcl_NewIntObj(pair.first);
        Tcl_Obj * obj = SWIG_Tcl_NewInstanceObj(interp, SWIG_as_voidptr(pair.second), SWIGTYPE_p_Path, 0);
        Tcl_DictObjPut(interp, dict, key, obj);
    }
    Tcl_SetObjResult(interp, dict);
}

%typemap(out) IntVector {
    Tcl_Obj *list = Tcl_NewListObj(0, NULL);
    for (auto const &item : $1.intVector) {
        Tcl_ListObjAppendElement(interp, list, Tcl_NewIntObj(item));
    }
    Tcl_SetObjResult(interp, list);
}

%typemap(out) IntVectorVector {
    Tcl_Obj *list = Tcl_NewListObj(0, NULL);
    for (auto const &item : $1.intVectorVector) {
        Tcl_Obj *sublist = Tcl_NewListObj(0, NULL);
        for (auto const &subitem : item) {
            Tcl_ListObjAppendElement(interp, sublist, Tcl_NewIntObj(subitem));
        }
        Tcl_ListObjAppendElement(interp, list, sublist);
    }
    Tcl_SetObjResult(interp, list);
}

%typemap(out) StringVector {
    Tcl_Obj *list = Tcl_NewListObj(0, NULL);
    for (auto const &item : $1.stringVector) {
        Tcl_ListObjAppendElement(interp, list, Tcl_NewStringObj(item.c_str(), -1));
    }
    Tcl_SetObjResult(interp, list);
}
