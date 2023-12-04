if {[string first cuarray.so [exec tail -n 1 "$env(VMD_TCL_INDEX_PATH)"]] == -1} {
    exec echo load "$env(TCL_NETVIS_LIB_PATH)" >> "$env(VMD_TCL_INDEX_PATH)"
    exec echo load "$env(TCL_CUARRAY_LIB_PATH)" >> "$env(VMD_TCL_INDEX_PATH)"
}

