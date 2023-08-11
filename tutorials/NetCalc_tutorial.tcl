load ../cuarray/tcl/cuarray.so
#Load the CuArray Tcl library


proc cuarray_access_example {} {
    set pyro_R [FloatCuArray]
    $pyro_R load "pyro_R.npy"
    #Create a new FloatCuArray object and load the numpy results file into it.

    set m [$pyro_R m]
    set n [$pyro_R n]
    #Set m equal to the number of columns and n equal to the number of rows in pyro_R.

    for {set i 0} {$i < $m} {incr i} {
        for {set j 0} {$j < $m} {incr j} {
            puts [$pyro_R get $i $j]
        }
    }
    #Iterate through pyro_R and print each element.
}

proc argsort_example {} {
    set pyro_R [FloatCuArray]
    $pyro_R load "pyro_R.npy"
    #Create a new FloatCuArray object and load the numpy results file into it.

    set m [$pyro_R m]
    set n [$pyro_R n]
    #Set m equal to the number of columns and n equal to the number of rows in pyro_R.

    for {set i 0} {$i < $m} {incr i} {
        set i_sorted_idxs [$pyro_R argsort $i]
        #Create an IntCuArray of the indices that descending sort the ith row of pyro_R.
        for {set j 0} {$j < $m} {incr j} {
            set sorted_idx [$i_sorted_idxs get 0 $j]
            #Set sorted_idx equal to the jth element in i_sorted_idxs
            set rij [$pyro_R get $i $sorted_idx]
            puts "$rij"
        }
    }
}


proc sort_example {} {
    set pyro_R [FloatCuArray]
    $pyro_R load "pyro_R.npy"
    #Create a new FloatCuArray object and load the numpy results file into it.

    set m [$pyro_R m]
    set n [$pyro_R n]
    #Set m equal to the number of columns and n equal to the number of rows in pyro_R.

    for {set i 0} {$i < $m} {incr i} {
        #Descending sort the ith row of pyro_R
        set sorted_row [$pyro_R sort $i]
        for {set j 0} {$j < $m} {incr j} {
            set rij [$sorted_row get 0 $j]
            puts "$rij"
        }
    }
}


sort_example

