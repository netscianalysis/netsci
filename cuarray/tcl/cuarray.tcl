proc loadFloatCuArray {fname} {
    set floatCuArray [FloatCuArray]
    $floatCuArray load $fname
    return $floatCuArray
}

proc saveFloatCuArray {fname cuArray} {
    $cuArray save $fname
}

proc argsortFloatCuArray {cuArray} {
    set argsortCuArray [$cuArray argsort]}
    return $argsortCuArray
}

proc sortFloatCuArray {cuArray} {
    set sortedCuArray [$cuArray sort]}
    return $sortedCuArray
}

proc floatCuArrayToList {cuArray} {
    set a [list]
    for {set i 0} {$i < [$cuArray m]} {incr i} {
     set b [list]
     for {set j 0} {$j < [$cuArray n]} {incr j} {
            lappend b [$cuArray get $i $j]
     }
        lappend a $b
    }
    return $a
}

proc loadIntCuArray {fname} {
    set intCuArray [IntCuArray]
    $intCuArray load $fname
    return $intCuArray
}

proc saveIntCuArray {fname cuArray} {
    $cuArray save $fname
}

proc intCuArrayToList {cuArray} {
    set a [list]
    for {set i 0} {$i < [$cuArray m]} {incr i} {
     set b [list]
     for {set j 0} {$j < [$cuArray n]} {incr j} {
            lappend b [$cuArray get $i $j]
     }
        lappend a $b
    }
    return $a
}

proc argsortIntCuArray {cuArray} {
    set argsortCuArray [$cuArray argsort]}
    return $argsortCuArray
}

proc sortIntCuArray {cuArray} {
    set sortedCuArray [$cuArray sort]}
    return $sortedCuArray
}