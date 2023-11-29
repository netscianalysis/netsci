%feature("docstring")
CuArray
"\nDescription\n"
"--------------------\n"
"    Destructor for CuArray.\n"
"    *\n"
"    Deallocates the memory on both the host and the device."
"\n"
;%feature("docstring")
CuArray::init
"\n"
"********************\n"
"*    Version 1     *\n"
"********************\n"

"init(m)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Initialize the CuArray with the specified dimensions.\n"
"    *\n"
"    Initializes the CuArray with the specified dimensions, allocating memory on both the host and the device.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"m\n"
"    The number of rows."
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
"\nExample\n"
"--------------------\n"
"    "

"\n"
"********************\n"
"*    Version 2     *\n"
"********************\n"

"init(host, n)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Initialize the CuArray with the specified host data and dimensions.\n"
"    *\n"
"    Initializes the CuArray with the specified host data and dimensions, allocating memory on both the host and the device.\n"
"    The data is shallow copied, meaning the ownership is not transferred.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"host\n"
"    Pointer to the input host data."
"\n"
"\n"
"n\n"
"    The number of columns."
"\n"
%feature("docstring")
CuArray::fromNumpy
"\n"
"********************\n"
"*    Version 1     *\n"
"********************\n"

"fromNumpy(NUMPY_ARRAY, NUMPY_ARRAY_DIM2)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Copy data from a NumPy array to the CuArray.\n"
"    *\n"
"    Copies data from the specified NumPy array to the CuArray.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"NUMPY_ARRAY\n"
"    Pointer to the input NumPy array."
"\n"
"\n"
"NUMPY_ARRAY_DIM2\n"
"    Pointer to the dimension 2 of the NumPy array."
"\n"
"\nExample\n"
"--------------------\n"
"    "

"\n"
"********************\n"
"*    Version 2     *\n"
"********************\n"

"fromNumpy(NUMPY_ARRAY)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Copy data from a NumPy array to the CuArray.\n"
"    *\n"
"    Copies data from the specified NumPy array to the CuArray.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"NUMPY_ARRAY\n"
"    Pointer to the input NumPy array."
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
"\nExample\n"
"--------------------\n"
"    "
%feature("docstring")
CuArray::toNumpy
"\n"
"********************\n"
"*    Version 1     *\n"
"********************\n"

"toNumpy(NUMPY_ARRAY, NUMPY_ARRAY_DIM2)->None\n"
"\nDescription\n"
"--------------------\n"
"    Copy data from the CuArray to a NumPy array.\n"
"    *\n"
"    Copies data from the CuArray to the specified NumPy array.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"NUMPY_ARRAY\n"
"    Pointer to the output NumPy array."
"\n"
"\n"
"NUMPY_ARRAY_DIM2\n"
"    Pointer to the dimension 2 of the NumPy array.\n"
"    *"
"\n"
"\nExample\n"
"--------------------\n"
"    "

"\n"
"********************\n"
"*    Version 2     *\n"
"********************\n"

"toNumpy(NUMPY_ARRAY)->None\n"
"\nDescription\n"
"--------------------\n"
"    Copy data from the CuArray to a NumPy array.\n"
"    *\n"
"    Copies data from the CuArray to the specified NumPy array.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"NUMPY_ARRAY\n"
"    Pointer to the output NumPy array."
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::CuArray
"CuArray()->constructor\n"
"\nDescription\n"
"--------------------\n"
"    Default constructor for CuArray.Constructs an empty\n"
"    CuArray object.\n"
"    *"
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::fromCuArrayShallowCopy
"fromCuArrayShallowCopy(cuArray, end, n)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Shallow copy data from another CuArray.\n"
"    *\n"
"    Shallow copies the host data from the provided CuArray. All data\n"
"    in the range of rows, specified by the 'start' and 'end'\n"
"    parameters, is copied. The range is inclusive. This CuArray\n"
"    does not own the data, so the data can only be deallocated by\n"
"    deleting the source CuArray.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"cuArray\n"
"    Pointer to the source CuArray."
"\n"
"\n"
"end\n"
"    The index of the last row to copy."
"\n"
"\n"
"n\n"
"    The number of columns in this CuArray."
"\n"
;%feature("docstring")
CuArray::fromCuArrayDeepCopy
"fromCuArrayDeepCopy(cuArray, end, n)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Deep copy data from another CuArray.\n"
"    *\n"
"    Deep copies the host data from the provided CuArray. All data in\n"
"    the range of rows, specified by the 'start' and 'end'\n"
"    parameters, is copied.  The range is inclusive.\n"
"    Memory is allocated on the host CuArray.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"cuArray\n"
"    Pointer to the source CuArray."
"\n"
"\n"
"end\n"
"    The index of the last row to copy."
"\n"
"\n"
"n\n"
"    The number of columns in this CuArray."
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::n
"n()->int\n"
"\nDescription\n"
"--------------------\n"
"    Get the number of columns in the CuArray.\n"
"    *\n"
"    Returns the number of columns in the CuArray.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The number of columns.\n"
"    *"
"\n"
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::m
"m()->int\n"
"\nDescription\n"
"--------------------\n"
"    Get the number of rows in the CuArray.\n"
"    *\n"
"    Returns the number of rows in the CuArray.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The number of rows.\n"
"    *"
"\n"
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::size
"size()->int\n"
"\nDescription\n"
"--------------------\n"
"    Get the total number of elements in the CuArray.\n"
"    *\n"
"    Returns the total number of elements in the CuArray, which is equal to the number of rows multiplied by the number of columns.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The total number of elements.\n"
"    *"
"\n"
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::bytes
"bytes()->size_t\n"
"\nDescription\n"
"--------------------\n"
"    Get the total size in bytes of the CuArray data.\n"
"    *\n"
"    Returns the total size in bytes of the CuArray data, including both the host and device memory.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The size in bytes.\n"
"    *"
"\n"
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::host
"host()->T *&\n"
"\nDescription\n"
"--------------------\n"
"    Get a reference to the host data.\n"
"    *\n"
"    Returns a reference to the host data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    A reference to the host data.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::device
"device()->T\n"
"\nDescription\n"
"--------------------\n"
"    Get a reference to the device data.\n"
"    *\n"
"    Returns a reference to the device data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    A reference to the device data.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::allocateHost
"allocateHost()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Allocate memory for the host data.\n"
"    *\n"
"    Allocates memory for the host data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::allocateDevice
"allocateDevice()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Allocate memory for the device data.\n"
"    *\n"
"    Allocates memory for the device data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::allocatedHost
"allocatedHost()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Check if memory is allocated for the host data.\n"
"    *\n"
"    Checks if memory is allocated for the host data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::allocatedDevice
"allocatedDevice()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Check if memory is allocated for the device data.\n"
"    *\n"
"    Checks if memory is allocated for the device data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::toDevice
"toDevice()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Copy data from the host to the device.\n"
"    *\n"
"    Copies the data from the host to the device.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::toHost
"toHost()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Copy data from the device to the host.\n"
"    *\n"
"    Copies the data from the device to the host.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::deallocateHost
"deallocateHost()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Deallocate memory for the host data.\n"
"    *\n"
"    Deallocates the memory for the host data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::deallocateDevice
"deallocateDevice()->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Deallocate memory for the device data.\n"
"    *\n"
"    Deallocates the memory for the device data.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The CuArrayError indicating the success or failure of the operation.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::get
"get(i)->T\n"
"\nDescription\n"
"--------------------\n"
"    Get the value at the specified position in the CuArray.\n"
"    *\n"
"    Returns the value at the specified position (i, j) in the CuArray.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"i\n"
"    The row index."
"\n"
"\nReturns\n"
"--------------------\n"
"    The value at the specified position.\n"
"    *"
"\n"
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::set
"set(value, j)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Set the value at the specified position in the CuArray.\n"
"    *\n"
"    Sets the value at the specified position (i, j) in the CuArray to the given value.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"value\n"
"    The value to set."
"\n"
"\n"
"j\n"
"    The column index."
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::load
"load(fname)->CuArrayError\n"
"\nDescription\n"
"--------------------\n"
"    Load the CuArray from a file.\n"
"    *\n"
"    Loads the CuArray data from the specified file.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"fname\n"
"    The name of the file to load."
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::save
"save(fname)->None\n"
"\nDescription\n"
"--------------------\n"
"    Save the CuArray to a file.\n"
"    *\n"
"    Saves the CuArray data to the specified file.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"fname\n"
"    The name of the file to save.\n"
"    *"
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::sort
"sort(i)->CuArray<T>\n"
"\nDescription\n"
"--------------------\n"
"    Sort the CuArray based on the specified row.\n"
"    *\n"
"    Sorts the CuArray in descending order based on the values in the\n"
"    specified row.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"i\n"
"    The index of the row to sort."
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::operator[]
"operator[](i)->T &\n"
"\nDescription\n"
"--------------------\n"
"    Get a reference to the element at the specified index in the CuArray.\n"
"    *\n"
"    Returns a reference to the element at the specified index in the CuArray.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"i\n"
"    The index of the element."
"\n"
"\nExample\n"
"--------------------\n"
"    "
;%feature("docstring")
CuArray::owner
"owner()->int\n"
"\nDescription\n"
"--------------------\n"
"    Get the owner of the CuArray.\n"
"    *\n"
"    Returns the owner of the CuArray, which indicates whether the\n"
"    CuArray is responsible for memory deallocation.\n"
"    *"
"\n"
"\nReturns\n"
"--------------------\n"
"    The owner of the CuArray.\n"
"    *"
"\n"
"\n"
;%feature("docstring")
CuArray::argsort
"argsort(i)->CuArray\n"
"\nDescription\n"
"--------------------\n"
"    Perform an argsort on the specified row of the CuArray.\n"
"    *\n"
"    Performs an argsort on the specified row of the CuArray and\n"
"    returns a new CuArray that contains the sorted indices.\n"
"    *"
"\n"
"\nParameters\n"
"--------------------\n"
"i\n"
"    The column index to argsort."
"\n"
"\nExample\n"
"--------------------\n"
"    "
;