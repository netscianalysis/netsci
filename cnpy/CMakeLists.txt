
#When make clean is called, all files in the lib directory are deleted
set_directory_properties(
        PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/lib/*"
)

add_library(
        cnpy SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cnpy.cpp
)
set_target_properties(
        cnpy
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${CNPY_LINK_LIBRARIES}"
)