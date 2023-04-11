
set(CLEAN_FILES_AND_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/*
        )
set_directory_properties(
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${CLEAN_FILES_AND_DIRS}"
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