# add_subdirectory(states)

################################################
# The low level states class
################################################
set(SOURCES
    main.cpp
)
pybind11_add_module(_low_level ${SOURCES})
target_include_directories(_low_level PRIVATE ${LIB_LOW_LEVEL_INCLUDE_DIR})
target_link_libraries(_low_level PRIVATE liblowlevel)
