add_library(liblowlevel STATIC
    src/density_matrix_state.cpp
)
target_include_directories(liblowlevel PUBLIC 
    ${LIB_LOW_LEVEL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)