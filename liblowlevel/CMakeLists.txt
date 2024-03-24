add_library(liblowlevel STATIC
  src/states/state.cpp
  src/states/state_utils.cpp
)
target_include_directories(liblowlevel PUBLIC 
    ${LIB_LOW_LEVEL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)