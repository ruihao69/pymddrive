cmake_minimum_required(VERSION 3.15...3.26)
project(low_level LANGUAGES CXX)

# Compiler options: C++17, warnings, and optimization
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)              # Generic maximum code optimization
add_compile_options(-ftree-vectorize) # SIMD vectorize the loops
add_compile_options(-funroll-loops)   # explicit optimize the loops
add_compile_options(-Wall -Wextra -Wpedantic) # Enable warning
# if on Linux, add the following options
if(UNIX AND NOT APPLE)
    add_compile_options(-fPIC)
endif()
# add_compile_options(-DNDEBUG)          # Disable assert
# add_compile_options(-march=native)    # Enable the native architecture
# apply this: -DEIGEN_DONT_VECTORIZE
# add_compile_definitions(EIGEN_DONT_VECTORIZE)


# use Eigen3 in the thirdparty folder
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/thirdparty/eigen/)
include_directories(${EIGEN3_INCLUDE_DIR})

# add the include for pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/pybind11)

# add the low_level library, compile the low_level library
set(LIB_LOW_LEVEL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/liblowlevel/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/liblowlevel)

# create python bindings for the low_level library
add_subdirectory(${CMAKE_SOURCE_DIR}/pymddrive/low_level)
