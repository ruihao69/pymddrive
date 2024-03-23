cmake_minimum_required(VERSION 3.15)
project(liblowlevel)

# Compiler options: C++17, warnings, and optimization
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O2)              # Generic maximum code optimization
add_compile_options(-ftree-vectorize) # SIMD vectorize the loops
add_compile_options(-funroll-loops)   # explicit optimize the loops
add_compile_options(-Wall -Wextra -Wpedantic) # Enable warning
# add archetecture specific optimization
# add_compile_options(-march=native)

# add the include for pybind11
add_subdirectory(thirdparty/pybind11)
    
# Add your source files here
set(SOURCES
    src/main.cpp
)

# Create the pybind11 module
pybind11_add_module(my_module ${SOURCES})
