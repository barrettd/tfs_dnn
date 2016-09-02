cmake_minimum_required(VERSION 2.8)
project(tfs_dnn)

# set( CMAKE_VERBOSE_MAKEFILE 1 )

set( DEFAULT_C_LANG CXX )

set(CMAKE_BUILD_TYPE Release)

add_subdirectory (src)
add_subdirectory (test)

