cmake_minimum_required(VERSION 2.8)
project(tfs_dnn)

# set( CMAKE_VERBOSE_MAKEFILE 1 )

set( DEFAULT_C_LANG CXX )

if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS “-Wall”)                            ## Report all warnings
set(CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -O3”)           ## Optimize for speed
set(CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -std=c++11”)    ## Conform to C++ 11
set(CMAKE_EXE_LINKER_FLAGS "-s")                        ## Strip symbols from binary
endif()

add_subdirectory (src)
add_subdirectory (test)

