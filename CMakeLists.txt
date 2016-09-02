cmake_minimum_required(VERSION 2.8)
project(tfs_dnn)

set( DEFAULT_C_LANG CXX )

if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++11")        ## Optimize for speed
set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()

add_subdirectory (src)
add_subdirectory (test)

