# test app for tfs_dnn


set(SOURCE_FILES
        main.cpp
        test2D.cpp
        test2Layer.cpp
        testBuilder.cpp
        testCifar10.cpp
        testFullyConnected.cpp
        testIO.cpp
        TestMatrix.cpp)

include_directories (../src)

set_source_files_properties( ${SOURCE_FILES} PROPERTIES LANGUAGE CXX )

add_executable(test_tfs_dnn ${SOURCE_FILES})

target_link_libraries (test_tfs_dnn LINK_PUBLIC tfs_dnn)