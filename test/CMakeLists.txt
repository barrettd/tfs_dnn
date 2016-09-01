# test app for tfs_dnn


set(SOURCE_FILES
        main.cpp
        test2D.cpp
        test2D.hpp
        test2Layer.cpp
        test2Layer.hpp
        testBuilder.cpp
        testBuilder.hpp
        testCifar10.cpp
        testCifar10.hpp
        testFullyConnected.cpp
        testFullyConnected.hpp
        testIO.cpp
        testIO.hpp
        TestMatrix.cpp
        TestMatrix.hpp)

add_executable(test_tfs_dnn ${SOURCE_FILES})

target_link_libraries (test_tfs_dnn LINK_PUBLIC tfs_dnn)