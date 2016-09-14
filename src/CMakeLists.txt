# tfs_dnn library

set(SOURCE_FILES
        BinaryStream.cpp
        CsvFile.cpp
        Dnn.cpp
        DnnBuilder.cpp
        DnnLayer.cpp
        DnnLayerConvolution.cpp
        DnnLayerDeconvolution.cpp
        DnnLayerDropout.cpp
        DnnLayerFullyConnected.cpp
        DnnLayerInput.cpp
        DnnLayerLocalResponseNormalization.cpp
        DnnLayerMaxout.cpp
        DnnLayerPool.cpp
        DnnLayerRectifiedLinearUnit.cpp
        DnnLayerRegression.cpp
        DnnLayerSigmoid.cpp
        DnnLayerSoftmax.cpp
        DnnLayerSupportVectorMachine.cpp
        DnnLayerTanh.cpp
        DnnStream.cpp
        DnnTrainer.cpp
        DnnTrainerAdaDelta.cpp
        DnnTrainerAdam.cpp
        DnnTrainerNesterov.cpp
        DnnTrainerSGD.cpp
        Error.cpp
        Matrix.cpp
        Utility.cpp)

set_source_files_properties( ${SOURCE_FILES} PROPERTIES LANGUAGE CXX )

add_library(tfs_dnn ${SOURCE_FILES})

#target_include_directories (tfs_dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})