cmake_minimum_required(VERSION 2.8)

set(SOURCE_FILES
    BinaryStream.cpp
    BinaryStream.h
    Constants.h
    CsvFile.cpp
    CsvFile.h
    Dnn.cpp
    Dnn.h
    DnnBuilder.cpp
    DnnBuilder.h
    DnnLayer.cpp
    DnnLayer.h
    DnnLayerConvolution.cpp
    DnnLayerConvolution.h
    DnnLayerDeconvolution.cpp
    DnnLayerDeconvolution.h
    DnnLayerDropout.cpp
    DnnLayerDropout.h
    DnnLayerFullyConnected.cpp
    DnnLayerFullyConnected.h
    DnnLayerInput.cpp
    DnnLayerInput.h
    DnnLayerLocalResponseNormalization.cpp
    DnnLayerLocalResponseNormalization.h
    DnnLayerMaxout.cpp
    DnnLayerMaxout.h
    DnnLayerPool.cpp
    DnnLayerPool.h
    DnnLayerRectifiedLinearUnit.cpp
    DnnLayerRectifiedLinearUnit.h
    DnnLayerRegression.cpp
    DnnLayerRegression.h
    DnnLayers.h
    DnnLayerSigmoid.cpp
    DnnLayerSigmoid.h
    DnnLayerSoftmax.cpp
    DnnLayerSoftmax.h
    DnnLayerSupportVectorMachine.cpp
    DnnLayerSupportVectorMachine.h
    DnnLayerTanh.cpp
    DnnLayerTanh.h
    DnnStream.cpp
    DnnStream.h
    DnnTrainer.cpp
    DnnTrainer.h
    DnnTrainerAdaDelta.cpp
    DnnTrainerAdaDelta.h
    DnnTrainerAdam.cpp
    DnnTrainerAdam.h
    DnnTrainerSGD.cpp
    DnnTrainerSGD.h
    Error.cpp
    Error.h
    Matrix.cpp
    Matrix.h
    Utility.cpp
    Utility.h)

add_library(tfs_dnn ${SOURCE_FILES})

target_include_directories (tfs_dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})