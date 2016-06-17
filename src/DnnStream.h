// ----------------------------------------------------------------------------
//  DnnStream.hpp
//  Binary file stream for the Neural Net and intermediate files.
//
//  Created by Barrett Davis on 6/15/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// ----------------------------------------------------------------------------
#include "BinaryStream.h"
#include "Dnn.h"

// Forward declarations:
class DnnLayerInput;

class DnnLayerConvolution;
class DnnLayerDropout;
class DnnLayerFullyConnected;
class DnnLayerLocalResponseNormalization;
class DnnLayerMaxout;
class DnnLayerPool;
class DnnLayerRectifiedLinearUnit;
class DnnLayerRegression;
class DnnLayerSigmoid;
class DnnLayerSoftmax;
class DnnLayerSupportVectorMachine;
class DnnLayerTanh;


#ifndef DnnStream_h
#define DnnStream_h

namespace tfs {         // Tree Frog Software
    
    class OutDnnStream : public OutBinaryStream {
    protected:
        bool writeHeader( void );
        bool writeLayerBase( DnnLayer                       *layer );
        bool writeLayer( DnnLayerInput                      *layer );    // input
        bool writeLayer( DnnLayerConvolution                *layer );    // conv
        bool writeLayer( DnnLayerDropout                    *layer );    // dropout
        bool writeLayer( DnnLayerFullyConnected             *layer );    // fc
        bool writeLayer( DnnLayerLocalResponseNormalization *layer );    // lrn
        bool writeLayer( DnnLayerMaxout                     *layer );    // maxout
        bool writeLayer( DnnLayerPool                       *layer );    // pool
        bool writeLayer( DnnLayerRectifiedLinearUnit        *layer );    // relu
        bool writeLayer( DnnLayerRegression                 *layer );    // regression
        bool writeLayer( DnnLayerSigmoid                    *layer );    // sigmoid
        bool writeLayer( DnnLayerSoftmax                    *layer );    // softmax
        bool writeLayer( DnnLayerSupportVectorMachine       *layer );    // svm
        bool writeLayer( DnnLayerTanh                       *layer );    // tanh

    public:
        OutDnnStream( const char *path );
        
        bool writeDnn( Dnn &dnn );
        
    };
    
    class InDnnStream : public InBinaryStream {
    protected:
        bool readHeader( unsigned short &contentVersion );
        bool readEnum( int &value, int maxValue );

        bool readLayerBase( DnnLayer *layer );
        bool readLayerInput(                      Dnn &dnn );    // input
        bool readLayerConvolution(                Dnn &dnn );    // conv
        bool readLayerDropout(                    Dnn &dnn );    // dropout
        bool readLayerFullyConnected(             Dnn &dnn );    // fc
        bool readLayerLocalResponseNormalization( Dnn &dnn );    // lrn
        bool readLayerMaxout(                     Dnn &dnn );    // maxout
        bool readLayerPool(                       Dnn &dnn );    // pool
        bool readLayerRectifiedLinearUnit(        Dnn &dnn );    // relu
        bool readLayerRegression(                 Dnn &dnn );    // regression
        bool readLayerSigmoid(                    Dnn &dnn );    // sigmoid
        bool readLayerSoftmax(                    Dnn &dnn );    // softmax
        bool readLayerSupportVectorMachine(       Dnn &dnn );    // svm
        bool readLayerTanh(                       Dnn &dnn );    // tanh
        
    public:
        InDnnStream( const char *path );

        Dnn *readDnn( bool trainable = false );

    };

}   // namespace tfs

#endif /* DnnStream_h */
