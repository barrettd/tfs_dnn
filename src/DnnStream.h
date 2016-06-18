// ----------------------------------------------------------------------------
//  DnnStream.hpp
//  Binary file stream for the Neural Net and intermediate files.
//
//  Created by Barrett Davis on 6/15/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// ----------------------------------------------------------------------------
#include "BinaryStream.h"
#include "Dnn.h"
#include "DnnLayer.h"

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
        bool writeArray( const unsigned long *array, const unsigned long count );
        bool writeMatrix( const Matrix *matrix );

        bool writeLayerBegin( LayerType layerType );
        bool writeLayerBase( const DnnLayer                       &layer );
        bool writeLayer( const DnnLayerInput                      &layer );    // input
        bool writeLayer( const DnnLayerConvolution                &layer );    // conv
        bool writeLayer( const DnnLayerDropout                    &layer );    // dropout
        bool writeLayer( const DnnLayerFullyConnected             &layer );    // fc
        bool writeLayer( const DnnLayerLocalResponseNormalization &layer );    // lrn
        bool writeLayer( const DnnLayerMaxout                     &layer );    // maxout
        bool writeLayer( const DnnLayerPool                       &layer );    // pool
        bool writeLayer( const DnnLayerRectifiedLinearUnit        &layer );    // relu
        bool writeLayer( const DnnLayerRegression                 &layer );    // regression
        bool writeLayer( const DnnLayerSigmoid                    &layer );    // sigmoid
        bool writeLayer( const DnnLayerSoftmax                    &layer );    // softmax
        bool writeLayer( const DnnLayerSupportVectorMachine       &layer );    // svm
        bool writeLayer( const DnnLayerTanh                       &layer );    // tanh

    public:
        OutDnnStream( const char *path );
        
        bool writeDnn( const Dnn &dnn );
        
    };
    
    class InDnnStream : public InBinaryStream {
    protected:
        bool readHeader( unsigned short &contentVersion );
        bool readTag( const char *expected );
        bool readEnum( int &value, int maxValue );
        bool expectEnum( int value );
        
        unsigned long *readArrayUnsignedLong( unsigned long expectedCount );
        Matrix *readMatrix( void );

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
