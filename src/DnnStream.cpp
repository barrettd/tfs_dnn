// ----------------------------------------------------------------------------
//  DnnStream.hpp
//  Binary file stream for the Neural Net and intermediate files.
//
//  Created by Barrett Davis on 6/15/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// ----------------------------------------------------------------------------
// Sizes on OSX and Linux (64 bit Intel)
// sizeof( bool )           = 1
// sizeof( unsigned short ) = 2
// sizeof( unsigned long )  = 8
// sizeof( double )         = 8
// ----------------------------------------------------------------------------
// Header is:
// START_TAG,       char buffer,    4 bytes - "tfs"
// Header version,  unsigned short, 2 bytes - 1
// Content tag,     char buffer,    4 bytes - "dnn", "net", "dat", etc.
// Content version, unsigned short, 2 bytes - 1
// Byte order test, unsigned long,  8 bytes - 0x0102030405060708
// Type sizes,      char buffer,    8 bytes - 1,2,4,8,8,4,8,16
// Padding,         char buffer,    4 bytes - 0,0,0,0
// ----------------------------------------------------------------------------
// Total bytes:                    32 bytes - aligned to 8 byte boundary
// ----------------------------------------------------------------------------
#include "DnnLayers.h"
#include "DnnStream.h"
#include "Error.h"

namespace tfs {         // Tree Frog Software
    
    enum ObjectType {
        OBJECT_DNN = 1,
        OBJECT_LAYER,
        OBJECT_MATRIX,
        OBJECT_ARRAY_UNSIGNED_LONG,

        OBJECT_COUNT        // Used for range checking.
    };
 
    static const int  NN_FILE_TAG_LENGTH  = 4;                          // "tfs", "dnn", "net", "dat", etc.
    static const char START_TAG[ NN_FILE_TAG_LENGTH ] = { "tfs" };
    static const char DNN_TAG[   NN_FILE_TAG_LENGTH ] = { "dnn" };
    static const char END_TAG[   NN_FILE_TAG_LENGTH ] = { "eof" };
    
    static const unsigned short HEADER_VERSION  = 1;                   // Version of the header, not the content.
    static const unsigned short CONTENT_VERSION = 1;                   // Version of the content.
    static const int            TYPE_COUNT      = 8;
    static const unsigned long  BYTE_ORDER_TEST = 0x0102030405060708;  // 8 bytes
    static const int            PADDING_LENGTH  = 4;
    
    void copy( unsigned long *dst, const unsigned long *src, const unsigned long count ) {
        if( dst != 0 && src != 0 && count > 0 ) {
            memcpy( dst, src, count * sizeof( unsigned long ));
        }
    }
    
    OutDnnStream::OutDnnStream( const char *path ) :
    OutBinaryStream( path ) {
    }
    
    InDnnStream::InDnnStream( const char *path ):
    InBinaryStream( path ) {
    }
    
    bool
    OutDnnStream::writeHeader( void ) {
        if( m_stream.bad() ||
           !write( START_TAG, NN_FILE_TAG_LENGTH ) ||           // (4) start tag: "tfs"
           !write( HEADER_VERSION ) ||                          // (2) header format version
           !write( DNN_TAG, NN_FILE_TAG_LENGTH ) ||             // (4) content tag: "dnn"
           !write( CONTENT_VERSION ) ||                         // (2) file content version
           !write( BYTE_ORDER_TEST )) {                         // (8) byte order test
            return false;
        }
        unsigned char buffer[ TYPE_COUNT ];
        memset( buffer, 0, sizeof( buffer ));                   // Sizes of our base types
        buffer[0] = (unsigned char) sizeof( bool );                 //  1
        buffer[1] = (unsigned char) sizeof( unsigned short );       //  2
        buffer[2] = (unsigned char) sizeof( unsigned int );         //  4
        buffer[3] = (unsigned char) sizeof( unsigned long  );       //  8
        buffer[4] = (unsigned char) sizeof( unsigned long long );   //  8
        buffer[5] = (unsigned char) sizeof( float );                //  4
        buffer[6] = (unsigned char) sizeof( double );               //  8
        buffer[7] = (unsigned char) sizeof( long double );          // 16
        if( !this->write( buffer, sizeof( buffer ))) {          // (8) bytes
            return false;
        }
        memset( buffer, 0, sizeof( buffer ));                   // Clear buffer (again)
        return this->write( buffer, PADDING_LENGTH );           // (4) bytes of padding (zeros)
    }
    
    bool
    InDnnStream::readHeader( unsigned short &contentVersion ) {
        contentVersion = 0;
        if( m_stream.bad()) {
            return false;
        }
        // Check for our start tag: "tfs"
        if( !readTag( START_TAG )) {
            return false;
        }
        // Header format version, different from the content version.
        unsigned short format_version = 0;
        if( !read( format_version ) || format_version != HEADER_VERSION ) {
            return false;
        }
        // Check for our content tag: "dnn"
        if( !readTag( DNN_TAG )) {
            return false;
        }
        if( !read( contentVersion )) {
            return false;
        }
        unsigned long byte_order_test = 0;
        if( !read( byte_order_test ) || byte_order_test != BYTE_ORDER_TEST ) {  // Check that the byte order is as expected
            return false;
        }
        unsigned char buffer[ TYPE_COUNT ];
        memset( buffer, 0, sizeof( buffer ));
        if( !read( buffer, sizeof( buffer ))) {
            return false;
        }
        if( buffer[0] != (unsigned char) sizeof( bool )               ||    //  1, Check the sizes of our base types.
            buffer[1] != (unsigned char) sizeof( unsigned short )     ||    //  2
            buffer[2] != (unsigned char) sizeof( unsigned int )       ||    //  4
            buffer[3] != (unsigned char) sizeof( unsigned long  )     ||    //  8
            buffer[4] != (unsigned char) sizeof( unsigned long long ) ||    //  8
            buffer[5] != (unsigned char) sizeof( float  )             ||    //  6
            buffer[6] != (unsigned char) sizeof( double )             ||    //  8
            buffer[7] != (unsigned char) sizeof( long double )) {           // 16
            return false;
        }
        if( !read( buffer, PADDING_LENGTH )) {  // Consume padding at end of header.
            return false;
        }
        return true;                            // Return the content version
    }
    
    bool
    InDnnStream::readTag( const char *expected ) {
        if( expected == 0 || *expected == 0 ) {
            return false;
        }
        char tag_buffer[ NN_FILE_TAG_LENGTH ];
        return read( tag_buffer, NN_FILE_TAG_LENGTH ) && strncmp( tag_buffer, expected, NN_FILE_TAG_LENGTH ) == 0;
    }

    bool
    InDnnStream::readEnum( int &value, int maxValue ) {
        if( !read( value )) {
            return false;
        }
        if( value < 0 || value >= maxValue ) {
            return false;
        }
        return true;
    }
    
    bool
    InDnnStream::expectEnum( int expectedValue ) {
        int value = -1;
        return read( value ) && value == expectedValue;
    }
    
    bool
    OutDnnStream::writeArray( const unsigned long *array, const unsigned long count ) {
        if( !write( OBJECT_ARRAY_UNSIGNED_LONG )) {
            return log_error( "Unable to write matrix" );
        }
        const bool arrayIsNull = array == 0;
        if( !write( arrayIsNull )) {
            return log_error( "Unable to write array null / not null boolean" );
        }
        if( arrayIsNull ) {
            return true;            // Our work here is done.
        }
        if( !write( count )) {
            return log_error( "Unable to write array count" );
        }
        return write( array, count );
    }

    unsigned long*
    InDnnStream::readArrayUnsignedLong( unsigned long expectedCount ) {
        if( !expectEnum( OBJECT_ARRAY_UNSIGNED_LONG )) {
            log_error( "Did not find array object marker." );
            return 0;
        }
        bool arrayIsNull;
        if( !read( arrayIsNull )) {
            log_error( "Unable to read array null / not null boolean" );
            return 0;
        }
        if( arrayIsNull ) {
            return 0;
        }
        unsigned long count;
        if( !read( count )) {
            log_error( "Unable to read array size" );
            return 0;
        }
        if( count != expectedCount ) {
            log_error( "Array sizes do not match: expected = %lu, file = %lu", expectedCount, count );
            return 0;
        }
        unsigned long *array = new unsigned long[ count ];
        if( !read( array, count )) {
            log_error( "Unable to read array contents %lu", count );
            delete[] array;
            return 0;
        }
        return array;
    }

    bool
    OutDnnStream::writeMatrix( const Matrix *matrix ) {
        if( !write( OBJECT_MATRIX )) {
            return log_error( "Unable to write matrix" );
        }
        const bool matrixIsNull = matrix == 0;
        if( !write( matrixIsNull )) {
            return log_error( "Unable to write matrix null / not null boolean" );
        }
        if( matrixIsNull ) {
            return true;            // Our work here is done.
        }
        unsigned long aa = matrix->aa();
        unsigned long bb = matrix->bb();
        unsigned long cc = matrix->cc();
        unsigned long dd = matrix->dd();
        if( !write( aa ) || !write( bb ) || !write( cc ) || !write( dd )) {
            return log_error( "Unable to write matrix dimensions" );
        }
        const DNN_NUMERIC    *data = matrix->dataReadOnly();
        const unsigned long length = matrix->length();
        return write((const char*) data, length );
    }

    Matrix*
    InDnnStream::readMatrix( void ) {
        if( !expectEnum( OBJECT_MATRIX )) {
            log_error( "Did not find matrix object marker." );
            return 0;
        }
        bool matrixIsNull;
        if( !read( matrixIsNull )) {
            log_error( "Unable to read matrix null / not null boolean" );
            return 0;
        }
        if( matrixIsNull ) {
            return 0;
        }
        unsigned long aa, bb, cc, dd;
        if( !read( aa ) || !read( bb ) || !read( cc ) || !read( dd )) {
            log_error( "Unable to read matrix dimensions" );
            return 0;
        }
        Matrix *matrix = new Matrix( aa, bb, cc, dd );
        DNN_NUMERIC *data = matrix->data();
        const unsigned long length = matrix->length();
        if( !read((char*) data, length )) {
            log_error( "Unable to read %lu bytes", length );
            delete matrix;
            return 0;
        }
        return matrix;
    }

    bool
    OutDnnStream::writeLayerBegin( LayerType layerType ) {
        if( !write( OBJECT_LAYER ) || !write( layerType )) {
            return log_error( "Unable to write basic layer data" );
        }
        return true;
    }

    bool
    OutDnnStream::writeLayerBase( const DnnLayer &layer ) {
        writeMatrix( layer.weightsReadOnly());     // Internal: Weights, to act on input activations from previous layer
        writeMatrix( layer.gradiantReadOnly());    // Internal: Gradiant, will be null when not training.
        writeMatrix( layer.biasReadOnly());        // Internal: Bias, to act on input activations from previous layer
        writeMatrix( layer.biasDwReadOnly());      // Internal: Bias derivative, will be null when not training.
        writeMatrix( layer.outAReadOnly());        // Output:   Activations, output of a neuron.
        writeMatrix( layer.outDwReadOnly());       // Output:   Weight derivative, will be null when not training.
        write( layer.l1DecayMultiplier());
        write( layer.l2DecayMultiplier());
        return good();
    }
    
    bool
    InDnnStream::readLayerBase( DnnLayer *layer ) {
        if( layer == 0 ) {
            return log_error( "Layer is null" );
        }
        Matrix *weights  = readMatrix();
        Matrix *gradiant = readMatrix();
        Matrix *bias     = readMatrix();
        Matrix *biasDw   = readMatrix();
        Matrix *outA     = readMatrix();
        Matrix *outDw    = readMatrix();
        
        Matrix *layerWeights = layer->weights();
        if( layerWeights != 0 && weights != 0 ) {
            layerWeights->copy( *weights );
        }
        Matrix *layerGradiant = layer->gradiant();
        if( layerGradiant != 0 && gradiant != 0 ) {
            layerGradiant->copy( *gradiant );
        }
        Matrix *layerBias = layer->bias();
        if( layerBias != 0 && bias != 0 ) {
            layerBias->copy( *bias );
        }
        Matrix *layerBiasDw = layer->biasDw();
        if( layerBiasDw != 0 && biasDw != 0 ) {
            layerBiasDw->copy( *biasDw );
        }
        Matrix *layerOutA = layer->outA();
        if( layerOutA != 0 && outA != 0 ) {
            layerOutA->copy( *outA );
        }
        Matrix *layerOutDw = layer->outDw();
        if( layerOutDw != 0 && outDw != 0 ) {
            layerOutDw->copy( *outDw );
        }
        DNN_NUMERIC l1DecayMultiplier;
        DNN_NUMERIC l2DecayMultiplier;
        read( l1DecayMultiplier );
        read( l2DecayMultiplier );
        layer->l1DecayMultiplier( l1DecayMultiplier );
        layer->l2DecayMultiplier( l2DecayMultiplier );
        
        delete weights;
        delete gradiant;
        delete bias;
        delete biasDw;
        delete outA;
        delete outDw;
        return good();
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerInput &layer ) {
        writeMatrix( layer.outAReadOnly());        // Output:   Activations, output of a neuron.
        writeMatrix( layer.outDwReadOnly());       // Output:   Weight derivative, will be null when not training.
        return good();
    }
    
    bool
    InDnnStream::readLayerInput( Dnn &dnn ) {
        Matrix *outA  = readMatrix();
        Matrix *outDw = readMatrix();
        if( outA == 0 ) {
            delete outDw;
            return log_error( "No activation layer found" );
        }
        if( !dnn.addLayerInput( outA->aa(), outA->bb(), outA->cc())) {
            delete outA;
            delete outDw;
            return log_error( "Unable to add layer" );
        }
        DnnLayerInput *layer = dnn.getLayerInput();
        
        Matrix *layerOutA = layer->outA();
        if( layerOutA != 0 && outA != 0 ) {
            layerOutA->copy( *outA );
        }
        Matrix *layerOutDw = layer->outDw();
        if( layerOutDw != 0 && outDw != 0 ) {
            layerOutDw->copy( *outDw );
        }
        delete outA;
        delete outDw;
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerConvolution &layer ) {
        if( !write( layer.side()) || !write( layer.filterCount()) || !write( layer.stride()) || !write( layer.pad())) {
            return log_error( "Unable to write attributes" );
        }
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerConvolution( Dnn &dnn ) {
        unsigned long side;
        unsigned long filters;
        unsigned long stride;
        unsigned long pad;
        if( !read( side ) || !read( filters ) || !read( stride ) || !read( pad )) {
            return log_error( "Unable to read attributes" );
        }
        if( !dnn.addLayerConvolution( side, filters, stride, pad )) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerDropout &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerDropout( Dnn &dnn ) {
        if( !dnn.addLayerDropout()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerFullyConnected &layer ) {
        if( !write( layer.getNeuronCount())) {
            return log_error( "Unable to write attribute" );
        }
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerFullyConnected( Dnn &dnn ) {
        unsigned long neuronCount;
        if( !read( neuronCount )) {
            return log_error( "Unable to read attribute" );
        }
        if( !dnn.addLayerFullyConnected( neuronCount )) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerLocalResponseNormalization &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerLocalResponseNormalization( Dnn &dnn ) {
        if( !dnn.addLayerLocalResponseNormalization()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerMaxout &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerMaxout( Dnn &dnn ) {
        if( !dnn.addLayerMaxout()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerPool &layer ) {
        if( !write( layer.side()) || !write( layer.stride()) || !write( layer.pad())) {
            return log_error( "Unable to write attributes" );
        }
        const unsigned long *switches = layer.switchsReadOnly();
        unsigned long     switchCount = layer.switchCount();
        if( !writeArray( switches, switchCount )) {
            return log_error( "Unable to write switches" );
        }
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerPool( Dnn &dnn ) {
        unsigned long side;
        unsigned long stride;
        unsigned long pad;
        if( !read( side ) || !read( stride ) || !read( pad )) {
            return log_error( "Unable to read attributes" );
        }
        if( !dnn.addLayerPool( side, stride, pad )) {
            return log_error( "Unable to add layer" );
        }
        DnnLayerPool *layer = (DnnLayerPool*) dnn.getLayerOutput();
        if( layer == 0 ) {
            return log_error( "Unable to get the last layer." );
        }
        const unsigned long  expectedCount = layer->switchCount();
              unsigned long *layerSwitches = layer->switchs();
        const unsigned long      *switches = readArrayUnsignedLong( expectedCount );
        if( layerSwitches != 0 && switches != 0 ) {
            copy( layerSwitches, switches, expectedCount );
        }
        delete[] switches;
        return readLayerBase( layer );    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerRectifiedLinearUnit &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerRectifiedLinearUnit( Dnn &dnn ) {
        if( !dnn.addLayerRectifiedLinearUnit()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerRegression &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerRegression( Dnn &dnn ) {
        if( !dnn.addLayerRegression()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerSigmoid &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerSigmoid( Dnn &dnn ) {
        if( !dnn.addLayerSigmoid()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerSoftmax &layer ) {
        writeMatrix( layer.exponentialsReadOnly());
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerSoftmax( Dnn &dnn ) {
        if( !dnn.addLayerSoftmax()) {
            return log_error( "Unable to add layer" );
        }
        DnnLayerSoftmax *layer = (DnnLayerSoftmax*) dnn.getLayerOutput();
        if( layer == 0 ) {
            return log_error( "Unable to obtain last layer" );
        }
        Matrix *exponentials      = readMatrix();
        Matrix *layerExponentials = layer->exponentials();
        if( layerExponentials != 0 && exponentials != 0 ) {
            layerExponentials->copy( *exponentials );
        }
        delete exponentials;
        return readLayerBase( layer );
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerSupportVectorMachine &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerSupportVectorMachine( Dnn &dnn ) {
        if( !dnn.addLayerSupportVectorMachine()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerTanh &layer ) {
        return writeLayerBase( layer );
    }

    bool
    InDnnStream::readLayerTanh( Dnn &dnn ) {
        if( !dnn.addLayerTanh()) {
            return log_error( "Unable to add layer" );
        }
        return readLayerBase( dnn.getLayerOutput());    // Fill layer just added.
    }
    
    bool
    OutDnnStream::writeDnn( const Dnn &dnn ) {
        if( !writeHeader()) {
            return log_error( "Unable to write header" );
        }
        if( !write( OBJECT_DNN )) {
            return log_error( "Unable to write DNN enum" );
        }
        bool trainable = dnn.trainable();
        if( !write( trainable )) {
            return log_error( "Unable to write DNN trainable" );
        }
        unsigned long layerCount = dnn.count();
        if( !write( layerCount )) {
            return log_error( "Unable to write DNN layerCount" );
        }
        const DnnLayerInput *inputLayer = dnn.getLayerInputReadOnly();
        if( inputLayer == 0 ) {
            return log_error( "Cannot obtain the input layer" );
        }
        if( !writeLayerBegin( inputLayer->layerType()) || !writeLayer( *inputLayer )) {
            return log_error( "Unable to write input layer" );
        }
        const DnnLayer *layer = inputLayer->getNextLayer();
        while( layer != 0 ) {
            const LayerType layerType = layer->layerType();
            if( !writeLayerBegin( layerType )) {
                return log_error( "Unable write layer beginning" );
            }
            switch ( layerType ) {
                case LAYER_INPUT:                        writeLayer((const DnnLayerInput&)                      *layer ); break;
                case LAYER_CONVOLUTION:                  writeLayer((const DnnLayerConvolution&)                *layer ); break;
                case LAYER_DROPOUT:                      writeLayer((const DnnLayerDropout&)                    *layer ); break;
                case LAYER_FULLY_CONNECTED:              writeLayer((const DnnLayerFullyConnected&)             *layer ); break;
                case LAYER_LOCAL_RESPONSE_NORMALIZATION: writeLayer((const DnnLayerLocalResponseNormalization&) *layer ); break;
                case LAYER_MAXOUT:                       writeLayer((const DnnLayerMaxout&)                     *layer ); break;
                case LAYER_POOL:                         writeLayer((const DnnLayerPool&)                       *layer ); break;
                case LAYER_RECTIFIED_LINEAR_UNIT:        writeLayer((const DnnLayerRectifiedLinearUnit&)        *layer ); break;
                case LAYER_REGRESSION:                   writeLayer((const DnnLayerRegression&)                 *layer ); break;
                case LAYER_SIGMOID:                      writeLayer((const DnnLayerSigmoid&)                    *layer ); break;
                case LAYER_SOFTMAX:                      writeLayer((const DnnLayerSoftmax&)                    *layer ); break;
                case LAYER_SUPPORT_VECTOR_MACHINE:       writeLayer((const DnnLayerSupportVectorMachine&)       *layer ); break;
                case LAYER_TANH:                         writeLayer((const DnnLayerTanh&)                       *layer ); break;
                default: log_error( "Unknown layer type: %d", layerType );
            }
            layer = layer->getNextLayer();
        }
        
        return write( END_TAG, NN_FILE_TAG_LENGTH );
    }
    
    Dnn*
    InDnnStream::readDnn( bool trainable ) {
        unsigned short contentVersion = 0;
        if( !readHeader( contentVersion ) || contentVersion != CONTENT_VERSION ) {
            log_error( "Unable to read header" );
            return 0;
        }
        if( !expectEnum( OBJECT_DNN )) {
            log_error( "Unable to read object type dnn" );
            return 0;
        }
        bool savedIsTrainable = false;
        if( !read( savedIsTrainable )) {
            return 0;
        }
        unsigned long layerCount = 0;
        if( !read( layerCount )) {
            return 0;
        }
        Dnn *dnn = new Dnn( trainable );
        for( unsigned long ii = 0; ii < layerCount; ii++ ) {
            if( !expectEnum( OBJECT_LAYER )) {
                log_error( "Unable to read object type Layer %lu", ii );
                delete dnn;
                return 0;
            }
            int layerType;
            if( !readEnum( layerType, LAYER_COUNT )) {
                log_error( "Unable to read layer type" );
                delete dnn;
                return 0;
            }
            switch ( layerType ) {
                case LAYER_INPUT:                        readLayerInput(                      *dnn ); break;    // input
                case LAYER_CONVOLUTION:                  readLayerConvolution(                *dnn ); break;    // conv
                case LAYER_DROPOUT:                      readLayerDropout(                    *dnn ); break;    // dropout
                case LAYER_FULLY_CONNECTED:              readLayerFullyConnected(             *dnn ); break;    // fc
                case LAYER_LOCAL_RESPONSE_NORMALIZATION: readLayerLocalResponseNormalization( *dnn ); break;    // lrn
                case LAYER_MAXOUT:                       readLayerMaxout(                     *dnn ); break;    // maxout
                case LAYER_POOL:                         readLayerPool(                       *dnn ); break;    // pool
                case LAYER_RECTIFIED_LINEAR_UNIT:        readLayerRectifiedLinearUnit(        *dnn ); break;    // relu
                case LAYER_REGRESSION:                   readLayerRegression(                 *dnn ); break;    // regression
                case LAYER_SIGMOID:                      readLayerSigmoid(                    *dnn ); break;    // sigmoid
                case LAYER_SOFTMAX:                      readLayerSoftmax(                    *dnn ); break;    // softmax
                case LAYER_SUPPORT_VECTOR_MACHINE:       readLayerSupportVectorMachine(       *dnn ); break;    // svm
                case LAYER_TANH:                         readLayerTanh(                       *dnn ); break;    // tanh
                default: log_error( "Unknown layer type: %d", layerType );
            }
        }
        
        if( !readTag( END_TAG )) {
            delete dnn;
            return 0;
        }
        return dnn;
    }

}   // namespace tfs

