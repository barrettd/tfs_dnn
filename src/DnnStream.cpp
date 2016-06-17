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
// Type sizes,      char buffer,    8 bytes - 1,2,4,8,16,0,0,0,0
// Padding,         char buffer,    4 bytes - 0,0,0,0
// ----------------------------------------------------------------------------
// Total bytes:                    32 bytes - aligned to 8 byte boundary
// ----------------------------------------------------------------------------
#include "DnnLayers.h"
#include "DnnStream.h"
#include "Error.h"

namespace tfs {         // Tree Frog Software
    
    enum ObjectType {
        OBJECT_DNN = 0,
        OBJECT_LAYER,
        OBJECT_MATRIX,
        
        OBJECT_COUNT        // Used for range checking.
    };
 
    static const int NN_FILE_TAG_LENGTH  = 4;   // "tfs", "dnn", "net", "dat", etc.

    static const char START_TAG[ NN_FILE_TAG_LENGTH ] = { "tfs" };
    static const char DNN_TAG[   NN_FILE_TAG_LENGTH ] = { "dnn" };
    static const char END_TAG[   NN_FILE_TAG_LENGTH ] = { "eof" };
    
    static const unsigned short HEADER_VERSION  = 1;                   // Version of the header, not the content.
    static const unsigned short CONTENT_VERSION = 1;                   // Version of the content.
    static const int            TYPE_COUNT      = 8;
    static const unsigned long  BYTE_ORDER_TEST = 0x0102030405060708;  // 8 bytes
    static const int            PADDING_LENGTH  = 4;
    
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
        buffer[0] = (unsigned char) sizeof( bool );                 // 1
        buffer[1] = (unsigned char) sizeof( unsigned short );       // 2
        buffer[2] = (unsigned char) sizeof( unsigned int );         // 4
        buffer[3] = (unsigned char) sizeof( unsigned long  );       // 8
        buffer[4] = (unsigned char) sizeof( unsigned long long );   // 8
        buffer[5] = (unsigned char) sizeof( float );                // 4
        buffer[6] = (unsigned char) sizeof( double );               // 8
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
        unsigned long  byte_order_test = 0;
        if( !read( byte_order_test ) || byte_order_test != BYTE_ORDER_TEST ) {      // Check that the byte order is as expected
            return false;
        }
        unsigned char buffer[ TYPE_COUNT ];
        memset( buffer, 0, sizeof( buffer ));
        if( !read( buffer, sizeof( buffer ))) {
            return false;
        }
        if( buffer[0] != (unsigned char) sizeof( bool )               ||    // Check the sizes of our base types.
            buffer[1] != (unsigned char) sizeof( unsigned short )     ||
            buffer[2] != (unsigned char) sizeof( unsigned int )       ||
            buffer[3] != (unsigned char) sizeof( unsigned long  )     ||
            buffer[4] != (unsigned char) sizeof( unsigned long long ) ||
            buffer[5] != (unsigned char) sizeof( float  )             ||
            buffer[6] != (unsigned char) sizeof( double )             ||
            buffer[7] != (unsigned char) sizeof( long double )) {
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
        if( !write( matrix->aa()) || !write( matrix->bb()) || !write( matrix->cc()) || !write( matrix->dd())) {
            return log_error( "Unable to write matrix dimensions" );
        }
        const DNN_NUMERIC    *data = matrix->dataReadOnly();
        const unsigned long length = matrix->length();
        return write((const char*) data, length );
    }

    Matrix*
    InDnnStream::readMatrix( void ) {
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
    OutDnnStream::writeLayerBase( const DnnLayer *layer ) {
        if( layer == 0 || !write( OBJECT_LAYER ) || !write( layer->layerType())) {
            return log_error( "Unable to write basic layer data" );
        }
        
        return true;
    }
    
    bool
    InDnnStream::readLayerBase( DnnLayer *layer ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerInput *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }
    
    bool
    InDnnStream::readLayerInput( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerConvolution *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerConvolution( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerDropout *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerDropout( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerFullyConnected *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerFullyConnected( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerLocalResponseNormalization *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerLocalResponseNormalization( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerMaxout *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerMaxout( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerPool *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerPool( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerRectifiedLinearUnit *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerRectifiedLinearUnit( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerRegression *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerRegression( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerSigmoid *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerSigmoid( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerSoftmax *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerSoftmax( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerSupportVectorMachine *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerSupportVectorMachine( Dnn &dnn ) {
        return true;
    }
    
    bool
    OutDnnStream::writeLayer( const DnnLayerTanh *layer ) {
        if( !writeLayerBase( layer )) {
            return false;
        }
        return true;
    }

    bool
    InDnnStream::readLayerTanh( Dnn &dnn ) {
        return true;
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
        if( !writeLayer( inputLayer )) {
            return log_error( "Unable to write input layer" );
        }
        const DnnLayer *layer = inputLayer->getNextLayer();
        while( layer != 0 ) {
            const LayerType layerType = layer->layerType();
            switch ( layerType ) {
                case LAYER_INPUT:                        writeLayer((const DnnLayerInput*)                      layer ); break;
                case LAYER_CONVOLUTION:                  writeLayer((const DnnLayerConvolution*)                layer ); break;
                case LAYER_DROPOUT:                      writeLayer((const DnnLayerDropout*)                    layer ); break;
                case LAYER_FULLY_CONNECTED:              writeLayer((const DnnLayerFullyConnected*)             layer ); break;
                case LAYER_LOCAL_RESPONSE_NORMALIZATION: writeLayer((const DnnLayerLocalResponseNormalization*) layer ); break;
                case LAYER_MAXOUT:                       writeLayer((const DnnLayerMaxout*)                     layer ); break;
                case LAYER_POOL:                         writeLayer((const DnnLayerPool*)                       layer ); break;
                case LAYER_RECTIFIED_LINEAR_UNIT:        writeLayer((const DnnLayerRectifiedLinearUnit*)        layer ); break;
                case LAYER_REGRESSION:                   writeLayer((const DnnLayerRegression*)                 layer ); break;
                case LAYER_SIGMOID:                      writeLayer((const DnnLayerSigmoid*)                    layer ); break;
                case LAYER_SOFTMAX:                      writeLayer((const DnnLayerSoftmax*)                    layer ); break;
                case LAYER_SUPPORT_VECTOR_MACHINE:       writeLayer((const DnnLayerSupportVectorMachine*)       layer ); break;
                case LAYER_TANH:                         writeLayer((const DnnLayerTanh*)                       layer ); break;
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
        int objectType;
        if( !readEnum( objectType, OBJECT_COUNT ) || objectType != OBJECT_DNN ) {
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
            if( !readEnum( objectType, OBJECT_COUNT ) || objectType != OBJECT_LAYER ) {
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

