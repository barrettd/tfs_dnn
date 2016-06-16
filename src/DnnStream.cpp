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
        
        OBJECT_COUNT        // Used for range checking.
    };
 
    static const int NN_FILE_TAG_LENGTH  = 4;   // "tfs", "dnn", "net", "dat", etc.

    static const char START_TAG[ NN_FILE_TAG_LENGTH ] = { "tfs" };
    static const char DNN_TAG[   NN_FILE_TAG_LENGTH ] = { "dnn" };
    
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
        char tag_buffer[ NN_FILE_TAG_LENGTH ];
        if( !read( tag_buffer, NN_FILE_TAG_LENGTH ) || strncmp( tag_buffer, START_TAG, NN_FILE_TAG_LENGTH ) != 0 ) {
            return false;
        }
        
        // Header format version, different from the content version.
        unsigned short format_version = 0;
        if( !read( format_version ) || format_version != HEADER_VERSION ) {
            return false;
        }
        
        // Check for our content tag: "dnn"
        if( !read( tag_buffer, NN_FILE_TAG_LENGTH ) || strncmp( tag_buffer, DNN_TAG, NN_FILE_TAG_LENGTH ) != 0 ) {
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
    OutDnnStream::writeDnn( Dnn &dnn ) {
        if( !writeHeader()) {
            return log_error( "Unable to write header" );
        }
        bool trainable = dnn.trainable();
        if( !write( trainable )) {
            return false;
        }
        unsigned long layerCount = dnn.count();
        if( !write( layerCount )) {
            return false;
        }
        DnnLayerInput *inputLayer  = dnn.getLayerInput();
        DnnLayer      *outputLayer = dnn.getLayerOutput();
        
        
        return true;
    }


    Dnn*
    InDnnStream::readDnn( bool trainable ) {
        unsigned short contentVersion = 0;
        if( !readHeader( contentVersion ) || contentVersion != CONTENT_VERSION ) {
            log_error( "Unable to read header" );
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
        

        
        return dnn;
    }

}   // namespace tfs

