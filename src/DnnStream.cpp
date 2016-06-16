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
// Type sizes,      char buffer,    8 bytes - 1,2,8,8,0,0,0,0
// Padding,         char buffer,    4 bytes - 0,0,0,0
// ----------------------------------------------------------------------------
// Total bytes:                    32 bytes - aligned to 8 byte boundary
// ----------------------------------------------------------------------------
#include "DnnLayers.h"
#include "DnnStream.h"
#include "Error.h"

namespace tfs {         // Tree Frog Software
 
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
        buffer[0] = (unsigned char) sizeof( bool );
        buffer[1] = (unsigned char) sizeof( unsigned short );
        buffer[2] = (unsigned char) sizeof( unsigned long  );
        buffer[3] = (unsigned char) sizeof( float );
        buffer[4] = (unsigned char) sizeof( double );           // [5] = [6] = [7] = 0;
        if( !this->write( buffer, sizeof( buffer ))) {          // (8) bytes
            return false;
        }
        memset( buffer, 0, sizeof( buffer ));                   // Clear buffer (again)
        return this->write( buffer, PADDING_LENGTH );           // (4) bytes of padding (zeros)
    }
    
    unsigned short
    InDnnStream::readHeader( void ) {
        if( m_stream.bad()) {
            return 0;
        }
        char tag_buffer[ NN_FILE_TAG_LENGTH ];
        read( tag_buffer, NN_FILE_TAG_LENGTH );
        if( strncmp( tag_buffer, START_TAG, NN_FILE_TAG_LENGTH ) != 0 ) {   // Check for our start tag: "tfs"
            return 0;
        }
        unsigned short format_version = 0;
        read( format_version );
        if( format_version != HEADER_VERSION ) {                            // Header format version, different from the content version.
            return 0;
        }
        read( tag_buffer, NN_FILE_TAG_LENGTH );
        if( strncmp( tag_buffer, DNN_TAG, NN_FILE_TAG_LENGTH ) != 0 ) {     // Check for our content tag: "dnn"
            return 0;
        }
        unsigned short content_version = 0;             // Version of the content (after this header)
        read( content_version );
        unsigned long  byte_order_test = 0;
        read( byte_order_test );
        if( byte_order_test != BYTE_ORDER_TEST ) {      // Check that the byte order is as expected
            return 0;
        }
        unsigned char buffer[ TYPE_COUNT ];
        memset( buffer, 0, sizeof( buffer ));
        if( !read( buffer, sizeof( buffer ))) {
            return 0;
        }
        if( buffer[0] != (unsigned char) sizeof( bool )           ||    // Check the sizes of our base types.
            buffer[1] != (unsigned char) sizeof( unsigned short ) ||
            buffer[2] != (unsigned char) sizeof( unsigned long  ) ||
            buffer[3] != (unsigned char) sizeof( float  )         ||
            buffer[4] != (unsigned char) sizeof( double )) {
            return 0;
        }
        if( !read( buffer, PADDING_LENGTH )) {          // Consume padding at end of header.
            return 0;
        }
        return content_version;                         // Return the content version
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
        if( readHeader() != CONTENT_VERSION ) {
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

