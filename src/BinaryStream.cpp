//
//  BinaryStream.cpp
//
//  Created by Barrett Davis on 6/15/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "BinaryStream.hpp"

namespace tfs {         // Tree Frog Software
    
    OutBinaryStream::OutBinaryStream( const char *path ):
    m_stream( path, std::ofstream::binary ) {
    }
    
    OutBinaryStream::~OutBinaryStream( void ) {
    }
    
    bool
    OutBinaryStream::write( const char *buffer, unsigned long count ) {
        if( m_stream.bad() || buffer == 0 ) {
            return false;
        }
        if( count < 1 ) {
            return true;
        }
        m_stream.write( buffer, (std::streamsize)count );
        return m_stream.good();
    }

    
    InBinaryStream::InBinaryStream( const char *path ):
    m_stream( path, std::ifstream::binary ) {
    }
    
    InBinaryStream::~InBinaryStream( void ) {
    }
    
    bool
    InBinaryStream::read( char *buffer, unsigned long count ) {
        if( m_stream.bad() || buffer == 0 ) {
            return false;
        }
        if( count < 1 ) {
            return true;
        }
        m_stream.read( buffer, (std::streamsize)count );
        return m_stream.good();
    }


}   // namespace tfs

