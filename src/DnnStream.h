// ----------------------------------------------------------------------------
//  DnnStream.hpp
//  Binary file stream for the Neural Net and intermediate files.
//
//  Created by Barrett Davis on 6/15/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// ----------------------------------------------------------------------------
#include "BinaryStream.h"
#include "Dnn.h"

#ifndef DnnStream_h
#define DnnStream_h

namespace tfs {         // Tree Frog Software
    
    class OutDnnStream : public OutBinaryStream {
    protected:
        bool writeHeader( void );
        bool writeEnum( int value );

    public:
        OutDnnStream( const char *path );
        
        bool writeDnn( Dnn &dnn );
        
    };
    
    class InDnnStream : public InBinaryStream {
    protected:
        bool readHeader( unsigned short &contentVersion );
        
    public:
        InDnnStream( const char *path );

        Dnn *readDnn( bool trainable = false );

    };

}   // namespace tfs

#endif /* DnnStream_h */
