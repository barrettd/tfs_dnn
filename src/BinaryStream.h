//
//  BinaryStream.h
//
//  Created by Barrett Davis on 6/15/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <fstream>

#ifndef BinaryStream_h
#define BinaryStream_h

namespace tfs {         // Tree Frog Software
    
    class OutBinaryStream {
    protected:
        std::ofstream m_stream;
        
    public:
         OutBinaryStream( const char *path );
         virtual ~OutBinaryStream( void );
        
        inline void close( void ) { m_stream.close(); }
        inline bool good(  void ) const { return m_stream.good(); }
        inline bool bad(   void ) const { return m_stream.bad();  }
        inline bool operator!()   const { return !m_stream; }
        
        bool write( const char *buffer, unsigned long count );
        
        inline bool write( const unsigned char *buffer, unsigned long count ) { return write((const char *) buffer, count ); }
        
        inline bool write( const bool           value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const double         value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const float          value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const unsigned long  value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const unsigned int   value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const unsigned short value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const long           value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const int            value ) { return write(( const char *) &value, sizeof( value )); }
        inline bool write( const short          value ) { return write(( const char *) &value, sizeof( value )); }
    };
    
    
    class InBinaryStream {
    protected:
        std::ifstream m_stream;
        
    public:
        InBinaryStream( const char *path );
        virtual ~InBinaryStream( void );

        inline void close( void ) { m_stream.close(); }
        inline bool good(  void ) const { return m_stream.good(); }
        inline bool bad(   void ) const { return m_stream.bad();  }
        inline bool operator!()   const { return !m_stream; }

        bool read( char *buffer, unsigned long count );

        inline bool read( unsigned char *buffer, unsigned long count ) { return read((char*) buffer, count ); }

        inline bool read( const bool           &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const double         &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const float          &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const unsigned long  &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const unsigned int   &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const unsigned short &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const long           &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const int            &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( const short          &value ) { return read(( char *) &value, sizeof( value )); }
    };
    
}   // namespace tfs

#endif /* BinaryStream_h */
