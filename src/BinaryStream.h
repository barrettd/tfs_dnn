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
    
    class OutBinaryStream {                     // Binary output stream
    protected:
        std::ofstream m_stream;
        
    public:
         OutBinaryStream( const char *path );
         virtual ~OutBinaryStream( void );
        
        inline void close( void ) { m_stream.close(); }
        inline bool good(  void ) const { return m_stream.good(); }
        inline bool bad(   void ) const { return m_stream.bad();  }
        inline bool operator!()   const { return !m_stream; }
        
        bool write( const char *buffer, unsigned long count );  // All of the writes go through here.
        
        inline bool write( const unsigned char *buffer, unsigned long count ) { return write((const char *) buffer, count ); }
        
        inline bool write( const bool               value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const long double        value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const double             value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const float              value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const unsigned long long value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const unsigned long      value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const unsigned int       value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const unsigned short     value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const long long          value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const long               value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const int                value ) { return write((const char *) &value, sizeof( value )); }
        inline bool write( const short              value ) { return write((const char *) &value, sizeof( value )); }
        
        inline OutBinaryStream& operator<<(const bool               value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const long double        value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const double             value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const float              value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const unsigned long long value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const unsigned long      value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const unsigned int       value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const unsigned short     value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const long long          value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const long               value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const int                value) { write((const char *) &value, sizeof( value )); return *this; }
        inline OutBinaryStream& operator<<(const short              value) { write((const char *) &value, sizeof( value )); return *this; }
    };
    
    
    class InBinaryStream {                      // Binary input stream.
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

        inline bool read( bool               &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( long double        &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( double             &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( float              &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( unsigned long long &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( unsigned long      &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( unsigned int       &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( unsigned short     &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( long long          &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( long               &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( int                &value ) { return read(( char *) &value, sizeof( value )); }
        inline bool read( short              &value ) { return read(( char *) &value, sizeof( value )); }
        
        inline InBinaryStream& operator>>(bool               &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(long double        &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(double             &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(float              &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(unsigned long long &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(unsigned long      &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(unsigned int       &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(unsigned short     &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(long long          &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(long               &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(int                &value) { read((char*) &value, sizeof( value )); return *this; }
        inline InBinaryStream& operator>>(short              &value) { read((char*) &value, sizeof( value )); return *this; }
    };
    
}   // namespace tfs

#endif /* BinaryStream_h */
