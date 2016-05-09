//
//  Error.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cstdarg>  // va_list
#include <cstdio>   // vsnprintf
#include <iostream>
#include "Error.h"

namespace tfs {     // Tree Frog Software

    static void va_log( std::ostream &stream, const bool flush, const char *level, const char *file, const char *function, const unsigned long line, const char *format, va_list &args );

    static void va_log( std::ostream &stream, const bool flush, const char *level, const char *file, const char *function, const unsigned long line, const char *format, va_list &args ) {
        if( stream.bad() || format == 0 || format[0] == 0 ) {
            return;
        }
        if( level != 0 ) {
            stream << level << ": ";
        } else {
            level = "Unkn : ";
        }
        if( file != 0 ) {
            stream << file  << " ";
        }
        if( function != 0 ) {
            stream << function << " ";
        }
        stream << line << ": ";
        
        const int    BUFFER_SIZE = 4096;
        char buffer[ BUFFER_SIZE ];
        buffer[0] = 0;
        
        vsnprintf( buffer, BUFFER_SIZE, format, args );
        stream << buffer << "\n";
        if( flush ) {
            // Streams will generally accumulate a buffer[4096] before performing a flush() on their own.
            stream.flush();
        }
        return;
    }
    
    void
    report_message( const char *level, const char *file, const char *function, const unsigned long line, const char *format, ... ) {
        va_list args;
        va_start( args, format );
        va_log( std::cout, false, level, file, function, line, format, args );
        va_end( args );
        return;
    }
    
    bool
    report_problem( const char *level, const char *file, const char *function, const unsigned long line, const char *format, ... ) {
        va_list args;
        va_start( args, format );
        va_log( std::cerr, true, level, file, function, line, format, args );
        va_end( args );
        return false;       // Always returns false.
    }

}   // namespace tfs
