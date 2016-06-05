// --------------------------------------------------------------------
//  Error.h - Error reporting macros.
//  These are fairly simple reporting macros that are used to report state within the library.
//  log_debug() and log_info() are not be called from a production version of this library.
//  log_warn() and log_error() are called to report warning and error conditions.
//  Feel free to call these macros from your code and to modify the implementation to better suit your environment.
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef Error_h
#define Error_h

namespace tfs {     // Tree Frog Software

    void report_message( const char *level, const char *file, const char *function, const unsigned long line, const char *format, ... );
    bool report_problem( const char *level, const char *file, const char *function, const unsigned long line, const char *format, ... );
 
#define log_debug( message, args... ) report_message( "Debug", __FILE__, __FUNCTION__, __LINE__, message, ## args )
#define log_info(  message, args... ) report_message( "Info ", __FILE__, __FUNCTION__, __LINE__, message, ## args )
#define log_warn(  message, args... ) report_problem( "Warn ", __FILE__, __FUNCTION__, __LINE__, message, ## args )
#define log_error( message, args... ) report_problem( "Error", __FILE__, __FUNCTION__, __LINE__, message, ## args )

}   // namespace tfs

#endif /* Error_h */
