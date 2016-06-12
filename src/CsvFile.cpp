//
//  CsvFile.cpp
//
//  Created by Barrett Davis on 6/12/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <fstream>
#include <iostream>
#include <sstream>      // String stream
#include "CsvFile.h"
#include "Error.h"

namespace tfs {         // Tree Frog Software
    
    bool
    csvGetColumn( std::string &value, std::istream &stream, char delimiter = 0 ) {
        if( !stream.good()) {   // Not the same as stream.bad()
            value.clear();
            return false;
        }
        if( delimiter != 0 ) {
            std::getline( stream, value, delimiter );
        } else {
            std::getline( stream, value );
        }
        const size_t count = value.length();
        if( count > 2 && value[0] == '"' ) {                // Check for surrounding double quotes
            size_t last = count-1;
            if( value[last] != '"' ) {
                last = value.find_last_of( '"' );            // Sometimes there may be trailing whitespace after the last quote.
            }
            if( last != std::string::npos && last > 0  ) {   // Remove surrounding double quotes, if any.
                value = value.substr( 1, last-1 );
            }
        }
        return !stream.bad();   // Not the same as stream.good() - eof
    }

    static unsigned long countColumns( std::string &line ) {
        if( line.length() < 1 ) {
            return 0;
        }
        unsigned long count = 0;
        std::istringstream stream( line );
        std::string buffer;
        while( csvGetColumn( buffer, stream, ',' )) {
            count++;
        }
        return count;
    }
    
    static bool
    preflight( const char *path, unsigned long &columnCount, unsigned long &rowCount ) {
        columnCount = 0;
        rowCount    = 0;
        std::ifstream stream( path );
        if( stream.bad()) {
            return log_error("Cannot open file: %s", path );
        }
        std::string line;
        if( !std::getline( stream, line )) {        // Read the header row
            stream.close();
            return log_error( "Cannot read header from file: %s", path );
        }
        columnCount = countColumns( line );
        line.clear();
        while( std::getline( stream, line )) {
            if( line.length() > 0 ) {
                line.clear();
                rowCount++;
            }
        }
        stream.close();
        return true;
    }
    
    static DNN_NUMERIC*
    processRow( DNN_NUMERIC *data, std::string &line, const double notPresent ) {
        if( line.length() < 1 ) {
            return data;
        }
        std::istringstream stream( line );
        std::string buffer;
        while( csvGetColumn( buffer, stream, ',' )) {
            if( buffer.length() > 0 ) {
                *data++ = atof( buffer.c_str());
                buffer.clear();
            } else {
                *data++ = notPresent;
            }
        }
        return data;
    }
    
    static Matrix*
    fill( Matrix *matrix, const char *path, const double notPresent ) {
        std::ifstream stream( path );
        if( stream.bad()) {
            log_error("Cannot open file: %s", path );
            return matrix;
        }
        std::string line;
        if( !std::getline( stream, line )) {        // Read the header row
            stream.close();
            log_error( "Cannot read header from file: %s", path );
            return matrix;
        }
        DNN_NUMERIC *data = matrix->data();
        line.clear();
        while( std::getline( stream, line )) {
            data = processRow( data, line, notPresent );
            line.clear();
        }
        stream.close();
        return matrix;
    }
    
    Matrix*
    csvReadFile( const char *path, const double notPresent ) {
        if( path == 0 || *path == 0 ) {
            log_error( "Path is bad" );
            return 0;
        }
        unsigned long columnCount = 0;
        unsigned long rowCount    = 0;
        if( !preflight( path, columnCount, rowCount )) {
            return 0;
        }
        Matrix *matrix = new Matrix( columnCount, rowCount );
        return fill( matrix, path, notPresent );
    }
    
    
    
}   // namespace tfs
