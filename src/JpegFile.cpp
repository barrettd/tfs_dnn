// ----------------------------------------------------------------------------
//  JpegFile.cpp
//
//  Created by Barrett Davis on 6/19/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
//  Much of this code (and comments) came from the jpeg lib example.c
// ----------------------------------------------------------------------------
#include "Error.hpp"
#include "JpegFile.hpp"
#include "Jpeg.hpp"


namespace tfs {
    
    Matrix*
    createMatrix( ImageRgb &image ) {
        const unsigned long maxX = image.width();
        const unsigned long maxY = image.height();
        const unsigned long maxZ = 3;                   // RGB
        Matrix *matrix = new Matrix( maxX, maxY, maxZ );
        const unsigned char *in  = image.data();
        for( unsigned long yy = 0; yy < maxY; yy++ ) {
            for( unsigned long xx = 0; xx < maxX; xx++ ) {
                for( unsigned long zz = 0; zz < maxZ; zz++ ) {
                    const DNN_NUMERIC value = (DNN_NUMERIC) (*in++ & 0x00FF);
                    matrix->set( xx, yy, zz, value );
                }
            }
        }
        return matrix;
    }
    
    bool
    createImage( ImageRgb &image, const Matrix &matrix ) {
        const unsigned long maxX = matrix.width();
        const unsigned long maxY = matrix.height();
        const unsigned long maxZ = 3;                   // RGB
        image.create( maxX, maxY );
        unsigned char *dst = image.data();
        for( unsigned long yy = 0; yy < maxY; yy++ ) {
            for( unsigned long xx = 0; xx < maxX; xx++ ) {
                for( unsigned long zz = 0; zz < maxZ; zz++ ) {
                    DNN_NUMERIC value = matrix.get( xx, yy, zz );
                    if( value < 0.0 ) {
                        value = 0.0;
                    } else if( value > 255.0 ) {
                        value = 255.0;
                    }
                    *dst++ = (unsigned char) value;
                }
            }
        }
        return true;
    }
    
    Matrix*
    readJpeg( const char *fileName ) {
        if( fileName == 0 || *fileName == 0 ) {
            log_error( "Bad file name" );
            return 0;
        }
        ImageRgb image;
        if( !jpegRead( image, fileName )) {
            return 0;
        }
        return createMatrix( image );
    }
    
    bool
    writeJpeg( const Matrix &matrix, const char *fileName, int quality ) {
        if( fileName == 0 || *fileName == 0 ) {
            return log_error( "Bad file name" );
        }
        ImageRgb image;
        if( !createImage( image, matrix )) {
            return false;
        }
        return jpegWrite( image, fileName, quality );   // Quality bounds checked in jpegWrite()
    }

    
}  // tfs namespace


