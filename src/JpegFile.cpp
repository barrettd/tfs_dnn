// ----------------------------------------------------------------------------
//  JpegFile.cpp
//
//  Created by Barrett Davis on 6/19/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
//  Much of this code (and comments) came from the jpeg lib example.c
// ----------------------------------------------------------------------------
#include "Error.h"
#include "JpegFile.h"
#include "Jpeg.h"


namespace tfs {
    
    Matrix*
    createMatrix( ImageRgb &image ) {
        Matrix *matrix = new Matrix( 3, image.width(), image.height());
        const unsigned char *in  = image.data();
        const unsigned char *end = in + image.size();
        DNN_NUMERIC         *out = matrix->data();
        while( in < end ) {
            *out++ = (DNN_NUMERIC) (*in++ & 0x00FF);
        }
        return matrix;
    }
    
    bool
    createImage( ImageRgb &image, const Matrix &matrix ) {
        const unsigned long matrixX = matrix.bb();
        const unsigned long matrixY = matrix.cc();
        image.create( matrixX, matrixY );
        const DNN_NUMERIC *in  = matrix.dataReadOnly();
        const DNN_NUMERIC *end = matrix.end();
        unsigned char     *out = image.data();
        while( in < end ) {
            DNN_NUMERIC value = *in++;
            if( value < 0.0 ) {
                value = 0.0;
            } else if( value > 255.0 ) {
                value = 255.0;
            }
            *out++ = (unsigned char) value;
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


