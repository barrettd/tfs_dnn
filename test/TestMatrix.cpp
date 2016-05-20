//
//  TestMatrix.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/20/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "Matrix.h"
#include "TestMatrix.hpp"

namespace tfs {
    
    static bool
    localTestMatrixSize( void ) {
        DMatrix matrix = DMatrix( 2, 2, 1 );
        if( matrix.count() != 4 ) {
            return log_error( "Matrix count is not 4, it is %lu", matrix.count());
        }
        const unsigned long INTEGER_SIZE = sizeof( DNN_INTEGER );
        const unsigned long EXPECTED_LENGTH = matrix.count() * INTEGER_SIZE;
        if( matrix.length() != EXPECTED_LENGTH ) {
            return log_error( "Matrix count is not %lu, it is %lu", EXPECTED_LENGTH, matrix.length());
        }
              DNN_INTEGER *      data = matrix.data();
        const DNN_INTEGER * const end  = matrix.end();
        
        const DNN_INTEGER * const expectedEnd = &data[4];
        if( end != expectedEnd ) {
            return log_error( "End pointers do not match" );
        }
        matrix.zero();
        while( data < end ) {
            if( *data++ != 0 ) {
                return log_error( "Data element not zero" );
            }
        }
        if( matrix.width() != 2 ) {
            return log_error( "Width != 2" );
        }
        if( matrix.height() != 2 ) {
            return log_error( "Height != 2" );
        }
        if( matrix.depth() != 1 ) {
            return log_error( "Depth != 1" );
        }
        return true;
    }
    
    static bool
    localTestMatrixMath( void ) {
        Matrix aa = Matrix( 3, 1, 1 );
        Matrix bb = Matrix( 3, 1, 1 );
        DNN_NUMERIC *data = aa.data();
        *data++ = 1.0;
        *data++ = 2.0;
        *data++ = 3.0;
        data = bb.data();
        *data++ =  4.0;
        *data++ = -5.0;
        *data++ =  6.0;
        DNN_NUMERIC result = aa.dot( bb );
        if( result != 12.0 ) {
            return log_error( "dot product is not 12.0, it was %f", result );
        }
        result = aa.max();
        if( result != 3.0 ) {
            return log_error( "max is not 3.0, it was %f", result );
        }
        result = bb.max();
        if( result != 6.0 ) {
            return log_error( "max is not 6.0, it was %f", result );
        }
        bb.zero();
        result = bb.max();
        if( result != 0.0 ) {
            return log_error( "max is not 0.0, it was %f", result );
        }
        aa.randomize();
        bb.randomize();
        result = aa.dot( bb );
        if( result == 12.0 ) {
            log_warn( "dot product of two random matrices is 12.0, this is a suprise" );
        }
        return true;
    }
    
    static bool
    localTestMatrix( void ) {
        log_info( "Test Matrix - Start" );
        if( !localTestMatrixSize()) {
            return false;
        }
        if( !localTestMatrixMath()) {
            return false;
        }
        log_info( "Test Matrix - End" );
        return true;
    }

}  // tfs namespace

bool
testMatrix( void ) {
    return tfs::localTestMatrix();
}
