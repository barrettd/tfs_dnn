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
        const unsigned long X_MAX = 15;
        const unsigned long Y_MAX = 4;
        const unsigned long Z_MAX = 2;
        const unsigned long D_MAX = 3;
        const unsigned long EXPECTED_COUNT = X_MAX * Y_MAX * Z_MAX * D_MAX;
        DMatrix matrix = DMatrix( X_MAX, Y_MAX, Z_MAX, D_MAX );
        if( matrix.count() != EXPECTED_COUNT ) {
            return log_error( "Matrix count is not %lu, it is %lu", EXPECTED_COUNT, matrix.count());
        }
        const unsigned long INTEGER_SIZE = sizeof( DNN_INTEGER );
        const unsigned long EXPECTED_LENGTH = matrix.count() * INTEGER_SIZE;
        if( matrix.length() != EXPECTED_LENGTH ) {
            return log_error( "Matrix count is not %lu, it is %lu", EXPECTED_LENGTH, matrix.length());
        }
              DNN_INTEGER *      data = matrix.data();
        const DNN_INTEGER * const end  = matrix.end();
        
        const DNN_INTEGER * const expectedEnd = &data[EXPECTED_COUNT];
        if( end != expectedEnd ) {
            return log_error( "End pointers do not match" );
        }
        matrix.zero();
        while( data < end ) {
            if( *data++ != 0 ) {
                return log_error( "Data element not zero" );
            }
        }
        if( matrix.width() != X_MAX || matrix.aa() != X_MAX ) {
            return log_error( "Width != %lu", X_MAX );
        }
        if( matrix.height() != Y_MAX || matrix.bb() != Y_MAX ) {
            return log_error( "Height != %lu", Y_MAX );
        }
        if( matrix.depth() != Z_MAX || matrix.cc() != Z_MAX ) {
            return log_error( "Depth != %lu", Z_MAX );
        }
        if( matrix.dd() != D_MAX ) {
            return log_error( "4th dimension != %lu", D_MAX );
        }
        DNN_INTEGER count = 0;
        for( unsigned long d = 0; d < D_MAX; d++ ) {
            for( unsigned long z = 0; z < Z_MAX; z++ ) {
                for( unsigned long y; y < Y_MAX; y++ ) {
                    for( unsigned long x; x < X_MAX; x++ ) {
                        DNN_INTEGER val = matrix.get( x, y, z, d );
                        val += count++;
                        matrix.set( x, y, z, d, val );
                        matrix.plusEquals( x, y, z, count++ );
                    }
                }
            }
        }
        count = 0;
        for( unsigned long d = 0; d < D_MAX; d++ ) {
            for( unsigned long z = 0; z < Z_MAX; z++ ) {
                for( unsigned long y; y < Y_MAX; y++ ) {
                    for( unsigned long x; x < X_MAX; x++ ) {
                        matrix.set( x, y, z, d, count++ );
                    }
                }
            }
        }
        count = 0;
        for( unsigned long d = 0; d < D_MAX; d++ ) {
            for( unsigned long z = 0; z < Z_MAX; z++ ) {
                for( unsigned long y; y < Y_MAX; y++ ) {
                    for( unsigned long x; x < X_MAX; x++ ) {
                        DNN_INTEGER val = matrix.get( x, y, z, d );
                        if( val != count ) {
                            return log_error( "value %lu != %lu", val, count );
                        }
                        count++;
                    }
                }
            }
        }
        return true;
    }
    
    static bool
    localTestMatrixMath( void ) {
        Matrix aa = Matrix( 3, 1, 1 );
        Matrix bb = Matrix( 3, 1, 1 );
        if( !aa.ok() || !bb.ok()) {
            return log_error( "Poorly formed matrix" );
        }
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
        result = aa.min();
        if( result != 1.0 ) {
            return log_error( "min is not 1.0, it was %f", result );
        }
        result = aa.max();
        if( result != 3.0 ) {
            return log_error( "max is not 3.0, it was %f", result );
        }
        result = aa.sum();
        if( result != 6.0 ) {
            return log_error( "sum is not 6.0, it was %f", result );
        }
        result = bb.min();
        if( result != -5.0 ) {
            return log_error( "min is not -5.0, it was %f", result );
        }
        result = bb.max();
        if( result != 6.0 ) {
            return log_error( "max is not 6.0, it was %f", result );
        }
        result = bb.sum();
        if( result != 5.0 ) {
            return log_error( "sum is not 5.0, it was %f", result );
        }
        bb.zero();
        result = bb.min();
        if( result != 0.0 ) {
            return log_error( "min is not 0.0, it was %f", result );
        }
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
    localTestMatricCopy( void ) {
        Matrix aa = Matrix( 3, 2, 1 );
              DNN_NUMERIC *      data = aa.data();
        const DNN_NUMERIC * const end = aa.end();
        
        DNN_NUMERIC value = 0.0;
        while( data < end ) {
            *data++ = value += 1.0;     // Fill with predictable contents.
        }
        
        Matrix bb = Matrix( aa, true ); // Clone matrix aa
        
        if( !aa.equal( bb )) {
            return log_error( "aa && bb matrices are not the same." );
        }
        
        bb.randomize();
        if( aa.equal( bb )) {
            return log_error( "aa && bb matrices are the same." );
        }
        
        Matrix cc = Matrix( 1, 2, 3 );
        cc.copy( aa );                  // Same contents, different dimensions
        if( aa.equal( cc )) {
            return log_error( "aa && cc matrices are the same." );
        }
        Matrix dd = Matrix( cc );       // Create a matrix with the same dimension as dd, but do not copy.
        dd.copy( aa );                  // Copy contents from aa
        if( !cc.equal( dd )) {
            return log_error( "cc && dd matrices are not the same." );
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
        if( !localTestMatricCopy()) {
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
