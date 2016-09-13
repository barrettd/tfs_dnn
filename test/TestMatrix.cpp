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
        const unsigned long A_MAX = 15;
        const unsigned long B_MAX = 4;
        const unsigned long C_MAX = 2;
        const unsigned long D_MAX = 3;
        const unsigned long EXPECTED_COUNT = A_MAX * B_MAX * C_MAX * D_MAX;
        DMatrix matrix = DMatrix( A_MAX, B_MAX, C_MAX, D_MAX );
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
        if( matrix.width() != A_MAX || matrix.aa() != A_MAX ) {
            return log_error( "Width != %lu", A_MAX );
        }
        if( matrix.height() != B_MAX || matrix.bb() != B_MAX ) {
            return log_error( "Height != %lu", B_MAX );
        }
        if( matrix.depth() != C_MAX || matrix.cc() != C_MAX ) {
            return log_error( "Depth != %lu", C_MAX );
        }
        if( matrix.dd() != D_MAX ) {
            return log_error( "4th dimension != %lu", D_MAX );
        }
        DNN_INTEGER count = 0;
        for( unsigned long dd = 0; dd < D_MAX; dd++ ) {
            for( unsigned long cc = 0; cc < C_MAX; cc++ ) {
                for( unsigned long bb = 0; bb < B_MAX; bb++ ) {
                    for( unsigned long aa = 0; aa < A_MAX; aa++ ) {
                        DNN_INTEGER val = matrix.get( aa, bb, cc, dd );
                        val += count++;
                        matrix.set( aa, bb, cc, dd, val );
                        matrix.plusEquals( aa, bb, cc, dd, count++ );
                    }
                }
            }
        }
        count = 0;
        for( unsigned long dd = 0; dd < D_MAX; dd++ ) {
            for( unsigned long cc = 0; cc < C_MAX; cc++ ) {
                for( unsigned long bb = 0; bb < B_MAX; bb++ ) {
                    for( unsigned long aa = 0; aa < A_MAX; aa++ ) {
                        matrix.set( aa, bb, cc, dd, count++ );
                    }
                }
            }
        }
        count = 0;
        for( unsigned long dd = 0; dd < D_MAX; dd++ ) {
            for( unsigned long cc = 0; cc < C_MAX; cc++ ) {
                for( unsigned long bb = 0; bb < B_MAX; bb++ ) {
                    for( unsigned long aa = 0; aa < A_MAX; aa++ ) {
                        DNN_INTEGER val = matrix.get( aa, bb, cc, dd );
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
            log_warn( "dot product of two random matrices is 12.0, this is a surprise" );
        }
        return true;
    }
    
    static bool
    localTestMatrixMath2( void ) {
        Matrix aa = Matrix( 3, 1, 1 );
        Matrix bb = Matrix( 3, 1, 1 );
        aa.set( 0, 0.0 );
        aa.set( 1, 1.0 );
        aa.set( 2, 2.0 );
        bb[0] = 0.0;
        bb[1] = 1.0;
        bb[2] = 2.0;
        if( !aa.equal( bb )) {
            return log_error( "aa && bb matrices are not the same." );
        }
        aa.normalize();
        bb[1] = 1.0 / 3.0;
        bb[2] = 2.0 / 3.0;
        if( !aa.equal( bb )) {
            return log_error( "aa && bb matrices are not the same." );
        }
        aa.multiply( 3.0 );
        bb.divide( 1.0 / 3.0 );
        if( !aa.equal( bb )) {
            return log_error( "aa && bb matrices are not the same." );
        }
        return true;
    }
    
    static bool
    localTestMatrixCopy( void ) {
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
    
    static void
    fill( Matrix &matrix ) {
        const unsigned long maxA = matrix.aa();
        const unsigned long maxB = matrix.bb();
        const unsigned long maxC = matrix.cc();
        const unsigned long maxD = matrix.dd();
        DNN_NUMERIC value = 0.0;
        for( unsigned long dd = 0; dd < maxD; dd++ ) {
            for( unsigned long cc = 0; cc < maxC; cc++ ) {
                for( unsigned long bb = 0; bb < maxB; bb++ ) {
                    for( unsigned long aa = 0; aa < maxA; aa++ ) {
                        matrix.set( aa, bb, cc, dd, value );
                        value += 1.0;
                    }
                }
            }
        }
        return;
    }
    
    static bool
    localTestMatrixEqual( void ) {
        Matrix left( 10, 10, 3, 2 );
        Matrix right(10, 10, 3, 2 );
        fill( left );
        fill( right );
        if( !left.equal( right )) {
            return log_error( "matrices are equal, but reporting as different" );
        }
        right.set( 5, 5, 1, 1, 3.1415926 );
        if( left.equal( right )) {
            return log_error( "matrices are different, but reporting as equal" );
        }
        return true;
    }

    static bool
    localTestMatrixSubsample( void ) {
        Matrix source( 4, 4, 3 );
        Matrix destination( 2, 2, 3 );
        
        fill( source );
        destination.zero();
        
        unsigned long dx = 1;
        unsigned long dy = 1;
        if( !subsample( destination, source, dx, dy )) {
            return false;
        }
        const unsigned long maxY = destination.height();
        const unsigned long maxX = destination.width();
        const unsigned long maxZ = destination.depth();
        for( unsigned long zz = 0; zz < maxZ; zz++ ) {
            for( unsigned long yy = 0; yy < maxY; yy++ ) {
                for( unsigned long xx = 0; xx < maxX; xx++ ) {
                    const DNN_NUMERIC srcValue = source.get( xx + dx, yy + dy, zz );
                    const DNN_NUMERIC dstValue = destination.get( xx, yy, zz );
                    if( srcValue != dstValue ) {
                        return log_error( "Values do not match: %f/%f, %lu, %lu, %lu", srcValue, dstValue, xx, yy, zz );
                    }
                }
            }
        }
        dx = 1;
        dy = 2;
        if( !subsample( destination, source, dx, dy )) {
            return false;
        }
        for( unsigned long zz = 0; zz < maxZ; zz++ ) {
            for( unsigned long yy = 0; yy < maxY; yy++ ) {
                for( unsigned long xx = 0; xx < maxX; xx++ ) {
                    const DNN_NUMERIC srcValue = source.get( xx + dx, yy + dy, zz );
                    const DNN_NUMERIC dstValue = destination.get( xx, yy, zz );
                    if( srcValue != dstValue ) {
                        return log_error( "Values do not match: %f/%f, %lu, %lu, %lu", srcValue, dstValue, xx, yy, zz );
                    }
                }
            }
        }
        dx = 0;
        dy = 0;
        if( !subsample( destination, source, dx, dy )) {
            return false;
        }
        for( unsigned long zz = 0; zz < maxZ; zz++ ) {
            for( unsigned long yy = 0; yy < maxY; yy++ ) {
                for( unsigned long xx = 0; xx < maxX; xx++ ) {
                    const DNN_NUMERIC srcValue = source.get( xx + dx, yy + dy, zz );
                    const DNN_NUMERIC dstValue = destination.get( xx, yy, zz );
                    if( srcValue != dstValue ) {
                        return log_error( "Values do not match: %f/%f, %lu, %lu, %lu", srcValue, dstValue, xx, yy, zz );
                    }
                }
            }
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
        if( !localTestMatrixMath2()) {
            return false;
        }
        if( !localTestMatrixCopy()) {
            return false;
        }
        if( !localTestMatrixEqual()) {
            return false;
        }
        if( !localTestMatrixSubsample()) {
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
