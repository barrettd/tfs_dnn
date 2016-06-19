//
//  Matrix.cpp
//
//  Created by Barrett Davis on 6/18/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "Matrix.h"

namespace tfs {     // Tree Frog Software

    bool
    softmax( Matrix &dst, const Matrix &src ) {
        const unsigned long count = dst.count();
        if( count != src.count()) {
            return log_error( "Matrix dimensions do not match" );
        }
        const DNN_NUMERIC *         input = src.dataReadOnly();
        const DNN_NUMERIC * const   inEnd = src.end();
        const DNN_NUMERIC max  = src.max();     // Find input maximum
              DNN_NUMERIC esum = 0.0;
        
        DNN_NUMERIC exponent[count];
        DNN_NUMERIC *es = exponent;
        while( input < inEnd ) {                    // Compute exponentials
            const DNN_NUMERIC ee = exp( *input++ - max );
            *es++ = ee;
            esum += ee;
        }
        if( esum == 0.0 ) {                         // Avoid a divide by zero problem...
            log_error( "esum == 0.0 - divide by zero problem" );
            esum = 0.0001;
        }
              DNN_NUMERIC *        output = dst.data();
        const DNN_NUMERIC * const  outEnd = dst.end();
        es = exponent;
        while( output < outEnd ) {                  // normalize
            *output++ = *es++ / esum;
        }
        return true;
    }
    
    Matrix*
    gaussianKernel( const unsigned long side, const DNN_NUMERIC sigma ) {
        Matrix *kernel = new Matrix( side, side );
        const DNN_NUMERIC denominator = 2.0 * M_PI * sigma * sigma;
        const DNN_NUMERIC mean = side / 2.0;
        DNN_NUMERIC sum = 0.0;
        for( unsigned long xx = 0; xx < side; xx++ ) {
            for( unsigned long yy = 0; yy < side; yy++ ) {
                const DNN_NUMERIC value = exp( -0.5 * ( pow((xx-mean)/sigma, 2.0) + pow((yy-mean)/sigma, 2.0))) / denominator;
                kernel->set(xx, yy, value );
                sum += value;
            }
        }
        for( unsigned long xx = 0; xx < side; xx++ ) {
            for( unsigned long yy = 0; yy < side; yy++ ) {
                DNN_NUMERIC value = kernel->get( xx, yy );
                kernel->set( xx, yy, value / sum );
            }
        }
        return kernel;
    }
    
    Matrix*
    kernelOperation( const Matrix &src, const Matrix &kernel, const unsigned long stride ) {
        const unsigned long kernalX = kernel.aa();
        const unsigned long kernalY = kernel.bb();
        if( src.aa() < kernalX || src.bb() < kernalY ) {
            log_error( "Source matrix smaller than kernel" );
            return 0;
        }
        if( stride < 1 ) {
            log_error( "Stride too small" );
            return 0;
        }
        const unsigned long dstX = (unsigned long) floor((src.aa() - kernel.aa()) / stride + 1.0 );
        const unsigned long dstY = (unsigned long) floor((src.bb() - kernel.bb()) / stride + 1.0 );
        const unsigned long dstZ = src.cc();
        Matrix *dst = new Matrix( dstX, dstY, dstZ );
        for( unsigned long dz = 0; dz < dstZ; dz++ ) {
            for( unsigned long dy = 0; dy < dstY; dy++ ) {
                const unsigned long sy = dy * stride;
                for( unsigned long dx = 0; dx < dstX; dx++ ) {
                    const unsigned long sx = dx * stride;
                    DNN_NUMERIC sum = 0.0;
                    for( unsigned long ky = 0; ky < kernalY; ky++ ) {
                        for( unsigned long kx = 0; kx < kernalX; kx++ ) {
                            const DNN_NUMERIC sourceValue = src.get( sx, sy, dz );
                            const DNN_NUMERIC kernalValue = kernel.get( kx, ky );
                            sum += sourceValue * kernalValue;
                        }
                    }
                    dst->set( dx, dy, dz, sum );
                }
            }
        }
        return dst;
    }
    
    Matrix*
    kernelOperationImage( const Matrix &src, const Matrix &kernel, const unsigned long stride ) {
        const unsigned long srcX    = src.bb();     // X
        const unsigned long srcY    = src.cc();     // Y
        const unsigned long srcZ    = src.aa();     // r,g,b channel
        const unsigned long kernalX = kernel.aa();  // X
        const unsigned long kernalY = kernel.bb();  // Y
        if( srcX < kernalX || srcY < kernalY ) {
            log_error( "Source matrix smaller than kernel" );
            return 0;
        }
        if( stride < 1 ) {
            log_error( "Stride too small" );
            return 0;
        }
        const unsigned long dstX = (unsigned long) floor((srcX - kernalX) / stride + 1.0 );
        const unsigned long dstY = (unsigned long) floor((srcY - kernalY) / stride + 1.0 );
        const unsigned long dstZ = srcZ;
        Matrix *dst = new Matrix( dstZ, dstX, dstY );
        for( unsigned long dy = 0; dy < dstY; dy++ ) {
            const unsigned long sy = dy * stride;
            for( unsigned long dx = 0; dx < dstX; dx++ ) {
                const unsigned long sx = dx * stride;
                for( unsigned long dz = 0; dz < dstZ; dz++ ) {  // R,G,B channel
                    DNN_NUMERIC sum = 0.0;
                    for( unsigned long ky = 0; ky < kernalY; ky++ ) {
                        for( unsigned long kx = 0; kx < kernalX; kx++ ) {
                            const DNN_NUMERIC sourceValue = src.get( dz, sx, sy );
                            const DNN_NUMERIC kernalValue = kernel.get( kx, ky );
                            sum += sourceValue * kernalValue;
                        }
                    }
                    dst->set( dz, dx, dy, sum );
                }
            }
        }
        return dst;
    }

    
    
}   // namespace tfs


