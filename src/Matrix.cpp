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
        if( count == 0 ) {
            return log_error( "Matrix empty" );
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
        // ---------------------------------------------------------------------------------
        // This is a zero pad reducing kernel function. Useful for sub-sampling
        // ---------------------------------------------------------------------------------
        const unsigned long srcX      = src.width();      // X
        const unsigned long srcY      = src.height();     // Y
        const unsigned long srcZ      = src.cc();         // r,g,b channel (if image)
        const unsigned long kernelX   = kernel.aa();
        const unsigned long kernelY   = kernel.bb();
        const DNN_NUMERIC *kernelData = kernel.dataReadOnly();
        
        if( srcX < kernelX || srcY < kernelY ) {
            log_error( "Source matrix smaller than kernel" );
            return 0;
        }
        if( stride < 1 ) {
            log_error( "Stride too small" );
            return 0;
        }
        const unsigned long dstX = (unsigned long) floor((srcX - kernelX) / stride + 1.0 );
        const unsigned long dstY = (unsigned long) floor((srcY - kernelY) / stride + 1.0 );
        const unsigned long dstZ = srcZ;
        Matrix      *dst = new Matrix( dstX, dstY, dstZ );
        DNN_NUMERIC *ptr = dst->data();
        for( unsigned long dz = 0; dz < dstZ; dz++ ) {
            for( unsigned long dy = 0; dy < dstY; dy++ ) {
                const unsigned long sy = dy * stride;
                for( unsigned long dx = 0; dx < dstX; dx++ ) {
                    const unsigned long sx = dx * stride;
                    const DNN_NUMERIC *kernelPtr = kernelData;
                    DNN_NUMERIC sum = 0.0;
                    for( unsigned long iy = sy; iy < (kernelY + sy); iy++ ) {
                        for( unsigned long ix = sx; ix < (kernelX + sx); ix++ ) {
                            const DNN_NUMERIC sourceValue = src.get( ix, iy, dz ); 
                            sum += sourceValue * *kernelPtr++;
                        }
                    }
                    *ptr++ = sum;       //  dst->set( dx, dy, dz, sum );
                }
            }
        }
        return dst;
    }
    
    bool
    subsample( Matrix &dstMatrix, const Matrix &srcMatrix, const unsigned long da, const unsigned long db ) {
        const unsigned long srcA  = srcMatrix.aa();     // X
        const unsigned long srcB  = srcMatrix.bb();     // Y
        const unsigned long srcC  = srcMatrix.cc();     // r,g,b channel (if image)
        const unsigned long srcD  = srcMatrix.dd();
        const unsigned long dstA  = dstMatrix.aa();
        const unsigned long dstB  = dstMatrix.bb();
        const unsigned long dstC  = dstMatrix.cc();
        const unsigned long dstD  = dstMatrix.dd();
        if(( dstA + da ) > srcA || (dstB + db ) > srcB || dstC != srcC || srcD != dstD ) {
            return log_error( "Matricies not compatible for subsample." );
        }
        const unsigned long maxB = dstB + db;
        const DNN_NUMERIC  *data = srcMatrix.dataReadOnly();
              DNN_NUMERIC  *dst  = dstMatrix.data();
        if( dstD == 1 ) {
            if( dstC == 1 ) {
                // 2D matrix( aa, bb )
                for( unsigned long bb = db; bb < maxB; bb++ ) {
                    const unsigned long srcIndex = srcMatrix.getIndex( da, bb );    // TODO: Speed up subsample
                    const DNN_NUMERIC *src = &data[srcIndex];
                    for( unsigned long aa = 0; aa < dstA; aa++ ) {
                        *dst++ = *src++;
                    }
                }
                return true;
            }
            // 3D matrix( aa, bb, cc )
            for( unsigned long cc = 0; cc < dstC; cc++ ) {
                for( unsigned long bb = db; bb < maxB; bb++ ) {
                    const unsigned long srcIndex = srcMatrix.getIndex( da, bb, cc );
                    const DNN_NUMERIC *src = &data[srcIndex];
                    for( unsigned long aa = 0; aa < dstA; aa++ ) {
                        *dst++ = *src++;
                    }
                }
            }
            return true;
        }
        // 4D matrix( aa, bb, cc, dd )
        for( unsigned long dd = 0; dd < dstD; dd++ ) {
            for( unsigned long cc = 0; cc < dstC; cc++ ) {
                for( unsigned long bb = db; bb < maxB; bb++ ) {
                    const unsigned long srcIndex = srcMatrix.getIndex( da, bb, cc, dd );
                    const DNN_NUMERIC *src = &data[srcIndex];
                    for( unsigned long aa = 0; aa < dstA; aa++ ) {
                        *dst++ = *src++;
                    }
                }
            }
        }
        return true;
    }


}   // namespace tfs


