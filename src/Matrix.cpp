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
    
}   // namespace tfs


