// --------------------------------------------------------------------
//  Utility.cpp
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cmath>        // log()
#include <cstdlib>      // rand(), srand(), RAND_MAX
#include "Utility.h"

namespace tfs {     // Tree Frog Software
    
    // Random functions:
    // The thinking here is that by centeralizing the random functions, we can choose different implementations
    // in the future and change the implementation in just this file.  For example, if we want a predictable
    // distribution.
    
    void
    randomSeed( unsigned int seed ) {
        srand( seed );
    }

    DNN_NUMERIC
    random( void ) {    // 0.0 to 1.0
        return (DNN_NUMERIC) rand() / (RAND_MAX);
    }

    DNN_NUMERIC
    random( const DNN_NUMERIC maxValue ) {    // 0.0 to maxValue
        return ((DNN_NUMERIC) rand() / (RAND_MAX)) * maxValue;
    }
    
    DNN_NUMERIC
    random( const DNN_NUMERIC minValue, const DNN_NUMERIC maxValue ) {  // minValue to maxValue
        const DNN_NUMERIC difference = maxValue - minValue;
        return (((DNN_NUMERIC) rand() / (RAND_MAX)) * difference) + minValue;
        
    }
    
    DNN_NUMERIC
    randomSigmoid( void ) {  //  0.0 to 1.0
        return ((DNN_NUMERIC) rand() / (RAND_MAX));
    }
    
    DNN_NUMERIC
    randomTanh( void ) {  // -1.0 to 1.0
        return (((DNN_NUMERIC) rand() / (RAND_MAX)) * 2.0) - 1.0;
    }
    
    DNN_NUMERIC
    randomGauss( void ) {   // 0.0 to 1.0, gausian distribution.
        static bool return_v = false;
        static DNN_NUMERIC value = 0.0;
        if( return_v ) {
            return_v = false;
            return value;
        }
        const DNN_NUMERIC u = 2.0 * ((DNN_NUMERIC) rand() / (RAND_MAX)) -1.0;     // [ -1.0, 1.0 ]
        const DNN_NUMERIC v = 2.0 * ((DNN_NUMERIC) rand() / (RAND_MAX)) -1.0;
        const DNN_NUMERIC r = u*u + v*v;
        if( r == 0.0 || r > 1.0 ) {
            return randomGauss();
        }
        const DNN_NUMERIC c = sqrt( -2.0 * log( r ) / r );
        value = v * c;          // cache this
        return_v = true;
        return u * c;
    }


    
    
}   // namespace tfs
