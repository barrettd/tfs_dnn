// --------------------------------------------------------------------
//  Utility.cpp
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cmath>        // log()
#include <cstdlib>      // rand(), srand(), RAND_MAX
#include <limits>
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
        return (DNN_NUMERIC) rand() / (DNN_NUMERIC)(RAND_MAX);
    }

    DNN_NUMERIC
    random( const DNN_NUMERIC maxValue ) {    // 0.0 to maxValue
        return (((DNN_NUMERIC) rand() * maxValue ) / (DNN_NUMERIC)(RAND_MAX));
    }
    
    DNN_NUMERIC
    random( const DNN_NUMERIC minValue, const DNN_NUMERIC maxValue ) {  // minValue to maxValue
        const DNN_NUMERIC difference = maxValue - minValue;
        return (((DNN_NUMERIC) rand() / (DNN_NUMERIC)(RAND_MAX)) * difference) + minValue;
    }
    
    DNN_INTEGER
    randomInt( void ) {
        return (DNN_INTEGER) rand() / (DNN_INTEGER)(RAND_MAX);
    }
    
    DNN_INTEGER
    randomInt( const DNN_INTEGER maxValue ) {
        return (((DNN_INTEGER) rand() * maxValue ) / (DNN_INTEGER)(RAND_MAX));
    }
    
    DNN_INTEGER
    randomInt( const DNN_INTEGER minValue, const DNN_INTEGER maxValue ) {
        const DNN_INTEGER difference = maxValue - minValue;
        return ((((DNN_INTEGER) rand() * difference ) / (DNN_INTEGER)(RAND_MAX))) + minValue;
    }

    DNN_NUMERIC
    randomSigmoid( void ) {  //  0.0 to 1.0
        return ((DNN_NUMERIC) rand() / (DNN_NUMERIC)(RAND_MAX));
    }
    
    DNN_NUMERIC
    randomTanh( void ) {  // -1.0 to 1.0
        return (((DNN_NUMERIC) rand() / (DNN_NUMERIC)(RAND_MAX)) * 2.0) - 1.0;
    }
    
    DNN_NUMERIC
    randomGauss( void ) {
        // --------------------------------------------------------------------
        // 0.0 to 1.0, gausian distribution. Box–Muller transform
        // This form probably a little faster than randomGaussian() below.
        // --------------------------------------------------------------------
        const DNN_NUMERIC scale = 2.0 / (DNN_NUMERIC) RAND_MAX;
        
        static DNN_NUMERIC z1 = 0.0;
        static bool return_z1 = false;
        if( return_z1 ) {
            return_z1 = false;
            return z1;          // Return z1 (prevously calculated)
        }
        return_z1 = true;
        DNN_NUMERIC u, v, r;
        do {
            u = ((DNN_NUMERIC) rand() * scale) -1.0;     // [ -1.0, 1.0 ]
            v = ((DNN_NUMERIC) rand() * scale) -1.0;
            r = u*u + v*v;
        } while( r == 0.0 || r > 1.0 );

        const DNN_NUMERIC c = sqrt( -2.0 * log( r ) / r );
        z1 = v * c;             // Remember z1 for next time.
        return u * c;           // Return z0
    }

    DNN_NUMERIC
    randomGaussian( DNN_NUMERIC mu, DNN_NUMERIC sigma ) {
        // --------------------------------------------------------------------
        // From: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        // --------------------------------------------------------------------
        const DNN_NUMERIC epsilon = std::numeric_limits<double>::min();
        const DNN_NUMERIC two_pi  = 2.0 * M_PI;
        const DNN_NUMERIC scale   = 1.0 / (DNN_NUMERIC) RAND_MAX;
        
        static DNN_NUMERIC z1 = 0.0;
        static bool generate  = false;
        
        generate = !generate;
        
        if( !generate ) {
            return z1 * sigma + mu;
        }
        
        DNN_NUMERIC u1, u2;
        do {
            u1 = rand() * scale;    // [0.0,1.0]
            u2 = rand() * scale;
        } while ( u1 <= epsilon );
        
        const DNN_NUMERIC pu = two_pi * u2;
        const DNN_NUMERIC cc = sqrt(-2.0 * log(u1));
        const DNN_NUMERIC z0 = cc * cos( pu );
                          z1 = cc * sin( pu );
        return z0 * sigma + mu;
    }

    
    
}   // namespace tfs
