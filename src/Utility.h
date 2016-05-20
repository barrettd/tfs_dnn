// --------------------------------------------------------------------
//  Utility.hpp
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef Utility_hpp
#define Utility_hpp

#include "Constants.h"

namespace tfs {     // Tree Frog Software
    
    void randomSeed( const unsigned int seed );                                     // srand()
    
    DNN_NUMERIC random( const DNN_NUMERIC maxValue );                               // 0.0 to maxValue
    DNN_NUMERIC random( const DNN_NUMERIC minValue, const DNN_NUMERIC maxValue );   // minValue to maxValue
    
    inline DNN_NUMERIC random( const DNN_INTEGER maxValue ) { return random((const DNN_NUMERIC) maxValue ); }
    
    DNN_NUMERIC randomSigmoid( void );  //  0.0 to 1.0
    DNN_NUMERIC randomTanh(    void );  // -1.0 to 1.0

}   // namespace tfs

#endif /* Utility_hpp */
