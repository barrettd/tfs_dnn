// --------------------------------------------------------------------
//  DnnLayerInput.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerInput.h"

namespace tfs {


    DnnLayerInput::DnnLayerInput( void ) {
        // Constructor
    }
    
    DnnLayerInput::DnnLayerInput( unsigned long xx, unsigned long yy, unsigned long zz ) :
    DnnLayer( xx, yy, zz ) {
        // Constructor
    }
    
    DnnLayerInput::~DnnLayerInput( void ) {
        // Destructor
    }


}   // namespace tfs
