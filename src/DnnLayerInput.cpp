// --------------------------------------------------------------------
//  DnnLayerInput.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerInput.h"

namespace tfs {

    static const char *NAME = "input";
    
    const char*
    DnnLayerInput::className( void ) {
        return NAME;
    }
    
    DnnLayerInput::DnnLayerInput( void ):
    DnnLayer( NAME ) {
        // Constructor
    }
    
    DnnLayerInput::DnnLayerInput( unsigned long xx, unsigned long yy, unsigned long zz ) :
    DnnLayer( NAME, xx, yy, zz ) {
        // Constructor
    }
    
    DnnLayerInput::~DnnLayerInput( void ) {
        // Destructor
    }


}   // namespace tfs
