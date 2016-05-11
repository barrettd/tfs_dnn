// --------------------------------------------------------------------
//  DnnLayerFullyConnected.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerFullyConnected.h"

namespace tfs {
    
    static const char *NAME = "fc";

    const char*
    DnnLayerFullyConnected::className( void ) {
        return NAME;
    }

    DnnLayerFullyConnected::DnnLayerFullyConnected( DnnLayer *previousLayer ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
        
    DnnLayerFullyConnected::~DnnLayerFullyConnected( void ) {
        // Destructor
    }
    
    
}   // namespace tfs
