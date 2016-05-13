// --------------------------------------------------------------------
//  DnnLayerDropout.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "DnnLayerDropout.h"
#include "Error.h"

namespace tfs {
    
    static const char *NAME = "dropout";

    const char*
    DnnLayerDropout::className( void ) {
        return NAME;
    }

    DnnLayerDropout::DnnLayerDropout( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
    
    DnnLayerDropout::~DnnLayerDropout( void ) {
        // Destructor
    }
    
}   // namespace tfs
