// --------------------------------------------------------------------
//  DnnLayerSoftmax.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerSoftmax.h"

namespace tfs {
    
    static const char *NAME = "softmax";

    const char*
    DnnLayerSoftmax::className( void ) {
        return NAME;
    }

    DnnLayerSoftmax::DnnLayerSoftmax( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
    
    DnnLayerSoftmax::~DnnLayerSoftmax( void ) {
        // Destructor
    }
    
}   // namespace tfs

