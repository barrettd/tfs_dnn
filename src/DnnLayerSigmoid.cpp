// --------------------------------------------------------------------
//  DnnLayerSigmoid.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerSigmoid.h"
#include "Error.h"

namespace tfs {
    
    static const char *NAME = "sigmoid";

    const char*
    DnnLayerSigmoid::className( void ) {
        return NAME;
    }

    DnnLayerSigmoid::DnnLayerSigmoid( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
    
    DnnLayerSigmoid::~DnnLayerSigmoid( void ) {
        // Destructor
    }
    
}   // namespace tfs

