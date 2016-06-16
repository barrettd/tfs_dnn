// --------------------------------------------------------------------
//  DnnLayerLocalResponseNormalization.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerLocalResponseNormalization.h"
#include "Error.h"

namespace tfs {
    
    DnnLayerLocalResponseNormalization::DnnLayerLocalResponseNormalization( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_LOCAL_RESPONSE_NORMALIZATION, previousLayer ) {
        // Constructor
    }
    
    DnnLayerLocalResponseNormalization::~DnnLayerLocalResponseNormalization( void ) {
        // Destructor
    }
    
    bool
    DnnLayerLocalResponseNormalization::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
    bool
    DnnLayerLocalResponseNormalization::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }

    
}   // namespace tfs
