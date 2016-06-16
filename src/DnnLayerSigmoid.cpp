// --------------------------------------------------------------------
//  DnnLayerSigmoid.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerSigmoid.h"
#include "Error.h"

namespace tfs {
    
    DnnLayerSigmoid::DnnLayerSigmoid( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_SIGMOID, previousLayer ) {
        // Constructor
    }
    
    DnnLayerSigmoid::~DnnLayerSigmoid( void ) {
        // Destructor
    }
    
    bool
    DnnLayerSigmoid::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
    bool
    DnnLayerSigmoid::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }

}   // namespace tfs

