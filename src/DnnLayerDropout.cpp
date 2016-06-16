// --------------------------------------------------------------------
//  DnnLayerDropout.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "DnnLayerDropout.h"
#include "Error.h"

namespace tfs {
    
    DnnLayerDropout::DnnLayerDropout( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_DROPOUT, previousLayer ) {
        // Constructor
        // TODO:
    }
    
    DnnLayerDropout::~DnnLayerDropout( void ) {
        // Destructor
    }
    
    bool
    DnnLayerDropout::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
    bool
    DnnLayerDropout::runPredict(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with predict()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
    bool
    DnnLayerDropout::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used both with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }

    
}   // namespace tfs
