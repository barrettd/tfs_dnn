// --------------------------------------------------------------------
//  DnnLayerMaxout.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerMaxout.h"
#include "Error.h"

namespace tfs {
    
    DnnLayerMaxout::DnnLayerMaxout( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_MAXOUT, previousLayer ) {
        // Constructor
    }
    
    DnnLayerMaxout::~DnnLayerMaxout( void ) {
        // Destructor
    }
    
    bool
    DnnLayerMaxout::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
    bool
    DnnLayerMaxout::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }

}   // namespace tfs

