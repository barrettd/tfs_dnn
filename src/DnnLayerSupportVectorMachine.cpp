// --------------------------------------------------------------------
//  DnnLayerSupportVectorMachine.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerSupportVectorMachine.h"
#include "Error.h"

namespace tfs {
    
    DnnLayerSupportVectorMachine::DnnLayerSupportVectorMachine( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_SUPPORT_VECTOR_MACHINE, previousLayer ) {
        // Constructor
    }
    
    DnnLayerSupportVectorMachine::~DnnLayerSupportVectorMachine( void ) {
        // Destructor
    }
    
    bool
    DnnLayerSupportVectorMachine::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
    bool
    DnnLayerSupportVectorMachine::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }

    
}   // namespace tfs

