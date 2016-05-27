// --------------------------------------------------------------------
//  DnnLayerPool.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerPool.h"
#include "Error.h"

namespace tfs {
    
    static const char *NAME = "pool";

    const char*
    DnnLayerPool::className( void ) {
        return NAME;
    }

    DnnLayerPool::DnnLayerPool( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
    
    DnnLayerPool::~DnnLayerPool( void ) {
        // Destructor
    }
    
    bool
    DnnLayerPool::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        return true;
    }
    
    bool
    DnnLayerPool::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
        return true;
    }
    

}   // namespace tfs

