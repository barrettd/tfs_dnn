// --------------------------------------------------------------------
//  DnnLayerPool.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerPool.h"

namespace tfs {
    
    static const char *NAME = "pool";

    const char*
    DnnLayerPool::className( void ) {
        return NAME;
    }

    DnnLayerPool::DnnLayerPool( void ):
    DnnLayer( NAME ) {
        // Constructor
    }
    
    DnnLayerPool::~DnnLayerPool( void ) {
        // Destructor
    }
    
}   // namespace tfs

