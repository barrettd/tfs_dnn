// --------------------------------------------------------------------
//  DnnLayerRectifiedLinearUnit.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerRectifiedLinearUnit.h"

namespace tfs {
    
    static const char *NAME = "relu";

    const char*
    DnnLayerRectifiedLinearUnit::className( void ) {
        return NAME;
    }

    DnnLayerRectifiedLinearUnit::DnnLayerRectifiedLinearUnit( void ):
    DnnLayer( NAME ) {
        // Constructor
    }
    
    DnnLayerRectifiedLinearUnit::~DnnLayerRectifiedLinearUnit( void ) {
        // Destructor
    }
    
}   // namespace tfs

