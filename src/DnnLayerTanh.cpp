// --------------------------------------------------------------------
//  DnnLayerTanh.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerTanh.h"

namespace tfs {
    
    static const char *NAME = "tanh";

    const char*
    DnnLayerTanh::className( void ) {
        return NAME;
    }

    DnnLayerTanh::DnnLayerTanh( DnnLayer *previousLayer ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
    
    DnnLayerTanh::~DnnLayerTanh( void ) {
        // Destructor
    }
    
}   // namespace tfs

