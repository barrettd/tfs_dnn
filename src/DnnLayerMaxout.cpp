// --------------------------------------------------------------------
//  DnnLayerMaxout.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerMaxout.h"

namespace tfs {
    
    static const char *NAME = "maxout";

    const char*
    DnnLayerMaxout::className( void ) {
        return NAME;
    }

    DnnLayerMaxout::DnnLayerMaxout( void ):
    DnnLayer( NAME ) {
        // Constructor
    }
    
    DnnLayerMaxout::~DnnLayerMaxout( void ) {
        // Destructor
    }
    
}   // namespace tfs

