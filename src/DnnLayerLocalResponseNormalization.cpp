// --------------------------------------------------------------------
//  DnnLayerLocalResponseNormalization.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerLocalResponseNormalization.h"

namespace tfs {
    
    static const char *NAME = "lrn";
 
    const char*
    DnnLayerLocalResponseNormalization::className( void ) {
        return NAME;
    }

    DnnLayerLocalResponseNormalization::DnnLayerLocalResponseNormalization( DnnLayer *previousLayer ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
    
    DnnLayerLocalResponseNormalization::~DnnLayerLocalResponseNormalization( void ) {
        // Destructor
    }
    
}   // namespace tfs
