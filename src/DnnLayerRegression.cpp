// --------------------------------------------------------------------
//  DnnLayerRegression.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerRegression.h"

namespace tfs {
    
    static const char *NAME = "regression";

    const char*
    DnnLayerRegression::className( void ) {
        return NAME;
    }
    
    DnnLayerRegression::DnnLayerRegression( DnnLayer *previousLayer ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
    }
    
    DnnLayerRegression::~DnnLayerRegression( void ) {
        // Destructor
    }
    
}   // namespace tfs

