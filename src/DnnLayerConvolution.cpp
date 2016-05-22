// --------------------------------------------------------------------
//  DnnLayerConvolution.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerConvolution.h"
#include "Error.h"

namespace tfs {
  
    static const char *NAME = "conv";

    const char*
    DnnLayerConvolution::className( void ) {
        return NAME;
    }

    DnnLayerConvolution::DnnLayerConvolution( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
        m_l1_decay_mul = 0.0;
        m_l2_decay_mul = 1.0;
    }
    
    DnnLayerConvolution::~DnnLayerConvolution( void ) {
        // Destructor
    }
    
    
}   // namespace tfs