// --------------------------------------------------------------------
//  DnnLayerInput.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerInput.h"
#include "Error.h"

namespace tfs {

    static const char *NAME = "input";
    
    const char*
    DnnLayerInput::className( void ) {
        return NAME;
    }
    
    DnnLayerInput::DnnLayerInput( unsigned long xx, unsigned long yy, unsigned long zz, const bool trainable ) :
    DnnLayer( NAME ) {
        // Constructor
        m_out_a = new Matrix( xx, yy, zz );         // Input data copied to here.
        if( trainable ) {
            m_out_dw = new Matrix( xx, yy, zz );    // dw is ignored for this layer, but here to satisfy lower layer backpropagate.
        }
    }
        
    DnnLayerInput::~DnnLayerInput( void ) {
        // Destructor
    }


}   // namespace tfs
