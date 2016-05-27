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
    
    DnnLayerInput::DnnLayerInput( unsigned long xx, unsigned long yy, unsigned long zz, const bool trainable, const bool retain_dw  ) :
    DnnLayer( NAME ) {
        // Constructor
        m_out_a = new Matrix( xx, yy, zz );         // Input data copied to here.
        if( retain_dw ) {
            m_out_dw = new Matrix( xx, yy, zz );    // dw is generally unused here. Lower layers need to check for dw == 0.
        }
    }
        
    DnnLayerInput::~DnnLayerInput( void ) {
        // Destructor
    }


}   // namespace tfs
