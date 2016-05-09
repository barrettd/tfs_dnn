// --------------------------------------------------------------------
//  DnnLayer.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "DnnLayer.h"

namespace tfs {
    
    DnnLayer::DnnLayer( void ):
    m_x( 0 ), m_y( 0 ), m_z( 0 ) {
        // Constructor
    }
    
    DnnLayer::DnnLayer( unsigned long xx, unsigned long yy, unsigned long zz ) :
    m_x( xx ), m_y( yy ), m_z( zz ) {
        // Constructor
    }
    
    DnnLayer::~DnnLayer( void ) {
        // Destructor
    }
    
    void
    DnnLayer::randomize( void ) { // Randomize weights and bias.
        return;
    }

    bool
    DnnLayer::forward(  void ) {  // Forward propagate while training
        return true;
    }

    bool
    DnnLayer::backprop( void ) {  // Back propagate while training
        return true;
    }

    bool
    DnnLayer::predict(  void ) {  // Forward progagate when predicting
        return true;
    }


    
}   // namespace tfs
