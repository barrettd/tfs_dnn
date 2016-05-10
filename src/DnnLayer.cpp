// --------------------------------------------------------------------
//  DnnLayer.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "DnnLayer.h"

namespace tfs {
    
    DnnLayer::DnnLayer( const char *name ):
    m_name( name ),
    m_in_x(  0 ), m_in_y(  0 ), m_in_z(  0 ),
    m_out_x( 0 ), m_out_y( 0 ), m_out_z( 0 ) {
        // Constructor
    }
    
    DnnLayer::DnnLayer( const char *name, unsigned long xx, unsigned long yy, unsigned long zz ) :
    m_name( name ),
    m_in_x(  xx ), m_in_y(  yy ), m_in_z(  zz ),
    m_out_x( xx ), m_out_y( yy ), m_out_z( zz ) {
        // Constructor
    }
    
    DnnLayer::~DnnLayer( void ) {
        // Destructor
    }
    
    const char*
    DnnLayer::getName( void ) const {
        return m_name;
    }
    
    const char*
    DnnLayer::setName( const char *name ) {
        return m_name = name;
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
