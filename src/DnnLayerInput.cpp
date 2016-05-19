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
    DnnLayer( NAME ),
    m_x( xx ), m_y( yy ), m_z( zz ), m_size( xx * yy * zz ) {
        // Constructor
        if( trainable ) {
            m_dw = new Matrix( xx, yy, zz );    // dw is ignored for input, but here to satisfy lower layer write.
        }
    }
        
    DnnLayerInput::~DnnLayerInput( void ) {
        // Destructor
    }
    
    unsigned long
    DnnLayerInput::aX( void ) const {
        return m_x;
    }
    
    unsigned long
    DnnLayerInput::aY( void ) const {
        return m_y;
    }
    
    unsigned long
    DnnLayerInput::aZ( void ) const {
        return m_z;
    }

    unsigned long
    DnnLayerInput::aSize( void ) const {
        return m_size;
    }

    void
    DnnLayerInput::initialize( void ) {
        // Do not intialize the input.
        if( m_next_layer != 0 ) {
            m_next_layer->initialize();
        }
        return;
    }

    void
    DnnLayerInput::randomize( void ) {
        // Do not randomize the input.
        if( m_next_layer != 0 ) {
            m_next_layer->randomize();
        }
        return;
    }
    
    bool
    DnnLayerInput::forward( const Matrix &data ) {
        // Forward propagate while training
        // Generally all of the layers are configured to accept input from the previous layer activation.
        // The input layer needs to pass the ever changing data as input to the next layer.
        if( m_size != data.size()) {
            return log_error( "Input matrix does not match expected size" );
        }
        return DnnLayer::forward( data );   // Base class has a pass-through implementation that we can use.
    }
        
    bool
    DnnLayerInput::predict( const Matrix &data ) {
        // Forward progagate when predicting
        if( m_size != data.size()) {
            return log_error( "Input matrix does not match expected size" );
        }
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( data );
        }
        return true;
    }



}   // namespace tfs
