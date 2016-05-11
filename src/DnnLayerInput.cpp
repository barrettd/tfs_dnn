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
    
    DnnLayerInput::DnnLayerInput( unsigned long xx, unsigned long yy, unsigned long zz ) :
    DnnLayer( NAME ) {
        // Constructor
    }
        
    DnnLayerInput::~DnnLayerInput( void ) {
        // Destructor
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
        if( m_next_layer != 0 ) {
            return m_next_layer->forward( data );
        }
        return true;
    }
        
    bool
    DnnLayerInput::predict( const Matrix &data ) {
        // Forward progagate when predicting
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( data );
        }
        return true;
    }



}   // namespace tfs
