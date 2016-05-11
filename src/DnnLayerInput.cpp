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
        m_x = xx;
        m_y = yy;
        m_z = zz;
        m_size = m_x * m_y * m_z;
    }
        
    DnnLayerInput::~DnnLayerInput( void ) {
        // Destructor
        m_w = 0;
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
    DnnLayerInput::forward( const DNN_NUMERIC *data, const unsigned long length ) {
        // Forward propagate while training
        if( data == 0 || length != m_size ) {
            return log_error( "Invalid parameters" );
        }
        m_w = const_cast <DNN_NUMERIC*>( data );
        if( m_next_layer != 0 ) {
            return m_next_layer->forward( data, length );
        }
        return true;
    }
        
    bool
    DnnLayerInput::predict( const DNN_NUMERIC *data, const unsigned long length ) {
        // Forward progagate when predicting
        if( data == 0 || length != m_size ) {
            return log_error( "Invalid parameters" );
        }
        m_w = const_cast <DNN_NUMERIC*>( data );
        if( m_next_layer != 0 ) {
            return m_next_layer->forward( data, length );
        }
        return true;
    }



}   // namespace tfs
