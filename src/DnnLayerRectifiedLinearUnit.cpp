// --------------------------------------------------------------------
//  DnnLayerRectifiedLinearUnit.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerRectifiedLinearUnit.hpp"
#include "Error.hpp"

namespace tfs {
    
    DnnLayerRectifiedLinearUnit::DnnLayerRectifiedLinearUnit( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_RECTIFIED_LINEAR_UNIT, previousLayer ) {
        // Constructor
        setup( trainable );
    }
    
    DnnLayerRectifiedLinearUnit::~DnnLayerRectifiedLinearUnit( void ) {
        // Destructor
    }
    
    bool
    DnnLayerRectifiedLinearUnit::runForward( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // The output is a copy of the input, except that any negative values are clamped at zero.
        // ok 11 June 2016
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0  ) {
            return log_error( "Not configured" );
        }
        const DNN_NUMERIC *       src = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const end = m_in_a->end();
              DNN_NUMERIC *       dst = m_out_a->data();
        
        while( src < end ) {
            if( *src > 0.0 ) {      // Threshold at 0.0
                *dst++ = *src++;
            } else {
                *dst++ = 0.0;
                src++;
            }
        }
        return true;
    }
    
    bool
    DnnLayerRectifiedLinearUnit::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // Backpropagate output dw to input dw, except that negative values are clamped to zero.
        // ok 14 June 2016
        // -----------------------------------------------------------------------------------
        if( m_in_dw == 0 || m_out_dw == 0  ) {
            return log_error( "Not configured" );
        }
        const DNN_NUMERIC *       val = m_out_a->dataReadOnly();
        const DNN_NUMERIC *       src = m_out_dw->dataReadOnly();
        const DNN_NUMERIC * const end = m_out_dw->end();
              DNN_NUMERIC *       dst = m_in_dw->data();

        while( src < end ) {
            if( *val++ <= 0.0 ) {   // Use the output value to determine thesholding
                *dst++ = 0.0;       // Input gradient = 0.0;
                src++;
            } else {
                *dst++ = *src++;    // Input gradient = output gradient.
            }
        }
        return true;
    }
    
}   // namespace tfs

