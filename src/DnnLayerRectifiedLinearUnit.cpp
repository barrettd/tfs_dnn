// --------------------------------------------------------------------
//  DnnLayerRectifiedLinearUnit.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerRectifiedLinearUnit.h"
#include "Error.h"

namespace tfs {
    
    static const char *NAME = "relu";

    const char*
    DnnLayerRectifiedLinearUnit::className( void ) {
        return NAME;
    }

    DnnLayerRectifiedLinearUnit::DnnLayerRectifiedLinearUnit( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
        setup( trainable );
    }
    
    DnnLayerRectifiedLinearUnit::~DnnLayerRectifiedLinearUnit( void ) {
        // Destructor
    }
    
    inline bool
    threshold( DNN_NUMERIC *dst, const DNN_NUMERIC *src, const DNN_NUMERIC * const end ) {
        // -----------------------------------------------------------------------------------
        // Copy src to dst, except clamping negative values to 0.0
        // -----------------------------------------------------------------------------------
        if( dst == 0 || src == 0 || end == 0 ) {
            return log_error( "Not configured" );
        }
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
        return threshold( dst, src, end );
    }
    
    bool
    DnnLayerRectifiedLinearUnit::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // Backpropagate output dw to input dw, except that negative values are clamped to zero.
        // ok 11 June 2016
        // -----------------------------------------------------------------------------------
        if( m_in_dw == 0 || m_out_dw == 0  ) {
            return log_error( "Not configured" );
        }
        const DNN_NUMERIC *       src = m_out_dw->dataReadOnly();
        const DNN_NUMERIC * const end = m_out_dw->end();
              DNN_NUMERIC *       dst = m_in_dw->data();
        return threshold( dst, src, end );
    }
    
}   // namespace tfs

