// --------------------------------------------------------------------
//  DnnLayerTanh.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cmath>
#include "DnnLayerTanh.h"
#include "Error.h"

namespace tfs {
    
    static const char *NAME = "tanh";

    const char*
    DnnLayerTanh::className( void ) {
        return NAME;
    }
    
    DnnLayerTanh::DnnLayerTanh( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ) {
        // Constructor
        if( previousLayer != 0 ) {  // previousLayer should not be null.
            setup( trainable );
        } else {
            log_error( "previousLayer is null" );
        }
    }
    
    DnnLayerTanh::~DnnLayerTanh( void ) {
        // Destructor
    }
    
    bool
    DnnLayerTanh::forward( void ) {
        // -----------------------------------------------------------------------------------
        // Forward propagate while training
        // S = size of input data
        // m_in_a[S]
        // m_out_a[S]
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *        input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const  inEnd = m_in_a->end();     // A pointer just past the end of the input
        DNN_NUMERIC       *       output = m_out_a->data();   // activations[S] for the neurons in this layer
        
        while( input < inEnd ) {
            *output++ = tanh( *input++ );
        }
        
        if( m_next_layer != 0 ) {
            return m_next_layer->forward();
        }
        return true;
    }
    
    bool
    DnnLayerTanh::backprop( void ) {
        // -----------------------------------------------------------------------------------
        // Back propagate while training
        // S = size of input data
        // m_in_a[S]
        // m_in_dw[S]
        // m_out_a[S]
        // m_out_dw[S]
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_in_dw == 0 || m_w == 0 || m_dw == 0 || m_out_a == 0 || m_out_dw == 0 ) {
            return log_error( "Not configured for training" );
        }
              DNN_NUMERIC *        inputDw = m_in_dw->data();
        const DNN_NUMERIC * const  inDwEnd = m_in_dw->end();             // A pointer just past the end of the input
        const DNN_NUMERIC *        output  = m_out_a->dataReadOnly();
        const DNN_NUMERIC *          outDw = m_out_dw->dataReadOnly();
        
        
        while( inputDw < inDwEnd ) {
            const DNN_NUMERIC out = *output++;
            *inputDw++ = (1.0 - out * out) * *outDw++;
        }
        
        if( m_prev_layer != 0 ) {
            return m_prev_layer->backprop();
        }
        return true;
    }
    
    bool
    DnnLayerTanh::predict( const Matrix &data ) {
        // -----------------------------------------------------------------------------------
        // Forward progagate when predicting
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_dw == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for predicting" );
        }
        if( m_w->count() != data.count()) {
            return log_error( "Input matrix does not match expected size" );
        }
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( data );
        }
        return true;
    }


}   // namespace tfs

