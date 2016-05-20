// --------------------------------------------------------------------
//  DnnLayerSoftmax.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cmath>
#include "DnnLayerSoftmax.h"
#include "Error.h"

namespace tfs {
    
    static const char *NAME = "softmax";

    const char*
    DnnLayerSoftmax::className( void ) {
        return NAME;
    }

    DnnLayerSoftmax::DnnLayerSoftmax( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( NAME, previousLayer ),
    m_es( 0 ) {
        // Constructor
        setup( trainable );
    }
    
    DnnLayerSoftmax::~DnnLayerSoftmax( void ) {
        // Destructor
        delete m_es;
        m_es = 0;
    }
    
    void
    DnnLayerSoftmax::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // S = size of input data
        // m_in_a[S]    Input activations
        // m_in_dw[S]
        // m_es[S]      Exponentials caculated in forward()
        // m_out_a[S]   Output activations
        // m_out_dw[S]
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 ) {
            log_error( "input A is null" );
            return;
        }
        const unsigned long classCount = m_in_a->size();    // 1d input, S elements.
        if( classCount < 1 ) {
            log_error( "input size < 1" );
            return;
        }
        m_es    = new Matrix( classCount, 1, 1 );
        m_out_a = new Matrix( classCount, 1, 1 );
        if( trainable ) {
            m_out_dw = new Matrix( classCount, 1, 1 );
        }
        return;
    }
    
    bool
    DnnLayerSoftmax::forward( void ) {
        // -----------------------------------------------------------------------------------
        // Forward propagate while training
        // S = size of input data
        // m_in_a[S]    Input activations
        // m_es[S]      Exponentials caculated in forward()
        // m_out_a[S]   Output activations
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_es == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *         input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const   inEnd = m_in_a->end();
              DNN_NUMERIC * const esStart = m_es->data();
              DNN_NUMERIC *        output = m_out_a->data();
        const DNN_NUMERIC * const  outEnd = m_out_a->end();
        
        const DNN_NUMERIC max = m_in_a->max();
              DNN_NUMERIC esum = 0.0;
        
        DNN_NUMERIC *es = esStart;
        while( input < inEnd ) {    // compute exponentials
            const DNN_NUMERIC ee = exp( *input++ - max );
            *es++ = ee;
            esum += ee;
        }
        if( esum == 0.0 ) {
            log_error( "esum == 0.0" );     // Avoid a divide by zero problem...
            esum = 0.0001;
        }
        es = esStart;
        while( output < outEnd ) {  // normalize
            *es /= esum;
            *output++ = *es++;
        }
        if( m_next_layer != 0 ) {
            return m_next_layer->forward();
        }
        return true;
    }
    
    DNN_NUMERIC
    DnnLayerSoftmax::backprop( const DMatrix &expectation ) {
        // -----------------------------------------------------------------------------------
        // Back propagate while training
        // S = size of input data
        // m_in_a[S]    Input activations
        // m_in_dw[S]
        // m_es[S]      Exponentials caculated in forward()
        // m_out_a[N]   Output activations
        // m_out_dw[N]
        // -----------------------------------------------------------------------------------
        if( expectation.isEmpty() || m_in_dw == 0 || m_es == 0 ) {
            log_error( "Not configured for training" );
            return 0.0;
        }
        const DNN_INTEGER yy = *(expectation.dataReadOnly());
        if( yy >= m_es->size()) {
            log_error( "Expectation is too large" );
            return 0.0;
        }
              DNN_NUMERIC *inputDw = m_in_dw->data();
        const DNN_INTEGER    count = m_in_dw->size();
        const DNN_NUMERIC      *es = m_es->data();
        
        DNN_NUMERIC delta;
        DNN_NUMERIC loss = 0.0;
        for( DNN_INTEGER ii = 0; ii < count; ii++ ) {
            if( ii == yy ) {
                loss  = log( *es );
                delta = *es++ - 1.0;
            } else {
                delta = *es++;
            }
            *inputDw++ = delta;
        }
        return loss;
    }
    
    bool
    DnnLayerSoftmax::predict( const Matrix &data ) {
        // -----------------------------------------------------------------------------------
        // Forward progagate when predicting
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_dw == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for predicting" );
        }
        if( m_w->size() != data.size()) {
            return log_error( "Input matrix does not match expected size" );
        }
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( data );
        }
        return true;
    }


    
}   // namespace tfs

