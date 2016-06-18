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
    
    DnnLayerSoftmax::DnnLayerSoftmax( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_SOFTMAX, previousLayer ),
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
        if( matrixBad( m_in_a )) {
            log_error( "input A is bad" );
            return;
        }
        const unsigned long classCount = m_in_a->count();    // 1d input, S elements.
        if( classCount < 1 ) {
            log_error( "input size < 1" );
            return;
        }
        m_es    = new Matrix( classCount );
        m_out_a = new Matrix( *m_es );
        if( trainable ) {
            m_out_dw = new Matrix( *m_out_a );
        }
        return;
    }
    
    Matrix*
    DnnLayerSoftmax::exponentialsReadOnly( void ) const {
        return m_es;
    }
    Matrix*
    DnnLayerSoftmax::exponentials( void ) {
        return m_es;
    }
    
    bool
    DnnLayerSoftmax::runForward( void ) {
        // -----------------------------------------------------------------------------------
        // Forward propagate while training
        // S = size of input data
        // m_in_a[S]    Input activations
        // m_es[S]      Exponentials caculated in forward()
        // m_out_a[S]   Output activations
        // ok: 24 May 2016, 11 June 2016
        // -----------------------------------------------------------------------------------
        if( matrixBad( m_in_a ) || matrixBad( m_es ) || matrixBad( m_out_a )) {
            return log_error( "Not configured" );
        }
        const DNN_NUMERIC *         input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const   inEnd = m_in_a->end();
              DNN_NUMERIC * const esStart = m_es->data();
              DNN_NUMERIC *        output = m_out_a->data();
        const DNN_NUMERIC * const  outEnd = m_out_a->end();
        
        const DNN_NUMERIC max  = m_in_a->max();     // Find input activation maximum
              DNN_NUMERIC esum = 0.0;
        
        DNN_NUMERIC *es = esStart;
        while( input < inEnd ) {                    // Compute exponentials
            const DNN_NUMERIC ee = exp( *input++ - max );
            *es++ = ee;
            esum += ee;
        }
        if( esum == 0.0 ) {                         // Avoid a divide by zero problem...
            log_error( "esum == 0.0 - divide by zero problem" );
            esum = 0.0001;
        }
        es = esStart;
        while( output < outEnd ) {                  // normalize
            *output++ = *es++ /= esum;              // sum( m_output_a ) == 1.0 - 12 June 2016 verified
        }
        return true;
    }
    
    DNN_NUMERIC
    DnnLayerSoftmax::runBackprop( const DNN_INTEGER yy ) {
        // -----------------------------------------------------------------------------------
        // Back propagate while training
        // S = size of input data
        // m_in_a[S]    Input activations
        // m_in_dw[S]
        // m_es[S]      Exponentials caculated in forward()
        // m_out_a[N]   Output activations
        // m_out_dw[N]
        // ok: 24 May 2016, 11 June 2016
        // -----------------------------------------------------------------------------------
        if( m_in_dw == 0 || m_es == 0 ) {
            log_error( "Not configured for training" );
            return 0.0;
        }
        if( yy < 0 || yy >= (DNN_INTEGER) m_es->count()) {
            log_error( "Expectation is out of range" );
            return 0.0;
        }
              DNN_NUMERIC *inputDw = m_in_dw->data();
        const DNN_INTEGER    count = (DNN_INTEGER) m_in_dw->count();
        const DNN_NUMERIC      *es = m_es->data();
        
        DNN_NUMERIC loss = 0.0;
        for( DNN_INTEGER ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC esValue = *es++;
            if( ii == yy ) {
                *inputDw++ = esValue - 1.0;
                loss = -log( esValue );
            } else {
                *inputDw++ = esValue;
            }
        }
        return loss;
    }
    
}   // namespace tfs

