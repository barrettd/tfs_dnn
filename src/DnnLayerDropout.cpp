// --------------------------------------------------------------------
//  DnnLayerDropout.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "DnnLayerDropout.h"
#include "Error.h"
#include "Utility.h"

namespace tfs {
    
    DnnLayerDropout::DnnLayerDropout( DnnLayer *previousLayer, DNN_NUMERIC probability, const bool trainable ):
    DnnLayer( LAYER_DROPOUT, previousLayer ),
    m_probability( probability ),              // Drop out probability
    m_dropped( 0 ) {
        setup( trainable );
    }
    
    DnnLayerDropout::~DnnLayerDropout( void ) {
        delete[] m_dropped;
        m_dropped = 0;
    }
    
    DNN_NUMERIC
    DnnLayerDropout::probability( void ) const {
        return m_probability;
    }
    
    void
    DnnLayerDropout::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // S = size of input data
        // out_a[S]  = activations of each neuron
        // out_dw[S] = gradiant
        // -----------------------------------------------------------------------------------
        if( matrixBad( m_in_a )) {
            log_error( "Input activation matrix is bad" );
            return;
        }
        if( trainable && m_in_dw != 0 ) {       // Input layers can have null m_out_dw
            if( matrixBad( m_in_dw )) {
                log_error( "Input dw matrix is bad" );
                return;
            }
            if( m_in_a->count() != m_in_dw->count()) {  // By default, we expect the dimensions to be the same.
                log_error( "in_a != in_dw size" );
                return;
            }
        }
        m_out_a = new Matrix( *m_in_a );         // Output dimension matches input dimension
        if( trainable ) {
            m_out_dw = new Matrix( *m_out_a );   // Output dimension matches input dimension
        }
        const unsigned long count = m_out_a->count();
        m_dropped = new bool[ count ];
        for( unsigned long ii = 0; ii < count; ii++ ) {
            m_dropped[ii] = false;
        }
        return;
    }

    
    bool
    DnnLayerDropout::runPredict( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with predict()
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for predicting" );
        }
        const DNN_NUMERIC *src    = m_in_a->dataReadOnly();
        const DNN_NUMERIC *srcEnd = m_in_a->end();
              DNN_NUMERIC *dst    = m_out_a->data();
        const DNN_NUMERIC probability = m_probability;
        while( src < srcEnd ) {
            *dst++ = *src++ * probability;
        }
        return true;
    }
    
    bool
    DnnLayerDropout::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0 || m_dropped == 0 ) {
            return log_error( "Not configured for forward propagation" );
        }
        const DNN_NUMERIC *src    = m_in_a->dataReadOnly();
        const DNN_NUMERIC *srcEnd = m_in_a->end();
              DNN_NUMERIC *dst    = m_out_a->data();
        const DNN_NUMERIC probability = m_probability;
        bool *dropped = m_dropped;
        while( src < srcEnd ) {
            if( random() < probability ) {
                *dropped++ = true;
                *dst++ = 0.0;
                src++;
            } else {
                *dropped++ = false;
                *dst++ = *src++;
            }
        }
        return true;
    }
  
    bool
    DnnLayerDropout::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used both with backprop()
        // -----------------------------------------------------------------------------------
        if( m_in_dw == 0 || m_out_dw == 0 || m_dropped == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *          outDw = m_out_dw->dataReadOnly();
        const DNN_NUMERIC * const outDwEnd = m_out_dw->end();
              DNN_NUMERIC *           inDw = m_in_dw->data();
        const bool        *        dropped = m_dropped;
        while( outDw < outDwEnd ) {
            if( *dropped++ ) {
                *inDw++ = 0.0;
            } else {
                *inDw++ = *outDw;
            }
            outDw++;
        }
        return true;
    }

    
}   // namespace tfs
