// --------------------------------------------------------------------
//  DnnLayerFullyConnected.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerFullyConnected.h"
#include "Error.h"

namespace tfs {
    
    static const char *NAME = "fc";

    const char*
    DnnLayerFullyConnected::className( void ) {
        return NAME;
    }

    DnnLayerFullyConnected::DnnLayerFullyConnected( DnnLayer *previousLayer, unsigned long neuronCount, const bool trainable ):
    DnnLayer( NAME, previousLayer ),
    m_neuron_count( neuronCount ) {         // Constructor
        m_l1_decay_mul = 0.0;
        m_l2_decay_mul = 1.0;
        if( previousLayer != 0 ) {  // previousLayer should not be null.
            setup( trainable );
        } else {
            log_error( "previousLayer is null" );
        }
    }
    
    DnnLayerFullyConnected::~DnnLayerFullyConnected( void ) {
        // Destructor
    }
    
    void
    DnnLayerFullyConnected::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // N = number of neurons
        // S = size of input data
        //  w[N,S+1] = neuron weight + bias weight
        // dw[N,S+1] = gradiant + d/dw bias
        // out_a[N]  = activations of each neuron
        // out_dw[N] = gradiant
        // -----------------------------------------------------------------------------------
        if( m_neuron_count < 1 ) {
            log_error( "Neuron count is less than 1" );
            return;
        }
        if( m_in_a == 0 ) {
            log_error( "Input activation matrix is null" );
            return;
        }
        const unsigned long N = m_neuron_count;
        const unsigned long S = m_in_a->count(); // 1d input, S elements.

        m_w = new Matrix( N, S+1, 1 );          // 2d neuron weights N x (S+1)
        if( trainable ) {
            m_dw = new Matrix( N, S+1, 1 );     // 2d neuron weights N x (S+1)
        }
        m_out_a = new Matrix( N, 1, 1 );        // 1d N neuron activations (output)
        if( trainable ) {
            m_out_dw = new Matrix( N, 1, 1 );   // 1d N neuron dw
        }
        return;
    }
    
    unsigned long
    DnnLayerFullyConnected::neuronCount( void ) const {
        return m_neuron_count;
    }
    
    bool
    DnnLayerFullyConnected::forward( void ) {
        // -----------------------------------------------------------------------------------
        // Forward propagate while training
        // N = number of neurons
        // S = size of input data
        // m_in_a[S]
        // m_in_dw[S]
        // m_w[N,S+1]  = neuron weight + bias weight
        // m_dw[N,S+1] = dw + bias weight
        // m_out_a[N]  = activations of each neuron
        // m_out_dw[N]
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_dw == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *        input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const  inEnd = m_in_a->end();     // A pointer just past the end of the input
        const DNN_NUMERIC *           ww = m_w->data();       // weights[n,s]
              DNN_NUMERIC *       output = m_out_a->data();   // activations[n] for the neurons in this layer
        const DNN_NUMERIC * const outEnd = m_out_a->end();    // A pointer just past the end of the activations
        
        while( output < outEnd ) {              // for( i = 0; i < N; ) Loop for each neuron activation
            *output = 0.0;                      // a[i] = 0; Start with zero activation for this neuron.
            const DNN_NUMERIC *in = input;      // Beginning of the input data
            while( in < inEnd ) {               // for( j = 0; j < S; ) Loop for each input element
                *output += *ww++ * *in++;       // a[i] += w[i][j] * in[j];
            }
            *output += *ww++;                   // Add the bias (1.0 * bias)
            output++;                           // i++
        }

        if( m_next_layer != 0 ) {
            return m_next_layer->forward();
        }
        return true;
    }

    bool
    DnnLayerFullyConnected::backprop( void ) {
        // -----------------------------------------------------------------------------------
        // Back propagate while training
        // N = number of neurons
        // S = size of input data
        // m_in_a[S]   = input data
        // m_in_dw[S]
        // m_w[N,S+1]  = neuron weight + bias weight
        // m_dw[N,S+1] = dw + bias weight
        // m_out_a[N]  = activations of each neuron
        // m_out_dw[N]
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_in_dw == 0 || m_w == 0 || m_dw == 0 || m_out_dw == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *          input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const    inEnd = m_in_a->end();             // A pointer just past the end of the input
              DNN_NUMERIC *        inputDw = m_in_dw->data();
        const DNN_NUMERIC *             ww = m_w->dataReadOnly();       // weights[n,s]
              DNN_NUMERIC *             dw = m_dw->data();              // dw[n,s]
        const DNN_NUMERIC *          outDw = m_out_dw->dataReadOnly();
        const DNN_NUMERIC * const outDwEnd = m_out_dw->end();
        
        m_in_dw->zero();                            // Zero previous back propagation result.

        while( outDw < outDwEnd ) {                 // for ii = 0 to N: Loop for each neuron activation
            const DNN_NUMERIC   *in = input;
                  DNN_NUMERIC *inDw = inputDw;
            const DNN_NUMERIC  grad = *outDw++;
            
            while( in < inEnd ) {                   // Loop for each input element
                *inDw++ += *ww++ * grad;
                *dw++   += *in++ * grad;
            }
            *dw++ += grad;                          // bias dw
            ww++;                                   // bias weight (skip)
        }

        if( m_prev_layer != 0 ) {
            return m_prev_layer->backprop();
        }
        return true;
    }
    
    bool
    DnnLayerFullyConnected::predict( const Matrix &data ) {
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
