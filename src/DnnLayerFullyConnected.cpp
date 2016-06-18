// --------------------------------------------------------------------
//  DnnLayerFullyConnected.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerFullyConnected.h"
#include "Error.h"

namespace tfs {
    
    DnnLayerFullyConnected::DnnLayerFullyConnected( DnnLayer *previousLayer, unsigned long neuronCount, const bool trainable ):
    DnnLayer( LAYER_FULLY_CONNECTED, previousLayer ),
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
    
    unsigned long
    DnnLayerFullyConnected::getNeuronCount( void ) const {
        return m_neuron_count;
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
        if( matrixBad( m_in_a )) {
            log_error( "Input activation matrix is bad" );
            return;
        }
        const unsigned long N = m_neuron_count;
        const unsigned long S = m_in_a->count();    // 1d input, S elements.

        m_w = new Matrix( N, S+1 );                 // 2d neuron weights N x (S+1)
        if( trainable ) {
            m_dw = new Matrix( *m_w );              // 2d neuron weights N x (S+1)
        }
        m_out_a = new Matrix( N );                  // 1d N neuron activations (output)
        if( trainable ) {
            m_out_dw = new Matrix( *m_out_a );      // 1d N neuron dw
        }
        return;
    }
    
    unsigned long
    DnnLayerFullyConnected::neuronCount( void ) const {
        return m_neuron_count;
    }
    
    bool
    DnnLayerFullyConnected::runBias( DNN_NUMERIC value ) {
        // -----------------------------------------------------------------------------------
        // virtual: Set the biases for the learning layers.
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_out_a == 0 ) {
            return false;
        }
        // TODO: Clean this up.
//        const unsigned long N = m_neuron_count;
//        const unsigned long S = m_in_a->count(); // 1d input, S elements.
//        DNN_NUMERIC       *ww = m_w->data();       // weights[n,s]
//
//        for( unsigned long ii = 1; ii < N; ii++ ) {
//            unsigned long index = ii * S;
//            ww[index] = 0.0;
//        }
        const DNN_NUMERIC *        input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const  inEnd = m_in_a->end();     // A pointer just past the end of the input
              DNN_NUMERIC *           ww = m_w->data();       // weights[n,s]
        const DNN_NUMERIC *       output = m_out_a->data();   // activations[n] for the neurons in this layer
        const DNN_NUMERIC * const outEnd = m_out_a->end();    // A pointer just past the end of the activations

        while( output < outEnd ) {              // for( i = 0; i < N; ) Loop for each neuron activation
            const DNN_NUMERIC *in = input;      // Beginning of the input data
            while( in < inEnd ) {               // for( j = 0; j < S; ) Loop for each input element
                ww++;
                in++;
            }
            *ww++ = value;                      // Set the bias
            output++;
        }
        return true;
    }
    
    bool
    DnnLayerFullyConnected::runForward( void ) {
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
        // ok: 24 May 2016
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *        input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const  inEnd = m_in_a->end();     // A pointer just past the end of the input
        const DNN_NUMERIC *           ww = m_w->data();       // weights[n,s]
              DNN_NUMERIC *       output = m_out_a->data();   // activations[n] for the neurons in this layer
        const DNN_NUMERIC * const outEnd = m_out_a->end();    // A pointer just past the end of the activations
        
        while( output < outEnd ) {              // for( i = 0; i < N; ) Loop for each neuron activation
            DNN_NUMERIC aa = 0.0;               // a[i] = 0; Start with zero activation for this neuron.
            const DNN_NUMERIC *in = input;      // Beginning of the input data
            while( in < inEnd ) {               // for( j = 0; j < S; ) Loop for each input element
                aa += *ww++ * *in++;            // a[i] += w[i][j] * in[j];
            }
            *output++ = aa + *ww++;             // Add the bias (1.0 * bias)
        }
        return true;
    }
    
    bool
    DnnLayerFullyConnected::runBackprop( void ) {
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
        // ok: 24 May 2016
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_dw == 0 || m_out_dw == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *          input = m_in_a->dataReadOnly();
        const DNN_NUMERIC * const    inEnd = m_in_a->end();             // A pointer just past the end of the input
              DNN_NUMERIC *             dw = m_dw->data();              // dw[n,s]
        const DNN_NUMERIC *          outDw = m_out_dw->dataReadOnly();
        const DNN_NUMERIC * const outDwEnd = m_out_dw->end();
        
        if( m_in_dw == 0 ) {                            // Previous layer is input layer, no dw backprop needed.
            while( outDw < outDwEnd ) {                 // for ii = 0 to N: Loop for each neuron activation
                const DNN_NUMERIC   *in = input;
                const DNN_NUMERIC  grad = *outDw++;
                
                while( in < inEnd ) {                   // Loop for each input element
                    *dw++ += *in++ * grad;
                }
                *dw++ += grad;                          // bias dw
            }
        } else {                                        // Previous layer is a normal layer that needs dw backpropagation.
                  DNN_NUMERIC *inputDw = m_in_dw->data();
            const DNN_NUMERIC *     ww = m_w->dataReadOnly();       // weights[n,s]

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
        }
        return true;
    }
    

    
}   // namespace tfs
