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
    m_neuron_count( neuronCount ),
    m_l1_decay_mul( 0.0 ),
    m_l2_decay_mul( 1.0 ) {         // Constructor
        if( previousLayer != 0 ) {  // previousLayer should not be null.
            setup( previousLayer, trainable );
        } else {
            log_error( "previousLayer is null" );
        }
    }
    
    DnnLayerFullyConnected::~DnnLayerFullyConnected( void ) {
        // Destructor
    }
    
    void
    DnnLayerFullyConnected::setup( DnnLayer *previousLayer, const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // N = number of neurons
        // S = size of input data
        //  w[N,S+1] = neuron weight + bias weight
        // dw[N]     = gradiant + d/dw bias
        //  a[N]     = activations of each neuron
        // -----------------------------------------------------------------------------------
        teardown();
        const unsigned long N = m_neuron_count;
        const unsigned long S = previousLayer->aSize(); // 1d input, S elements.

        m_w = new Matrix( N, S + 1, 1 );        // 2d neuron weights N x (S+1)
        if( trainable ) {
            m_dw = new Matrix( N, 1, 1 );       // 1d N neuron dw
        }
        m_a = new Matrix( N, 1, 1 );            // 1d N neuron activations (output)
        return;
    }
    
    unsigned long
    DnnLayerFullyConnected::neuronCount( void ) const {
        return m_neuron_count;
    }
    
    DNN_NUMERIC
    DnnLayerFullyConnected::l1DecayMultiplier( void ) const {
        return m_l1_decay_mul;
    }
    DNN_NUMERIC
    DnnLayerFullyConnected::l1DecayMultiplier( DNN_NUMERIC value ) {
        return m_l1_decay_mul = value;
    }

    DNN_NUMERIC
    DnnLayerFullyConnected::l2DecayMultiplier( void ) const {
        return m_l2_decay_mul;
    }
    DNN_NUMERIC
    DnnLayerFullyConnected::l2DecayMultiplier( DNN_NUMERIC value ) {
        return m_l2_decay_mul = value;
    }
    
    bool
    DnnLayerFullyConnected::forward( void ) {
        // -----------------------------------------------------------------------------------
        // Forward propagate while training
        // N = number of neurons
        // S = size of input data
        // m_w[N,S+1] = neuron weight + bias weight
        // m_a[N]     = activations of each neuron
        // -----------------------------------------------------------------------------------
        if( m_w == 0 || m_dw == 0 || m_a == 0 || m_pa == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *      input = m_pa->dataReadOnly();
        const DNN_NUMERIC * const iEnd = m_pa->end(); // A pointer just past the end of the input
        const DNN_NUMERIC *         ww = m_w->data(); // weights[n,s]
              DNN_NUMERIC *         aa = m_a->data(); // activations[n] for the neurons in this layer
        const DNN_NUMERIC * const aEnd = m_a->end();  // A pointer just past the end of the activations
        
        while( aa < aEnd ) {                    // for( i = 0; i < N; ) Loop for each neuron activation
            *aa = 0.0;                          // a[i] = 0; Start with zero activation for this neuron.
            const DNN_NUMERIC *in = input;      // Beginning of the input data
            while( in < iEnd ) {                // for( j = 0; j < S; ) Loop for each input element
                *aa += *ww++ * *in++;           // a[i] += w[i][j] * in[j];
            }
            *aa += *ww++;                       // Add the bias (1.0 * bias)
            aa++;                               // i++
        }

        if( m_next_layer != 0 ) {
            return m_next_layer->forward();
        }
        return true;
    }
/*
                                var FullyConnLayer = function(opt) {
    N = m_neuron_count;         this.out_depth opt.num_neurons;
    m_l1_decay_mul = 0.0;       this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;
    m_l2_decay_mul = 1.0;       this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;
         
    S = previousLayer->aSize(); this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    m_a->x                      this.out_sx = 1;
    m_a->y                      this.out_sy = 1;
 
    bais = 1.0;                 var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    m_w[n,s+1] & m_dw[n,s+1]    this.filters = [];
                                for(var i=0;i<this.out_depth ;i++) {
                                    this.filters.push(new Vol(1, 1, this.num_inputs)); 
                                }
                                this.biases = new Vol(1, 1, this.out_depth, bias);
    
                                forward: function(V, is_training) {
    m_pa, m_pdw                 this.in_act = V;
    m_a,                        var A = new Vol(1, 1, this.out_depth, 0.0);
    m_pa                        var Vw = V.w;
    while( aa < aEnd ) {        for(var i=0;i<this.out_depth;i++) {
        *aa = 0.0;              var a = 0.0;
         ww                     var wi = this.filters[i].w;
         while( in < iEnd ) {   for(var d=0;d<this.num_inputs;d++) {
            *aa += *ww++ * *in++;       a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
        }                       }
        *aa += *ww++;           a += this.biases.w[i];
                                A.w[i] = a;
    }                           }
                                this.out_act = A;
                                return this.out_act;
                                },
                                backward: function() {
    m_pa, m_pdw                 var V = this.in_act;
    m_pdw->zero();              V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
        
                                for(var i=0;i<this.out_depth;i++) {
    m_w[n,s+1] & m_dw[n]    var tfi = this.filters[i];
            var chain_grad = this.out_act.dw[i];
            for(var d=0;d<this.num_inputs;d++) {
                V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
                tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
            }
            this.biases.dw[i] += chain_grad;
        }
    },
*/
    bool
    DnnLayerFullyConnected::backprop( void ) {
        // -----------------------------------------------------------------------------------
        // Back propagate while training
        // N = number of neurons
        // S = size of input data
        //     m_pa[S] = Activations of previous layer
        //  m_pdw[S+1] = gradiant + d/dw bias of previous layer
        //  m_w[N,S+1] = neuron weight + bias weight
        // m_dw[N]     = gradiant
        //  m_a[N]     = activations of each neuron
        // -----------------------------------------------------------------------------------
        if( m_w == 0 || m_dw == 0 || m_a == 0 || m_pa == 0 || m_pdw == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *         input = m_pa->dataReadOnly();
        const DNN_NUMERIC * const    iEnd = m_pa->end(); // A pointer just past the end of the input
              DNN_NUMERIC *          inDw = m_pdw->data();
              DNN_NUMERIC *            ww = m_w->data(); // weights[n,s]
              DNN_NUMERIC *            dw = m_dw->data(); // d/dw weights[n,s]
        const DNN_NUMERIC * const   dwEnd = m_dw->end(); // A pointer just past the end of the input
        const DNN_NUMERIC *            aa = m_a->data(); // activations[n] for the neurons in this layer

        m_pdw->zero();                      // Zero input gradiant
        
        while( dw < dwEnd ) {                    // for ii = 0 to N: Loop for each neuron activation
            const DNN_NUMERIC grad = *dw++;
            
//            while( in < iEnd ) {                // Loop for each input element
//            }
            
            aa++;
        }

        if( m_prev_layer != 0 ) {
            return m_prev_layer->backprop();
        }
        return true;
    }
    
    bool
    DnnLayerFullyConnected::predict( const Matrix &data ) {
        // Forward progagate when predicting
        if( m_w == 0 || m_a == 0 ) {
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
