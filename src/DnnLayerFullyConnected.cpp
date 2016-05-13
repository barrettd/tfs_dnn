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

    DnnLayerFullyConnected::DnnLayerFullyConnected(DnnLayer     *previousLayer,
                                                   unsigned long neuronCount,
                                                   const bool    trainable ):
    DnnLayer( NAME, previousLayer ),
    m_neuron_count( neuronCount ),
    m_l1_decay_mul( 0.0 ),
    m_l2_decay_mul( 1.0 ) {
        // Constructor
        const unsigned long  inX = previousLayer->aX();   // previousLayer should not be null.
        const unsigned long  inY = previousLayer->aY();
        const unsigned long  inZ = previousLayer->aZ();
        const unsigned long outX = 1;
        const unsigned long outY = 1;
        const unsigned long outZ = neuronCount;
        
        setup( inX, inY, inZ, outX, outY, outZ, trainable );
    }
    
    DnnLayerFullyConnected::~DnnLayerFullyConnected( void ) {
        // Destructor
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
    
    //forward: function(V, is_training) {
    //    this.in_act = V;
    //    var A = new Vol(1, 1, this.out_depth, 0.0);
    //    var Vw = V.w;
    //    for(var i=0;i<this.out_depth;i++) {
    //        var a = 0.0;
    //        var wi = this.filters[i].w;
    //        for(var d=0;d<this.num_inputs;d++) {
    //            a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
    //        }
    //        a += this.biases.w[i];
    //        A.w[i] = a;
    //    }
    //    this.out_act = A;
    //    return this.out_act;
    //},

    bool
    DnnLayerFullyConnected::forward( const Matrix &data ) {
        // Forward propagate while training
        if( m_w == 0 || m_dw == 0 || m_a == 0 ) {
            return log_error( "Not configured for training" );
        }
        if( m_w->size() != data.size()) {
            return log_error( "Input matrix does not match expected size" );
        }
        m_pa = &data;
        DNN_NUMERIC *activation    = m_a->data();
        DNN_NUMERIC *activationEnd = activation + m_a->size();
        while( activation < activationEnd ) {
            DNN_NUMERIC aa = 0.0;
            
        }
        if( m_next_layer != 0 ) {
            return m_next_layer->forward( data );
        }
        return true;
    }

//backward: function() {
//    var V = this.in_act;
//    V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
//    
//    // compute gradient wrt weights and data
//    for(var i=0;i<this.out_depth;i++) {
//        var tfi = this.filters[i];
//        var chain_grad = this.out_act.dw[i];
//        for(var d=0;d<this.num_inputs;d++) {
//            V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
//            tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
//        }
//        this.biases.dw[i] += chain_grad;
//    }
//},
//
    bool
    DnnLayerFullyConnected::backprop( void ) {
        // Back propagate while training
        if( m_w == 0 || m_dw == 0 || m_a == 0 || m_pa == 0 ) {
            return log_error( "Not configured for training" );
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
