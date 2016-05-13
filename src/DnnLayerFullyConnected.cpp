// --------------------------------------------------------------------
//  DnnLayerFullyConnected.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerFullyConnected.h"

namespace tfs {
    
    static const char *NAME = "fc";

    const char*
    DnnLayerFullyConnected::className( void ) {
        return NAME;
    }

    DnnLayerFullyConnected::DnnLayerFullyConnected(DnnLayer     *previousLayer,
                                                   unsigned long neuronCount,
                                                   DNN_NUMERIC   bias,
                                                   const bool    trainable ):
    DnnLayer( NAME, previousLayer ),
    m_neuron_count( neuronCount ),
    m_bias( bias ),
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

    
    DnnLayerFullyConnected::~DnnLayerFullyConnected( void ) {
        // Destructor
    }
    
    unsigned long
    DnnLayerFullyConnected::neuronCount( void ) const {
        return m_neuron_count;
    }
    
    DNN_NUMERIC
    DnnLayerFullyConnected::bias( void ) const {
        return m_bias;
    }
    DNN_NUMERIC
    DnnLayerFullyConnected::bias( DNN_NUMERIC value ) {
        return m_bias = value;
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

    
}   // namespace tfs
