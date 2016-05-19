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
    }
    
    DnnLayerRectifiedLinearUnit::~DnnLayerRectifiedLinearUnit( void ) {
        // Destructor
    }
    
    bool
    DnnLayerRectifiedLinearUnit::threshold( const Matrix &data ) {
        if( m_out_a == 0 ) {
            return log_error( "Activation matrix not allocated" );
        }
        unsigned long count = m_out_a->size();
        if( count != data.size()) {
            return log_error( "Input matrix does not match activation matrix size" );
        }
        const DNN_NUMERIC *src = data.dataReadOnly();
        const DNN_NUMERIC *end = data.end();
              DNN_NUMERIC *dst = m_out_a->data();
        while( src < end ) {
            if( *src >= 0.0 ) {     // Threshold at 0.0
                *dst++ = *src++;
            } else {
                *dst++ = 0.0;
                src++;
            }
        }
        return true;
    }
   
    bool
    DnnLayerRectifiedLinearUnit::forward( void ) {
        // Forward propagate while training
        if( m_w == 0 || m_dw == 0 || m_out_a == 0 || m_in_a == 0 ) {
            return log_error( "Not configured for training" );
        }
        if( !threshold( *m_in_a )) {
            return log_error( "Thresholding failed" );
        }
        if( m_next_layer != 0 ) {
            return m_next_layer->forward();
        }
        return true;
    }
    
    bool
    DnnLayerRectifiedLinearUnit::backprop( void ) {  // Back propagate while training
        
        if( m_prev_layer != 0 ) {
            return m_prev_layer->backprop();
        }
        return true;
    }
    
    bool
    DnnLayerRectifiedLinearUnit::predict( const Matrix &data ) {
        // Forward progagate when predicting
        if( !threshold( data )) {
            return log_error( "Thresholding failed" );
        }
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( *m_out_a );
        }
        return true;
    }
    
//forward: function(V, is_training) {
//    this.in_act = V;
//    var V2 = V.clone();
//    var N = V.w.length;
//    var V2w = V2.w;
//    for(var i=0;i<N;i++) {
//        if(V2w[i] < 0) V2w[i] = 0; // threshold at 0
//    }
//    this.out_act = V2;
//    return this.out_act;
//},
//backward: function() {
//    var V = this.in_act; // we need to set dw of this
//    var V2 = this.out_act;
//    var N = V.w.length;
//    V.dw = global.zeros(N); // zero out gradient wrt data
//    for(var i=0;i<N;i++) {
//        if(V2.w[i] <= 0) V.dw[i] = 0; // threshold
//        else V.dw[i] = V2.dw[i];
//    }
//},
//getParamsAndGrads: function() {
//    return [];
//},

    
}   // namespace tfs

