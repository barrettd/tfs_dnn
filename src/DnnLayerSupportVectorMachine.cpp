// --------------------------------------------------------------------
//  DnnLayerSupportVectorMachine.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerSupportVectorMachine.hpp"
#include "Error.hpp"

namespace tfs {
    
    DnnLayerSupportVectorMachine::DnnLayerSupportVectorMachine( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_SUPPORT_VECTOR_MACHINE, previousLayer ) {
        // Constructor
        setup( false );     // We ignore m_out_dw for training, this should be the last layer.
    }
    
    DnnLayerSupportVectorMachine::~DnnLayerSupportVectorMachine( void ) {
        // Destructor
    }
    
    bool
    DnnLayerSupportVectorMachine::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        if( matrixBad( m_in_a ) || matrixBad( m_out_a )) {
            return log_error( "Not configured" );
        }
        m_out_a->copy( *m_in_a );   // Simple identity
        return true;
    }
    
    DNN_NUMERIC
    DnnLayerSupportVectorMachine::runBackprop( const DNN_INTEGER yy ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // Using a structured loss, which means that the score of the ground truth should be
        // higher than the score of any other class, by a margin
        // -----------------------------------------------------------------------------------
        if( matrixBad( m_in_a ) || matrixBad( m_in_dw )) {
            log_error( "Not configured for training" );
            return 0.0;
        }
        if( yy < 0 || yy >= (DNN_INTEGER) m_in_a->count()) {
            log_error( "Expectation is out of range" );
            return 0.0;
        }
        const DNN_NUMERIC *    inA = m_in_a->dataReadOnly();
              DNN_NUMERIC *   inDw = m_in_dw->data();
              DNN_NUMERIC *  inDwY = &inDw[yy];
        const DNN_NUMERIC *inDwEnd = m_in_dw->end();
        const DNN_NUMERIC   margin = 1.0;
        const DNN_NUMERIC   yscore = inA[yy];        // Ground truth score
        const DNN_NUMERIC   mscore = margin - yscore;
              DNN_NUMERIC     loss = 0.0;

        while( inDw < inDwEnd ) {
            if( inDw != inDwY ) {
                const DNN_NUMERIC ydiff = *inA + mscore;
                if( ydiff > 0.0 ) {     // Violating dimension, apply loss
                    *inDw  += 1.0;
                    *inDwY -= 1.0;
                    loss   += ydiff;
                }
            }
            inDw++;
            inA++;
        }
        return loss;
    }

    
}   // namespace tfs

