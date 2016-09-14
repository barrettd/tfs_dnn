// --------------------------------------------------------------------
//  DnnLayerSigmoid.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerSigmoid.hpp"
#include "Error.hpp"

namespace tfs {
    
    DnnLayerSigmoid::DnnLayerSigmoid( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_SIGMOID, previousLayer ) {
        // Constructor
    }
    
    DnnLayerSigmoid::~DnnLayerSigmoid( void ) {
        // Destructor
    }
    
    inline DNN_NUMERIC sigmoid( DNN_NUMERIC value ) {
        //        value *= STEEPNESS;
        return 1.0 / ( 1.0 + exp( -value ));    // Returns [0.0 to 1.0]
    }
    
    inline DNN_NUMERIC sigmoidD( DNN_NUMERIC sig ) {      // Helper function to generate dSigmoid/dZ from the Sigmoid.
        return sig * ( 1.0 - sig );
    }
    
    inline DNN_NUMERIC sigmoidDerivative( DNN_NUMERIC value ) {
        const DNN_NUMERIC sig = sigmoid( value );
        return sigmoidD( sig );
    }

    bool
    DnnLayerSigmoid::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0 ) {
            return log_error( "Not configured for forward propagation" );
        }
        const DNN_NUMERIC *src    = m_in_a->dataReadOnly();
        const DNN_NUMERIC *srcEnd = m_in_a->end();
              DNN_NUMERIC *dst    = m_out_a->data();
        while( src < srcEnd ) {
            *dst++ = sigmoid( *src++ );
        }
        return true;
    }
    
    bool
    DnnLayerSigmoid::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        if( m_in_dw == 0 || m_out_dw == 0 ) {
            return log_error( "Not configured for training" );
        }
        const DNN_NUMERIC *          outDw = m_out_dw->dataReadOnly();
        const DNN_NUMERIC * const outDwEnd = m_out_dw->end();
        const DNN_NUMERIC *           outA = m_out_a->dataReadOnly();
              DNN_NUMERIC *           inDw = m_in_dw->data();
        while( outDw < outDwEnd ) {
            const DNN_NUMERIC activation = *outA++;
            *inDw++ = activation * ( 1.0 - activation ) * *outDw++;
        }
        return true;
    }

}   // namespace tfs

