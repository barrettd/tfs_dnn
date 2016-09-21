// --------------------------------------------------------------------
//  DnnLayerRegression.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerRegression.hpp"
#include "Error.hpp"

namespace tfs {
    
    DnnLayerRegression::DnnLayerRegression( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_REGRESSION, previousLayer ) {
        // Constructor
        setup( trainable );
    }
    
    DnnLayerRegression::~DnnLayerRegression( void ) {
        // Destructor
    }
    
    bool
    DnnLayerRegression::runForward(  void ) {
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
    DnnLayerRegression::runBackprop( const  Matrix &expectation ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        if( m_in_dw == 0 ) {
            log_error( "Not configured for training" );
            return 0.0;
        }
        if( m_in_dw->count() != expectation.count()) {
            log_error( "Gradiant and expectation not the same size" );
            return 0.0;
        }
        const DNN_NUMERIC *             yy = expectation.dataReadOnly();
        const DNN_NUMERIC *            inA = m_in_a->dataReadOnly();
              DNN_NUMERIC *           inDw = m_in_dw->data();
        const DNN_NUMERIC * const  inDwEnd = m_in_dw->end();
              DNN_NUMERIC             loss = 0.0;
        
        while( inDw < inDwEnd ) {
            const DNN_NUMERIC dy = *inA++ - *yy++;
            *inDw++ = dy;
            loss += 0.5 * dy*dy;
        }
        return loss;
    }

    
}   // namespace tfs

