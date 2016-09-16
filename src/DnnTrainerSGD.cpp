// --------------------------------------------------------------------
//  DnnTrainerSGD.cpp - Stochastic Gradient Descent
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnTrainerSGD.hpp"
#include "Error.hpp"

namespace tfs {
    
    DnnTrainerSGD::DnnTrainerSGD( Dnn *dnn ) :
    tfs::DnnTrainer( dnn ) {
        // Constructor
    }
    
    DnnTrainerSGD::~DnnTrainerSGD( void ) {
        // Destructor
    }
    
    DNN_NUMERIC
    DnnTrainerSGD::adjustWeights( void ) {
        DNN_NUMERIC *gsum = 0;
        if( m_momentum > 0.0 ) {
            if( m_gsum == 0 ) {
                setupSums( false );  // setup only gsum[]
            }
            gsum = m_gsum->data();
        }
        
        Trainable **trainableHandle = m_trainable_handle;
        Trainable **trainableEnd    = m_trainable_end;
        
        while( trainableHandle < trainableEnd ) {
            Trainable   *trainable   = *trainableHandle++;      // trainable != 0 & ok() from DnnTrainer::setUpTrainables()
            DNN_NUMERIC *weight      = trainable->weightStart;
            const DNN_NUMERIC *weightEnd   = trainable->weightEnd;
            DNN_NUMERIC *gradient    = trainable->gradientStart;
            const DNN_NUMERIC *gradientEnd = trainable->gradientEnd;
            const DNN_NUMERIC l1_decay     = m_l1_decay * trainable->l1_decay_mul;
            const DNN_NUMERIC l2_decay     = m_l2_decay * trainable->l2_decay_mul;
            
            if( weight == 0 || gradient == 0 ) {
                log_error( "Trainable has weight or gradient == 0" );
                return m_loss;
            }
            
            while( weight < weightEnd ) {
                const DNN_NUMERIC     ww = *weight;
                const DNN_NUMERIC l1grad = l1_decay * (ww > 0.0 ? 1.0 : -1.0);
                const DNN_NUMERIC l2grad = l2_decay * ww;
                
                const DNN_NUMERIC gij = ( l1grad + l2grad + *gradient ) / m_batch_size; // raw batch gradient
                
                if( gradient >= gradientEnd ) {
                    log_error( "Trainable has gradient[] smaller than weight[] in size." );
                    return m_loss;
                }
                if( m_momentum > 0.0 && gsum != 0 ) {
                    const DNN_NUMERIC dx = m_momentum * *gsum - m_learning_rate * gij;
                    *gsum++    = dx;    // Save next momentum iteration
                    *weight++ += dx;    // apply corrected gradient
                } else {
                    *weight++ -= m_learning_rate * gij;
                }
                *gradient++ = 0.0;
            }
        }
        return m_loss;
    }
    
}   // namespace tfs
