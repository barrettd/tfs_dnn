// --------------------------------------------------------------------
//  DnnTrainerSGD.cpp - Stochastic Gradient Descent
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnTrainerSGD.h"
#include "Error.h"

namespace tfs {
    
    DnnTrainerSGD::DnnTrainerSGD( Dnn *dnn ) :
    tfs::DnnTrainer( dnn ) {
        // Constructor
    }
    
    DnnTrainerSGD::~DnnTrainerSGD( void ) {
        // Destructor
    }
    
    DNN_NUMERIC
    DnnTrainerSGD::train( const DNN_INTEGER expectation ) {
        // Assume: Input matrix already set for the DNN
        m_loss = 0.0;
        if( m_dnn == 0 ) {
            log_error( "No DNN set" );
            return m_loss;
        }
        if( m_batch_size == 0 ) {
            log_error( "Batch size 0 - will be a divide by zero error" );
            return m_loss;
        }
        if( m_trainable_handle == 0 || m_trainable_end == 0 ) {
            log_error( "No gradiants available for training" );
            return m_loss;
        }
        log_debug( "about to do forward()" );
        if( !m_dnn->forward()) {
            return m_loss;
        }
        log_debug( "about to do backprop()" );
        m_loss = m_dnn->backprop( expectation );
        m_k++;
        if( m_k % m_batch_size ) {
            log_debug( "returning" );
            return m_loss;
        }
        log_debug( "about to modify gradiants" );
        // Set up for modifying the gradiants.
        DNN_NUMERIC l1_decay_loss = 0.0;
        DNN_NUMERIC l2_decay_loss = 0.0;
        
        Trainable **trainableHandle = m_trainable_handle;
        Trainable **trainableEnd    = m_trainable_end;
        
        while( trainableHandle < trainableEnd ) {
                  Trainable   *trainable   = *trainableHandle++;      // trainable != 0 & ok() from DnnTrainer::setUpTrainables()
                  DNN_NUMERIC *weight      = trainable->weightStart;
            const DNN_NUMERIC *weightEnd   = trainable->weightEnd;
                  DNN_NUMERIC *gradiant    = trainable->gradiantStart;
            const DNN_NUMERIC *gradiantEnd = trainable->gradiantEnd;
            const DNN_NUMERIC l1_decay     = m_l1_decay * trainable->l1_decay_mul;
            const DNN_NUMERIC l2_decay     = m_l2_decay * trainable->l2_decay_mul;
            
            if( weight == 0 || gradiant == 0 ) {
                log_error( "Trainable has weight or gradiant == 0" );
                return m_loss;
            }
            
            while( weight < weightEnd ) {
                const DNN_NUMERIC ww = *weight;
                
                l1_decay_loss += l1_decay * fabs( ww );
                l2_decay_loss += l2_decay * ww * ww / 2.0;          // accumulate weight decay loss
                
                const DNN_NUMERIC l1grad = l1_decay * (ww > 0.0 ? 1.0 : -1.0);
                const DNN_NUMERIC l2grad = l2_decay * ww;
            
                const DNN_NUMERIC gij = ( l1grad + l2grad + *gradiant ) / m_batch_size; // raw batch gradient

                if( gradiant >= gradiantEnd ) {
                    log_error( "Trainable has gradiant[] smaller than weight[] in size." );
                    return m_loss;
                }
                *gradiant++ = 0.0;
                *weight++ -= m_learning_rate * gij;
            }
        }
        log_debug( "returning" );
        return m_loss += (l1_decay_loss + l2_decay_loss);
    }
    
}   // namespace tfs
