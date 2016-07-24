//
//  DnnTrainerAdaDelta.cpp
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "DnnTrainerAdaDelta.h"
#include "Error.h"

namespace tfs {
    
    DnnTrainerAdaDelta::DnnTrainerAdaDelta( Dnn *dnn ) :
    tfs::DnnTrainer( dnn ),
    m_ro(  0.95 ),
    m_eps( 1e-8 ) {
        // Constructor
    }
    
    DnnTrainerAdaDelta::~DnnTrainerAdaDelta( void ) {
        // Destructor
    }
    
    DNN_NUMERIC DnnTrainerAdaDelta::ro( void ) const              { return m_ro; }
    DNN_NUMERIC DnnTrainerAdaDelta::ro( const DNN_NUMERIC value ) { return m_ro = value; }
    
    DNN_NUMERIC DnnTrainerAdaDelta::eps( void ) const              { return m_eps; }
    DNN_NUMERIC DnnTrainerAdaDelta::eps( const DNN_NUMERIC value ) { return m_eps = value; }

    DNN_NUMERIC
    DnnTrainerAdaDelta::train( const DNN_INTEGER expectation ) {
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
            log_error( "No gradients available for training" );
            return m_loss;
        }
        if( !m_dnn->forward()) {
            return m_loss;
        }
        m_loss = m_dnn->backprop( expectation );
        m_k++;
        if( m_k % m_batch_size ) {
            return m_loss;
        }
        if( m_gsum == 0 ) {
            setupSums( true );  // setup gsum[] and xsum[]
        }
        // Set up for modifying the gradients.
        DNN_NUMERIC l1_decay_loss = 0.0;
        DNN_NUMERIC l2_decay_loss = 0.0;
        
        Trainable **trainableHandle = m_trainable_handle;
        Trainable **trainableEnd    = m_trainable_end;
        
        DNN_NUMERIC *gsum = m_gsum->data();
        DNN_NUMERIC *xsum = m_xsum->data();
        
        while( trainableHandle < trainableEnd ) {
                  Trainable   *trainable   = *trainableHandle++;      // trainable != 0 & ok() from DnnTrainer::setUpTrainables()
                  DNN_NUMERIC *weight      = trainable->weightStart;
            const DNN_NUMERIC *weightEnd   = trainable->weightEnd;
                  DNN_NUMERIC *gradient    = trainable->gradientStart;
            const DNN_NUMERIC *gradientEnd = trainable->gradientEnd;
            const DNN_NUMERIC l1_decay     = trainable->l1_decay_mul * m_l1_decay;
            const DNN_NUMERIC l2_decay     = trainable->l2_decay_mul * m_l2_decay;
            
            if( weight == 0 || gradient == 0 ) {
                log_error( "Trainable has weight or gradient == 0" );
                return m_loss;
            }
        
            while( weight < weightEnd ) {
                const DNN_NUMERIC ww = *weight;
                
                l1_decay_loss += l1_decay * fabs( ww );
                l2_decay_loss += l2_decay * ww * ww / 2.0;          // accumulate weight decay loss
                
                const DNN_NUMERIC l1grad = l1_decay * (ww > 0.0 ? 1.0 : -1.0);
                const DNN_NUMERIC l2grad = l2_decay * ww;
                
                const DNN_NUMERIC gij = ( l1grad + l2grad + *gradient ) / m_batch_size; // raw batch gradient
                
                if( gradient >= gradientEnd ) {
                    log_error( "Trainable has gradient[] smaller than weight[] in size." );
                    return m_loss;
                }
                *gsum = m_ro * *gsum + ( 1 - m_ro ) * gij * gij;
                
                const DNN_NUMERIC dx = - sqrt(( *xsum + m_eps ) / ( *gsum + m_eps )) * gij;
                
                *xsum = m_ro * *xsum + ( 1 - m_ro ) * dx * dx;      // xsum lags behind gsum by 1.
                *weight++ += dx;
                gsum++;
                xsum++;
                *gradient++ = 0.0;
            }
        }
        return m_loss += (l1_decay_loss + l2_decay_loss);
    }
    
}   // namespace tfs
