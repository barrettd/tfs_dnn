//
//  DnnTrainerAdam.cpp
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "DnnTrainerAdam.hpp"

namespace tfs {
    
    DnnTrainerAdam::DnnTrainerAdam( Dnn *dnn ):
    tfs::DnnTrainer( dnn ),
    m_eps( 1e-8 ),
    m_beta1( 0.9 ),
    m_beta2( 0.999 ) {

    }
    
    DnnTrainerAdam::~DnnTrainerAdam( void ) {        
    }
    
    DNN_NUMERIC DnnTrainerAdam::eps( void ) const               { return m_eps; }
    DNN_NUMERIC DnnTrainerAdam::eps( const DNN_NUMERIC value )  { return m_eps = value; }

    DNN_NUMERIC DnnTrainerAdam::beta1( void ) const             { return m_beta1; }
    DNN_NUMERIC DnnTrainerAdam::beta1( const DNN_NUMERIC value ){ return m_beta1 = value; }
    
    DNN_NUMERIC DnnTrainerAdam::beta2( void ) const             {  return m_beta2; }
    DNN_NUMERIC DnnTrainerAdam::beta2( const DNN_NUMERIC value ){  return m_beta2 = value; }
    
    DNN_NUMERIC
    DnnTrainerAdam::adjustWeights( void ) {
        if( m_gsum == 0 ) {
            setupSums( true );  // setup gsum[] and xsum[]
        }
        DNN_NUMERIC *gsum = m_gsum->data();
        DNN_NUMERIC *xsum = m_xsum->data();
        
        Trainable **trainableHandle = m_trainable_handle;
        Trainable **trainableEnd    = m_trainable_end;
        
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
                const DNN_NUMERIC     ww = *weight;
                const DNN_NUMERIC l1grad = l1_decay * (ww > 0.0 ? 1.0 : -1.0);
                const DNN_NUMERIC l2grad = l2_decay * ww;
                
                const DNN_NUMERIC gij = ( l1grad + l2grad + *gradient ) / m_batch_size; // raw batch gradient
                
                if( gradient >= gradientEnd ) {
                    log_error( "Trainable has gradient[] smaller than weight[] in size." );
                    return m_loss;
                }
                *gsum = *gsum * m_beta1 + (1- m_beta1) * gij;       // update biased first moment estimate
                *xsum = *xsum * m_beta2 + (1- m_beta2) * gij * gij; // update biased second moment estimate
                
                const DNN_NUMERIC biasCorr1 = *gsum * (1 - pow(m_beta1, m_k)); // correct bias first moment estimate
                const DNN_NUMERIC biasCorr2 = *xsum * (1 - pow(m_beta2, m_k)); // correct bias second moment estimate
                
                const DNN_NUMERIC dx = - m_learning_rate * biasCorr1 / (sqrt(biasCorr2) + m_eps);
                
                *weight++ += dx;
                gsum++;
                xsum++;
                *gradient++ = 0.0;
            }
        }
        return m_loss;
    }
    
    
}   // namespace tfs

