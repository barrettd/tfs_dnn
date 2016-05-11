// --------------------------------------------------------------------
//  dnnTrainer.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "DnnTrainer.h"

namespace tfs {
    
    DnnTrainer::DnnTrainer( Dnn *dnn ) :
    m_dnn( dnn ),
    m_learning_rate( 0.01 ),
    m_l1_decay( 0.0 ),
    m_l2_decay( 0.0 ),
    m_batch_size( 1 ),
    m_momentum( 0.9 ),
    m_k( 0 ) {
//        this.gsum = []; // last iteration gradients (used for momentum calculations)
//        this.xsum = []; // used in adam or adadelta
 
    }
    
    DnnTrainer::~DnnTrainer( void ) {
        m_dnn = 0;
    }
    
    DNN_NUMERIC
    DnnTrainer::learningRate( void ) const {
        return m_learning_rate;
    }
    DNN_NUMERIC
    DnnTrainer::learningRate( DNN_NUMERIC value ) {
        return m_learning_rate = value;
    }
    
    DNN_NUMERIC
    DnnTrainer::l1Decay( void ) const {
        return m_l1_decay;
    }
    DNN_NUMERIC
    DnnTrainer::l1Decay( DNN_NUMERIC value ) {
        return m_l1_decay = value;
    }

    DNN_NUMERIC
    DnnTrainer::l2Decay( void ) const {
        return m_l2_decay;
    }
    DNN_NUMERIC
    DnnTrainer::l2Decay( DNN_NUMERIC value ){
        return m_l2_decay = value;
    }

    unsigned long
    DnnTrainer::batchSize( void ) const {
        return m_batch_size;
    }
    unsigned long
    DnnTrainer::batchSize( unsigned long value ) {
        return m_batch_size = value;
    }
    
    DNN_NUMERIC
    DnnTrainer::momentum( void ) const {
        return m_momentum;
    }
    DNN_NUMERIC
    DnnTrainer::momentum( DNN_NUMERIC value ) {
        return m_momentum = value;
    }

    DNN_NUMERIC
    DnnTrainer::k( void ) const {
        return m_k;
    }
    DNN_NUMERIC
    DnnTrainer::k( DNN_NUMERIC value ) {
        return m_k = value;
    }
    
    DNN_NUMERIC
    DnnTrainer::loss( void ) const {
        return m_loss;
    }
    DNN_NUMERIC
    DnnTrainer::loss( DNN_NUMERIC value ) {
        return m_loss = value;
    }

    DNN_NUMERIC
    DnnTrainer::train( const Matrix &data, const Matrix &expectation ) {
        m_loss = 0.0;
        return m_loss;
    }
    
    
}   // namespace tfs
