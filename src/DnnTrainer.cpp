// --------------------------------------------------------------------
//  dnnTrainer.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayer.h"
#include "DnnTrainer.h"

namespace tfs {
    
    DnnTrainer::DnnTrainer( Dnn *dnn ) :
    m_dnn( dnn ),
    m_learning_rate( 0.01 ),
    m_l1_decay( 0.0 ),
    m_l2_decay( 0.0 ),
    m_momentum( 0.9 ),
    m_loss(     0.0 ),
    m_batch_size( 1 ),
    m_k(          0 ),
    m_trainable_handle( 0 ),    // Beginning of the "trainables" array.
    m_trainable_end(    0 ),    // End of the "trainables" array.
    m_gsum(   0 ),              // last iteration gradients (used for momentum calculations)
    m_xsum(   0 ) {             // used in adam or adadelta
        setUpTrainables();
    }
    
    DnnTrainer::~DnnTrainer( void ) {
        m_trainable_handle = 0;
        m_trainable_end    = 0;
        std::vector< Trainable* >::const_iterator trainer_end = m_trainables.end();
        for( std::vector< Trainable* >::const_iterator it = m_trainables.begin(); it != trainer_end; it++ ) {
            Trainable *trainable = *it;
            delete trainable;
        }
        m_trainables.clear();
        delete m_gsum;
        delete m_xsum;
        m_gsum = 0;
        m_xsum = 0;
        m_dnn  = 0;
    }
    
    void
    DnnTrainer::setUpTrainable( Matrix *weights, Matrix *gradiant, const DNN_NUMERIC l1_decay_mul, const DNN_NUMERIC l2_decay_mul ) {
        if( weights == 0 || gradiant == 0 ) {
            return;
        }
        Trainable *trainable = new Trainable( weights, gradiant );
        trainable->l1_decay_mul = l1_decay_mul;
        trainable->l2_decay_mul = l2_decay_mul;
        if( trainable->ok()) {
            m_trainables.push_back( trainable );
        } else {
            log_error( "Bad trainable detected." );
            delete trainable;
        }
        return;
    }
    
    void
    DnnTrainer::setUpTrainables( void ) {
        if( m_dnn == 0 ) {
            return;
        }
        DnnLayer *layer = (DnnLayer*) m_dnn->getLayerInput();
        while( layer != 0 ) {
            Matrix *weights  = layer->weights();
            Matrix *gradiant = layer->gradiant();
            Matrix *bias     = layer->bias();
            Matrix *biasDw   = layer->biasDw();
            
            const DNN_NUMERIC l1_decay_mul = layer->l1DecayMultiplier();
            const DNN_NUMERIC l2_decay_mul = layer->l2DecayMultiplier();
            
            setUpTrainable( weights, gradiant, l1_decay_mul, l2_decay_mul );
            setUpTrainable( bias,    biasDw,   l1_decay_mul, l2_decay_mul );
            
            layer = layer->getNextLayer();
        }
        const unsigned long trainableCount = m_trainables.size();
        if( trainableCount < 1 ) {
            log_error( "No trainable layers found" );
            return;
        }
        m_trainable_handle = m_trainables.data();
        m_trainable_end    = m_trainable_handle + trainableCount;
        return;
    }
    
    void
    DnnTrainer::setupSums( bool xsum ) {
        Trainable **trainableHandle = m_trainable_handle;
        Trainable **trainableEnd    = m_trainable_end;
        unsigned long count         = 0;
        while( trainableHandle < trainableEnd ) {
            Trainable *trainable = *trainableHandle++;
            count += trainable->weightCount();
        }
        m_gsum = new Matrix( count );
        m_gsum->zero();
        if( xsum ) {
            m_xsum = new Matrix( count );
            m_xsum->zero();
        }
        return;
    }
    
    Matrix*
    DnnTrainer::getMatrixInput( void ) {
        Matrix *matrix = 0;
        if( m_dnn != 0 ) {
            matrix = m_dnn->getMatrixInput();
        }
        return matrix;
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

    unsigned long
    DnnTrainer::k( void ) const {
        return m_k;
    }
    unsigned long
    DnnTrainer::k( unsigned long value ) {
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
    DnnTrainer::train( const Matrix &expectation ) {    // Input from input matrix.
        m_loss = 0.0;
        return m_loss;
    }

    DNN_NUMERIC
    DnnTrainer::train( const DMatrix &expectation ) {   // Input from input matrix.
        m_loss = 0.0;
        return m_loss;
    }
    
    DNN_NUMERIC
    DnnTrainer::train( const DNN_INTEGER expectation ) {
        return 0.0;
    }

    
}   // namespace tfs
