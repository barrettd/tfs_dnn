// --------------------------------------------------------------------
//  DnnTrainer.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnnTrainer_hpp
#define dnnTrainer_hpp

#include <vector>
#include "Dnn.hpp"

namespace tfs {

    class DnnTrainer {  // base class to all of the trainers.  The trainers differ in adjustWeights() implementation.
    protected:
        Dnn          *m_dnn;
        DNN_NUMERIC   m_learning_rate;
        DNN_NUMERIC   m_l1_decay;
        DNN_NUMERIC   m_l2_decay;
        DNN_NUMERIC   m_momentum;
        DNN_NUMERIC   m_loss;
        unsigned long m_batch_size;
        unsigned long m_k;            // iteration counter
        Trainable               **m_trainable_handle;
        Trainable               **m_trainable_end;
        std::vector< Trainable* > m_trainables;
        Matrix     *m_gsum;
        Matrix     *m_xsum;

    protected:
        void setUpTrainable( Matrix *weights, Matrix *gradient, DNN_NUMERIC l1_decay_mul, DNN_NUMERIC l2_decay_mul );
        void setUpTrainables( void );
        void setupSums( bool xsum );    // Set up gsum[] and (optionally) xsum[]

        virtual DNN_NUMERIC adjustWeights( void );
        
    public:
        DnnTrainer( Dnn *dnn );
        virtual ~DnnTrainer( void );
        
        Matrix*       getMatrixInput( void );

        DNN_NUMERIC   learningRate( void ) const;           // get()
        DNN_NUMERIC   learningRate( DNN_NUMERIC value );    // set( value )
        
        DNN_NUMERIC   l1Decay( void ) const;
        DNN_NUMERIC   l1Decay( DNN_NUMERIC value );

        DNN_NUMERIC   l2Decay( void ) const;
        DNN_NUMERIC   l2Decay( DNN_NUMERIC value );
        
        DNN_NUMERIC   momentum( void ) const;
        DNN_NUMERIC   momentum( DNN_NUMERIC value );
        
        DNN_NUMERIC   loss( void ) const;
        DNN_NUMERIC   loss( DNN_NUMERIC value );
        
        unsigned long batchSize( void ) const;
        unsigned long batchSize( unsigned long value );
        
        unsigned long k( void ) const;
        unsigned long k( unsigned long value );
        
        DNN_NUMERIC train( const Matrix     &expectation );     // Returns loss.
        DNN_NUMERIC train( const DNN_INTEGER expectation );     // Returns loss.
        
        
    };

}   // namespace tfs

#endif /* dnnTrainer_hpp */
