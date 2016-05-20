// --------------------------------------------------------------------
//  DnnTrainer.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnnTrainer_h
#define dnnTrainer_h

#include "Dnn.h"

namespace tfs {

    class DnnTrainer {  // sgd/adam/adagrad/adadelta/windowgrad/netsterov
    protected:
        Dnn          *m_dnn;
        DNN_NUMERIC   m_learning_rate;
        DNN_NUMERIC   m_l1_decay;
        DNN_NUMERIC   m_l2_decay;
        DNN_NUMERIC   m_momentum;
        DNN_NUMERIC   m_loss;
        unsigned long m_batch_size;
        unsigned long m_k;            // iteration counter
        
//        this.gsum = []; // last iteration gradients (used for momentum calculations)
        
    public:
        DnnTrainer( Dnn *dnn );
        virtual ~DnnTrainer( void );
        
        Matrix*        getMatrixInput( void );

        DNN_NUMERIC   learningRate( void ) const;
        DNN_NUMERIC   learningRate( DNN_NUMERIC value );
        
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
        
        unsigned long   k( void ) const;
        unsigned long   k( unsigned long value );
        
        virtual DNN_NUMERIC train( const  Matrix &expectation );     // Returns loss.
        virtual DNN_NUMERIC train( const DMatrix &expectation );     // Returns loss.
        
        
    };

}   // namespace tfs

#endif /* dnnTrainer_h */
