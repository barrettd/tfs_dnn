//
//  testCircle.cpp
//  Example inspired by http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
//
//  Created by Barrett Davis on 5/31/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "DnnBuilder.h"
#include "DnnTrainerSGD.h"
#include "Error.h"
#include "testCircle.hpp"
#include "Utility.h"

namespace tfs {
    
    static void
    spiralSetUpData( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, const int count ) {
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = random( 0.0, 2.0 );
            const DNN_NUMERIC tt = random( 0.0, 2.0 * M_PI );
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            data.push_back( xx );       // x,y pair
            data.push_back( yy );
            label.push_back( 0 );       // label
        }
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = random( 3.0, 5.0 );
            const DNN_NUMERIC tt = 2.0 * M_PI * ii / count;
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            data.push_back( xx );       // x,y pair
            data.push_back( yy );
            label.push_back( 1 );       // label
        }
        return;
    }
    
    static bool
    setupDnn( Dnn &dnn ) {
        DnnBuilder builder( dnn, ACTIVATION_TANH );
        if( !builder.addLayerInput( 1, 1, 2 )) {                // Input layer a single x, y data point.
            return log_error( "Cannot add Input layer" );
        }
        if( !builder.addLayerFullyConnected( 6 )) {             // 6 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !builder.addLayerFullyConnected( 2 )) {             // 2 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !builder.addLayerSoftmax( 2 )) {                    // Output classifier, 2 classes.
            return log_error( "Cannot add Softmax layer" );
        }
        dnn.initialize();                                       // Randomize the weights.
        const unsigned long count = dnn.count();
        log_info( "We have set up %lu layers", count );
        return true;
    }
    
    static bool
    localTestCircle( void ) {
        log_info( "Test Circle - Start" );
        
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerSGD trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.momentum(     0.1   );
        trainer.batchSize(   10     );
        trainer.l2Decay(      0.001 );
        
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        spiralSetUpData( data, label, 50 );
        
        Matrix *input = trainer.getMatrixInput();       // get the input matrix: x,y pair Matrix( 1, 1, 2 )
        if( input == 0 ) {
            return log_error( "Unable to obtain input matrix." );
        }
        const DNN_NUMERIC *dPtr = data.data();
        if( dPtr == 0 ) {
            return log_error( "Unable to obtain input data." );
        }
        const DNN_INTEGER *lPtr = label.data();
        if( lPtr == 0 ) {
            return log_error( "Unable to obtain input labels." );
        }
        const unsigned long DATA_COUNT = label.size();
        if( DATA_COUNT < 1 ) {
            return log_error( "Label data size < 1" );
        }
        const DNN_INTEGER *ePtr = lPtr + DATA_COUNT;
        
        const unsigned long MAX_ITERATION = 200;
        const DNN_NUMERIC   TARGET_LOSS   = 0.004;
        unsigned long count = 0;
        DNN_NUMERIC average_loss = 0.0;
        do {
            count++;
            average_loss = 0.0;
            for( unsigned long ii = 0; ii < MAX_ITERATION; ii++ ) {
                dPtr = data.data();
                lPtr = label.data();
                
                while( lPtr < ePtr ) {
                    DNN_NUMERIC *inPtr = input->data();
                    *inPtr++ = *dPtr++;     // x
                    *inPtr   = *dPtr++;     // y
                    average_loss += trainer.train( *lPtr++ );
                }
            }
            average_loss /= DATA_COUNT * MAX_ITERATION;
            log_info( "%lu: Average loss = %f", count, average_loss );
        } while( average_loss > TARGET_LOSS );
        
        log_info( "Test Circle - End" );
        return true;
    }
    
}  // tfs namespace


bool
testCircle( void ) {
    return tfs::localTestCircle();
}
