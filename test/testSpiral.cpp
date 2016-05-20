//
//  testSpiral.cpp
//  Example inspired by http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
//
//  Created by Barrett Davis on 5/9/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "DnnTrainerSGD.h"
#include "Error.h"
#include "testSpiral.hpp"
#include "Utility.h"

namespace tfs {
    
    static void
    spiralSetUpData( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, const int count ) {
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = ii / ( count * 5.0 ) + random( -0.1, 0.1 );
            const DNN_NUMERIC aa = 1.25 * ii / count * 2.0 * M_PI;
            const DNN_NUMERIC t0 = aa + random( -0.1, 0.1 );
            const DNN_NUMERIC t1 = aa + random( -0.1, 0.1 ) + M_PI;
            
            DNN_NUMERIC xx = rr * sin( t0 );
            DNN_NUMERIC yy = rr * cos( t0 );
            data.push_back( xx );       // x,y pair
            data.push_back( yy );
            label.push_back( 0 );       // label
            
            xx = rr * sin( t1 );
            yy = rr * cos( t1 );
            data.push_back( xx );       // x,y pair
            data.push_back( yy );
            label.push_back( 1 );       // label
        }
        return;
    }
    
    static bool
    setupDnn( Dnn &dnn ) {
        if( !dnn.addLayerInput( 1, 1, 2 )) {                // Input layer a single x, y data point.
            return log_error( "Cannot add Input layer" );
        }
        if( !dnn.addLayerFullyConnected( 8 )) {             // 8 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                          // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerFullyConnected( 6 )) {             // 6 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                          // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerFullyConnected( 2 )) {             // 2 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                          // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerSoftmax( )) {                      // Output classifier, 2 classes.
            return log_error( "Cannot add Softmax layer" );
        }
        dnn.initialize();                                   // Randomize the weights.
        const unsigned long count = dnn.count();
        log_info( "We have set up %lu layers", count );
        return true;
    }
    
    static bool
    localTestSpiral( void ) {
        log_info( "Test Spiral - Start" );

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
        
        spiralSetUpData( data, label, 100 );
        
        Matrix *input = trainer.getMatrixInput();       // get the input matrix: x,y pair Matrix( 1, 1, 2 )
        if( input == 0 ) {
            return log_error( "Unable to obtain input matrix." );
        }
        DMatrix output( 1, 1, 1 );   // label ( 0 or 1 )
        
        DNN_INTEGER *outPtr = output.data();

        const unsigned long MAX_ITERATION = 200;
        const unsigned long DATA_COUNT    = label.size();
        const DNN_NUMERIC   TARGET_LOSS   = 0.0001;
        DNN_NUMERIC average_loss = 0.0;
        do {
            average_loss = 0.0;
            for( unsigned long ii = 0; ii < MAX_ITERATION; ii++ ) {
                const DNN_NUMERIC *dPtr = data.data();
                const DNN_INTEGER *lPtr = label.data();
                const DNN_INTEGER *ePtr = lPtr + DATA_COUNT;
                
                while( lPtr < ePtr ) {
                    DNN_NUMERIC *inPtr = input->data();
                    *inPtr++ = *dPtr++;     // x
                    *inPtr++ = *dPtr++;     // y
                    *outPtr  = *lPtr++;     // label
                    average_loss += trainer.train( output );
                }
            }
            average_loss /= DATA_COUNT * MAX_ITERATION;
            log_info( "Average loss = %f", average_loss );
        } while( average_loss > TARGET_LOSS );

        log_info( "Test Spiral - End" );
        return true;
    }
    
}  // tfs namespace


bool
testSpiral( void ) {
    return tfs::localTestSpiral();
}