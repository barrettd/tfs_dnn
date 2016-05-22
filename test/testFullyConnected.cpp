//
//  testFullyConnected.cpp
//
//  Created by Barrett Davis on 5/20/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "DnnTrainerSGD.h"
#include "Error.h"
#include "testFullyConnected.hpp"

namespace tfs {
    

    
    static bool
    setupDnn( Dnn &dnn ) {
        if( !dnn.addLayerInput( 1, 1, 2 )) {                // Input layer a single x, y data point.
            return log_error( "Cannot add Input layer" );
        }
        if( !dnn.addLayerFullyConnected( 5 )) {             // 5 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                          // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerFullyConnected( 5 )) {             // 5 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                          // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerFullyConnected( 3 )) {             // 3 Neurons - for softmax
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerSoftmax( )) {                      // Output classifier, 3 classes.
            return log_error( "Cannot add Softmax layer" );
        }
        dnn.initialize();                                   // Randomize the weights.
        const unsigned long count = dnn.count();
        if( count != 7 ) {
            return log_error( "Expected 7 layers, received %lu", count );
        }
        return true;
    }
    
    static bool
    closeTo( DNN_NUMERIC value, DNN_NUMERIC target ) {
        const DNN_NUMERIC difference = fabs( value - target );
        return difference < (pow( 10.0, -2.0 ) / 2.0);
    }
    
    static bool
    localCheckForwardPropagation( Dnn &dnn ) {
        Matrix *input  = dnn.getMatrixInput();
        Matrix *output = dnn.getMatrixOutput();
        if( input == 0 ) {
            return log_error( "Input matrix is null" );
        }
        if( input->count() != 2 ) {
            return log_error( "Input matrix expected count = 2, was %lu", input->count());
        }
        if( output == 0 ) {
            return log_error( "Output matrix is null" );
        }
        if( output->count() != 3 ) {
            return log_error( "Output matrix expected count = 3, was %lu", output->count());
        }
        DNN_NUMERIC *data = input->data();
        if( data == 0 ) {
            return log_error( "Input data is null" );
        }
        *data++ =  0.2;
        *data++ = -0.3;
        if( !dnn.forward()) {
            return log_error( "Error during feed forward" );
        }
        const DNN_NUMERIC *       expect    = output->data();
        const DNN_NUMERIC * const expectEnd = output->end();
        if( expect == 0 ) {
            return log_error( "Output expect is null" );
        }
        if( expectEnd == 0 ) {
            return log_error( "Output expectEnd is null" );
        }
        if( expect >= expectEnd ) {
            return log_error( "Output expect >= expectEnd" );
        }
        DNN_NUMERIC sum = 0;
        while( expect < expectEnd ) {
            if( *expect <= 0.0 ) {
                return log_error( "expect <= 0.0: %f", *expect );
            }
            if( *expect >= 1.0 ) {
                return log_error( "expect >= 1.0: %f", *expect );
            }
            sum += *expect++;       // Each is about 0.3...something...
        }
        if( !closeTo( sum, 1.0 )) {
            return log_error( "sum not close to 1.0, was %f", sum );
        }
        return true;
    }
    
    static bool
    localTestTrainer( DnnTrainerSGD &trainer, Dnn &dnn ) {
        for( int ii = 0; ii < 100; ii++ ) {
            Matrix *input  = dnn.getMatrixInput();          // x,y pair
            Matrix *output = dnn.getMatrixOutput();         //
            if( input == 0 ) {
                return log_error( "Input matrix is null" );
            }
            if( input->count() != 2 ) {
                return log_error( "Input matrix expected count = 2, was %lu", input->count());
            }
            if( output == 0 ) {
                return log_error( "Output matrix is null" );
            }
            if( output->count() != 3 ) {
                return log_error( "Output matrix expected count = 3, was %lu", output->count());
            }
            DNN_NUMERIC *data = input->data();
            if( data == 0 ) {
                return log_error( "Input data is null" );
            }
            *data++ = random( -1.0, 1.0 );
            *data   = random( -1.0, 1.0 );
            if( !dnn.forward()) {
                return log_error( "Error during feed forward" );
            }
            Matrix previous = Matrix( *output, true );          // Copy output
            const DNN_INTEGER index = floor( random( 3.0 ));    // Index 0 to 2.
            if( index < 0 || index > 3 ) {
                return log_error( "Index out of range: %ld", index );
            }
            trainer.train( index );
            if( !dnn.forward()) {
                return log_error( "Error during feed forward" );
            }
            const DNN_NUMERIC *prev = previous.dataReadOnly();
            const DNN_NUMERIC *curr = output->dataReadOnly();
            if( prev == 0 || curr == 0 ) {
                return log_error( "Null data" );
            }
            if( curr[index] <= prev[index] ) {
                return log_error( "Current weight (%f) should be larger than previous (%f). ii = %d", curr[index], prev[index], ii );
            }
            
        }
        return true;
    }
    
    static bool
    localTestFullyConnected( void ) {
        log_info( "Test Fully Connected - Start" );
        
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerSGD trainer(  &dnn );
        trainer.learningRate( 0.0001 );
        trainer.momentum(     0.0    );
        trainer.batchSize(    1      );
        trainer.l2Decay(      0.0    );

        if( !localCheckForwardPropagation( dnn )) {
            return false;
        }
        if( !localTestTrainer( trainer, dnn )) {
            return false;
        }
        log_info( "Test Fully Connected - End" );
        return true;
    }
    
}  // tfs namespace

bool
testFullyConnected( void ) {
    return tfs::localTestFullyConnected();
}
