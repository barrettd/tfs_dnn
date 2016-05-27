//
//  testFullyConnected.cpp
//
//  Created by Barrett Davis on 5/20/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "DnnLayers.h"
#include "DnnTrainerSGD.h"
#include "Error.h"
#include "testFullyConnected.hpp"

namespace tfs {
    

    static bool
    setupDnn( Dnn &dnn ) {
        // --------------------------------------------------------------------
        // TODO: check
        // --------------------------------------------------------------------
        if( !dnn.addLayerInput( 1, 1, 2, true )) {          // Input layer a single x, y data point.
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
        // --------------------------------------------------------------------
        // ok 25 May 2016
        // --------------------------------------------------------------------
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
            const DNN_NUMERIC aa = random( -1.0, 1.0 );
            const DNN_NUMERIC bb = random( -1.0, 1.0 );
            *data++ = aa;
            *data   = bb;
            if( !dnn.forward()) {
                return log_error( "Error during feed forward" );
            }
            Matrix previous = Matrix( *output, true );          // Copy output
            const DNN_INTEGER index = floor( random( 3.0 ));    // Index 0 to 2.
            if( index < 0 || index > 3 ) {
                return log_error( "Index out of range: %ld", index );
            }
            trainer.train( index );
            data = input->data();
            if( *data++ != aa || *data != bb ) {
                return log_error( "Input data changed during training." );
            }
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
    localTestGradiant( DnnTrainerSGD &trainer, Dnn &dnn ) {
        // --------------------------------------------------------------------
        // Check the gradiant.
        // --------------------------------------------------------------------
        Matrix *input  = dnn.getMatrixInput();          // x,y pair
        Matrix *output = dnn.getMatrixOutput();         //
        if( input == 0 ) {
            return log_error( "Input matrix is null" );
        }
        const unsigned long inputCount = input->count();
        if( inputCount != 2 ) {
            return log_error( "Input matrix expected count = 2, was %lu", inputCount );
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
        const DNN_INTEGER index = floor( random( 3.0 ));    // Index 0 to 2.
        if( index < 0 || index > 3 ) {
            return log_error( "Index out of range: %ld", index );
        }
        trainer.train( index );         // Calulate gradients, propagate up to input layer.
        
        DnnLayerInput *inputLayer = dnn.getLayerInput();
        if( inputLayer == 0 ) {
            return log_error( "Input layer is null" );
        }
        Matrix *inputDw = inputLayer->outDw();
        if( inputDw == 0 ) {
            return log_error( "Input gradiant is null" );
        }
        DNN_NUMERIC *ww = input->data();
        if( ww == 0 ) {
            return log_error( "ww == 0" );
        }
        DNN_NUMERIC *dw = inputDw->data();
        if( dw == 0 ) {
            return log_error( "dw == 0" );
        }
        
        const DNN_NUMERIC delta = 0.000001;
        
        for( unsigned long ii = 0; ii < inputCount; ii++ ) {
            const DNN_NUMERIC grad_analytic = dw[ii];
            
            const DNN_NUMERIC xold = ww[ii];
            ww[ii] += delta;
            
            const DNN_NUMERIC c0 = dnn.getCostLoss( index );
            ww[ii] -= ( 2.0 * delta );
            
            const DNN_NUMERIC c1 = dnn.getCostLoss( index );
            ww[ii] = xold; // reset
            
            const DNN_NUMERIC grad_numeric = (c0 - c1)/(2 * delta);
            const DNN_NUMERIC rel_error = fabs(grad_analytic - grad_numeric)/fabs(grad_analytic + grad_numeric);
//            log_info( "%d, numeric: %f, analytic: %f, error: %f", ii, grad_numeric, grad_analytic, rel_error );
            if( rel_error >= 1.0e-2 ) {
                log_error( "rel_error = %f", rel_error );
            }
        }
        
        
        return true;
    }
    
    static bool
    localTestFullyConnected( void ) {
        // --------------------------------------------------------------------
        // ok: 24 May 2016
        // --------------------------------------------------------------------
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
        if( !localTestGradiant( trainer, dnn )) {
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
