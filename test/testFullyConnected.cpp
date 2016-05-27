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
    
//    it("should compute correct gradient at data", function() {
//        
//        // here we only test the gradient at data, but if this is
//        // right then that's comforting, because it is a function
//        // of all gradients above, for all layers.
//        
//        var x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
//        var gti = Math.floor(Math.random() * 3); // ground truth index
//        trainer.train(x, gti); // computes gradients at all layers, and at x
//        
//        var delta = 0.000001;
//        
//        for(var i=0;i<x.w.length;i++) {
//            
//            var grad_analytic = x.dw[i];
//            
//            var xold = x.w[i];
//            x.w[i] += delta;
//            var c0 = net.getCostLoss(x, gti);
//            x.w[i] -= 2*delta;
//            var c1 = net.getCostLoss(x, gti);
//            x.w[i] = xold; // reset
//            
//            var grad_numeric = (c0 - c1)/(2 * delta);
//            var rel_error = Math.abs(grad_analytic - grad_numeric)/Math.abs(grad_analytic + grad_numeric);
//            console.log(i + ': numeric: ' + grad_numeric + ', analytic: ' + grad_analytic + ' => rel error ' + rel_error);
//            expect(rel_error).toBeLessThan(1e-2);
//            
//        }
//    });
    
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
        DNN_NUMERIC loss = trainer.train( index );         // Calulate gradients, propagate up to input layer.
        
        DnnLayerInput *inputLayer = dnn.getLayerInput();
        if( inputLayer == 0 ) {
            return log_error( "Input layer is null" );
        }
        Matrix *weights = inputLayer->weights();            // This is the Matrix that contains the data() above.
        if( weights == 0 ) {
            return log_error( "Input weights are null" );
        }
        Matrix *gradiant = inputLayer->gradiant();
        if( gradiant == 0 ) {
            return log_error( "Input gradiant is null" );
        }
        DNN_NUMERIC *ww = weights->data();
        if( ww == 0 ) {
            return log_error( "ww == 0" );
        }
        DNN_NUMERIC *dw = gradiant->data();
        if( dw == 0 ) {
            return log_error( "dw == 0" );
        }
        
        const DNN_NUMERIC delta = 0.000001;
        
        for( unsigned long ii = 0; ii < inputCount; ii++ ) {
            DNN_NUMERIC grad_analytic = dw[ii];
            
            DNN_NUMERIC xold = ww[ii];
            ww[ii] += delta;
            
            DNN_NUMERIC c0 = loss;
            
            ww[ii] -= ( 2.0 * delta );
            //            var c1 = net.getCostLoss(x, gti);
            //            x.w[i] = xold; // reset
            //
            //            var grad_numeric = (c0 - c1)/(2 * delta);
            //            var rel_error = Math.abs(grad_analytic - grad_numeric)/Math.abs(grad_analytic + grad_numeric);
            //            console.log(i + ': numeric: ' + grad_numeric + ', analytic: ' + grad_analytic + ' => rel error ' + rel_error);
            //            expect(rel_error).toBeLessThan(1e-2);
           
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
