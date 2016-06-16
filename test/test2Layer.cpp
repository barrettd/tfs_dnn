//
//  test2Layer.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/27/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "DnnBuilder.h"
#include "DnnTrainerSGD.h"
#include "Error.h"
#include "test2Layer.hpp"

namespace tfs {
    
    static bool
    localTest2Layer( void ) {
        log_info( "Test 2 Layer - Start" );
        Dnn dnn;
        DnnBuilder builder( dnn, ACTIVATION_RELU );
        builder.addLayerInput( 1, 1, 2 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerSoftmax( 2 );
        dnn.initialize();                                   // Randomize the weights.

        DNN_NUMERIC *input  = dnn.getDataInput();
        DNN_NUMERIC *output = dnn.getDataOutput();
        if( input == 0 || output == 0 ) {
            return log_error( "bad pointers from dnn" );
        }
        *input++ =  0.3;                // Send a data point (0.3,-0.5) forward through the network.
        *input   = -0.5;
        if( !dnn.predict()) {
            log_error( "Problem with forward propagation through the network." );
        }
        DNN_NUMERIC probability1 = *output;
        
        DnnTrainerSGD trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.l2Decay(      0.001 );
        
        trainer.train( 0.0 );
        
        dnn.predict();
        
        DNN_NUMERIC probability2 = *output;
        
        if( probability2 <= probability1 ) {
            return log_error( "Probablity did not change for the better: %f to %f", probability1, probability2 );
        }
        
        log_info( "Test 2 Layer - End" );
        return true;
    }


}  // tfs namespace


bool
test2Layer( void ) {
    return tfs::localTest2Layer();
}
