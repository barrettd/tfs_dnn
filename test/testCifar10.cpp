//
//  testCifar10.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/9/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "Dnn.h"
#include "Error.h"
#include "testCifar10.hpp"


namespace tfs {
    
    static bool
    localTestCifar10( void ) {
        // Taking an example from ConvNetJS cifar10: http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
        //        layer_defs = [];
        //        layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
        //        layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
        //        layer_defs.push({type:'pool', sx:2, stride:2});
        //        layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
        //        layer_defs.push({type:'pool', sx:2, stride:2});
        //        layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
        //        layer_defs.push({type:'pool', sx:2, stride:2});
        //        layer_defs.push({type:'softmax', num_classes:10});
        //
        //        net = new convnetjs.Net();
        //        net.makeLayers(layer_defs);
        //
        //        trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});
        log_info( "Test Cifar10 - Start" );
        Dnn *dnn = new Dnn();
        if( !dnn->addLayerInput( 32, 32, 3 )) {             // Input layer for 32x32 RGB image
            return log_error( "Cannot add Input layer" );
        }
        // Convolution / Pool set:
        if( !dnn->addLayerConvolution( 5, 16, 1, 2 )) {     // 16 5x5 filters for convolution
            return log_error( "Cannot add Convolution layer" );
        }
        if( !dnn->addLayerRectifiedLinearUnit()) {          // Activation function for previous layer.
            return log_error( "Cannot add ReLu layer" );
        }
        if( !dnn->addLayerPool( 2, 2 )) {
            return log_error( "Cannot add Pool layer" );
        }
        // Convolution / Pool set:
        if( !dnn->addLayerConvolution( 5, 20, 1, 2 )) {     // 20 5x5 filters for convolution
            return log_error( "Cannot add Convolution layer" );
        }
        if( !dnn->addLayerRectifiedLinearUnit()) {          // Activation function for previous layer.
            return log_error( "Cannot add ReLu layer" );
        }
        if( !dnn->addLayerPool( 2, 2 )) {
            return log_error( "Cannot add Pool layer" );
        }
        // Convolution / Pool set:
        if( !dnn->addLayerConvolution( 5, 20, 1, 2 )) {     // 20 5x5 filters for convolution
            return log_error( "Cannot add Convolution layer" );
        }
        if( !dnn->addLayerRectifiedLinearUnit()) {          // Activation function for previous layer.
            return log_error( "Cannot add ReLu layer" );
        }
        if( !dnn->addLayerPool( 2, 2 )) {
            return log_error( "Cannot add Pool layer" );
        }
        // Fully connected layer with softmax:
        if( !dnn->addLayerFullyConnected( 1, 1, 10 )) {
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn->addLayerSoftmax( 10 )) {                  // Output classifier
            return log_error( "Cannot add Softmax layer" );
        }
        const unsigned long count = dnn->count();
        log_info( "We have %lu layers", count );
        log_info( "Test Cifar10 - End" );
        return true;
    }
    
}  // tfs namespace

bool
testCifar10( void ) {
    return tfs::localTestCifar10();
}
