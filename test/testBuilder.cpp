//
//  testBuilder.cpp
//  Test that the DnnBuilder is making correct layer stacks.
//  Using some of the layer definitions from Andrej Karpathy's JavaScript demos as examples.
//
//  Created by Barrett Davis on 5/26/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "DnnBuilder.h"
#include "Error.h"

#include "testBuilder.hpp"

namespace tfs {

    static bool
    localTestBuilder01( void ) {
        //var layer_defs = [];
        //layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
        //layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
        //layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
        //layer_defs.push({type:'softmax', num_classes:3});
        //net.makeLayers(layer_defs);
        Dnn dnn;
        DnnBuilder builder( dnn, ACTIVATION_TANH );
        builder.addLayerInput( 1, 1, 2 );
        builder.addLayerFullyConnected( 5 );
        builder.addLayerFullyConnected( 5 );
        builder.addLayerSoftmax( 3 );
        dnn.initialize();                                   // Randomize the weights.
        const unsigned long count = dnn.count();
        if( count != 7 ) {
            return log_error( "Expected 7 layers, received %lu", count );
        }
        return true;
    }

    static bool
    localTestBuilder02( void ) {
        //layer_defs = [];
        //layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
        //layer_defs.push({type:'fc', num_neurons:6, activation: 'tanh'});
        //layer_defs.push({type:'fc', num_neurons:2, activation: 'tanh'});
        //layer_defs.push({type:'softmax', num_classes:2});
        //net.makeLayers(layer_defs);
        Dnn dnn;
        DnnBuilder builder( dnn, ACTIVATION_TANH );
        builder.addLayerInput( 1, 1, 2 );
        builder.addLayerFullyConnected( 6 );
        builder.addLayerFullyConnected( 2 );
        builder.addLayerSoftmax( 2 );
        dnn.initialize();                                   // Randomize the weights.
        const unsigned long count = dnn.count();
        if( count != 7 ) {
            return log_error( "Expected 7 layers, received %lu", count );
        }
        return true;
    }

    
    static bool
    localTestBuilder03( void ) {
        //layer_defs = [];
        //layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2}); // 2 inputs: x, y
        //layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
        //layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
        //layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
        //layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
        //layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
        //layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
        //layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
        //layer_defs.push({type:'regression', num_neurons:3}); // 3 outputs: r,g,b
        //net.makeLayers(layer_defs);
        Dnn dnn;
        DnnBuilder builder( dnn, ACTIVATION_RELU );
        builder.addLayerInput( 1, 1, 2 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerRegression( 2 );
        dnn.initialize();                                   // Randomize the weights.
        const unsigned long count = dnn.count();
        if( count != 17 ) {
            return log_error( "Expected 17 layers, received %lu", count );
        }
        return true;
    }
    
    static bool
    localTestBuilder04( void ) {
        //layer_defs = [];
        //layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
        //layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
        //layer_defs.push({type:'pool', sx:2, stride:2});
        //layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
        //layer_defs.push({type:'pool', sx:2, stride:2});
        //layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
        //layer_defs.push({type:'pool', sx:2, stride:2});
        //layer_defs.push({type:'softmax', num_classes:10});
        //net = new convnetjs.Net();
        Dnn dnn;
        DnnBuilder builder( dnn, ACTIVATION_RELU );
        builder.addLayerInput( 32, 32, 3 );
        builder.addLayerConvolution( 5, 16, 1, 2 );
        builder.addLayerPool( 2, 2 );
        builder.addLayerConvolution( 5, 20, 1, 2 );
        builder.addLayerPool( 2, 2 );
        builder.addLayerConvolution( 5, 20, 1, 2 );
        builder.addLayerPool( 2, 2 );
        builder.addLayerSoftmax( 10 );
        dnn.initialize();                                   // Randomize the weights.
        const unsigned long count = dnn.count();
        // TODO: continue work here.
        if( count != 12 ) {
            return log_error( "Expected 12 layers, received %lu", count );
        }
        return true;
    }
    
    static bool
    localTestBuilder05( void ) {
        //layer_defs = [];
        //layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
        //layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
        //layer_defs.push({type:'pool', sx:2, stride:2});
        //layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
        //layer_defs.push({type:'pool', sx:2, stride:2});
        //layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
        //layer_defs.push({type:'pool', sx:2, stride:2});
        //layer_defs.push({type:'softmax', num_classes:10});
        //net = new convnetjs.Net();
        Dnn dnn;
        if( !dnn.addLayerInput( 32, 32, 3 )) {             // Input layer for 32x32 RGB image
            return log_error( "Cannot add Input layer" );
        }
        // Convolution / Activation / Pool set:
        if( !dnn.addLayerConvolution( 5, 16, 1, 2 )) {     // 16 5x5 filters for convolution
            return log_error( "Cannot add Convolution layer" );
        }
        if( !dnn.addLayerRectifiedLinearUnit()) {          // Activation function for previous layer.
            return log_error( "Cannot add ReLu layer" );
        }
        if( !dnn.addLayerPool( 2, 2 )) {
            return log_error( "Cannot add Pool layer" );
        }
        // Convolution / Activation / Pool set:
        if( !dnn.addLayerConvolution( 5, 20, 1, 2 )) {     // 20 5x5 filters for convolution
            return log_error( "Cannot add Convolution layer" );
        }
        if( !dnn.addLayerRectifiedLinearUnit()) {          // Activation function for previous layer.
            return log_error( "Cannot add ReLu layer" );
        }
        if( !dnn.addLayerPool( 2, 2 )) {
            return log_error( "Cannot add Pool layer" );
        }
        // Convolution / Activation / Pool set:
        if( !dnn.addLayerConvolution( 5, 20, 1, 2 )) {     // 20 5x5 filters for convolution
            return log_error( "Cannot add Convolution layer" );
        }
        if( !dnn.addLayerRectifiedLinearUnit()) {          // Activation function for previous layer.
            return log_error( "Cannot add ReLu layer" );
        }
        if( !dnn.addLayerPool( 2, 2 )) {
            return log_error( "Cannot add Pool layer" );
        }
        // Fully connected layer with softmax:
        if( !dnn.addLayerFullyConnected( 10 )) {               //  1, 1, 10
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerSoftmax()) {                          // Output classifier
            return log_error( "Cannot add Softmax layer" );
        }
        dnn.initialize();
        const unsigned long count = dnn.count();
        if( count != 12 ) {
            return log_error( "Expected 12 layers, received %lu", count );
        }
        return true;
    }


    static bool
    localTestBuilder( void ) {
        log_info( "Test Builder - Start" );
        if( !localTestBuilder01()) {
            return log_error( "builder 01 failed" );
        }
        if( !localTestBuilder02()) {
            return log_error( "builder 02 failed" );
        }
        if( !localTestBuilder03()) {
            return log_error( "builder 03 failed" );
        }
        if( !localTestBuilder04()) {
            return log_error( "builder 04 failed" );
        }
//        if( !localTestBuilder05()) {
//            return log_error( "builder 05 failed" );
//        }
        log_info( "Test Builder - End" );
        return true;
    }

}  // tfs namespace

bool
testBuilder( void ) {
    return tfs::localTestBuilder();
}
