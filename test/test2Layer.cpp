//
//  test2Layer.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/27/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "DnnBuilder.h"
#include "Error.h"
#include "test2Layer.hpp"

namespace tfs {
    
//    // species a 2-layer neural network with one hidden layer of 20 neurons
//    var layer_defs = [];
//    // input layer declares size of input. here: 2-D data
//    // ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
//    // then the first two dimensions (sx, sy) will always be kept at size 1
//    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
//    // declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
//    layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
//    // declare the linear classifier on top of the previous hidden layer
//    layer_defs.push({type:'softmax', num_classes:10});
//    
//    var net = new convnetjs.Net();
//    net.makeLayers(layer_defs);
//    
//    // forward a random data point through the network
//    var x = new convnetjs.Vol([0.3, -0.5]);
//    var prob = net.forward(x);
//    
//    // prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
//    console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101
//    
//    var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
//    trainer.train(x, 0); // train the network, specifying that x is class zero
//    
//    var prob2 = net.forward(x);
//    console.log('probability that x is class 0: ' + prob2.w[0]);
//    // now prints 0.50374, slightly higher than previous 0.50101: the networks
//    // weights have been adjusted by the Trainer to give a higher probability to
//    // the class we trained the network with (zero)
    
    static bool
    localTest2Layer( void ) {
        log_info( "Test 2 Layer - Start" );
        Dnn dnn;
        DnnBuilder builder( dnn, ACTIVATION_RELU );
        builder.addLayerInput( 1, 1, 2 );
        builder.addLayerFullyConnected( 20 );
        builder.addLayerSoftmax( 10 );
        dnn.initialize();                                   // Randomize the weights.

        

        log_info( "Test 2 Layer - Start" );
        return true;
    }


}  // tfs namespace


bool
test2Layer( void ) {
    return tfs::localTest2Layer();
}
