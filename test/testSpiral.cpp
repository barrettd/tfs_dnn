//
//  testSpiral.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/9/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "Dnn.h"
#include "Error.h"
#include "testSpiral.hpp"

namespace tfs {
    
    static bool
    localTestSpiral( void ) {
        //        layer_defs = [];
        //        layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
        //        layer_defs.push({type:'fc', num_neurons:8, activation: 'tanh'});
        //        layer_defs.push({type:'fc', num_neurons:6, activation: 'tanh'});
        //        layer_defs.push({type:'fc', num_neurons:2, activation: 'tanh'});
        //        layer_defs.push({type:'softmax', num_classes:2});
        //
        //        net = new convnetjs.Net();
        //        net.makeLayers(layer_defs);
        //
        //        trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.1, batch_size:10, l2_decay:0.001});

        log_info( "Test Spiral - Start" );
        
//        function spiral_data() {
//            data = [];
//            labels = [];
//            var n = 100;
//            for(var i=0;i<n;i++) {
//                var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
//                var t = 1.25*i/n*2*Math.PI + convnetjs.randf(-0.1, 0.1);
//                data.push([r*Math.sin(t), r*Math.cos(t)]);
//                labels.push(1);
//            }
//            for(var i=0;i<n;i++) {
//                var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
//                var t = 1.25*i/n*2*Math.PI + Math.PI + convnetjs.randf(-0.1, 0.1);
//                data.push([r*Math.sin(t), r*Math.cos(t)]);
//                labels.push(0);
//            }
//            N = data.length;
//        }

        

        log_info( "Test Spiral - End" );
        return true;
    }
    
}  // tfs namespace


bool
testSpiral( void ) {
    return tfs::localTestSpiral();
}