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
    spiralSetUpData( std::vector< DNN_NUMERIC > &data, std::vector< DNN_NUMERIC > &label, const int count ) {
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = ii / ( count * 5.0 ) + random( -0.1, 0.1 );
            const DNN_NUMERIC aa = 1.25 * ii / count * 2.0 * M_PI;
            const DNN_NUMERIC t0 = aa + random( -0.1, 0.1 );
            const DNN_NUMERIC t1 = aa + random( -0.1, 0.1 ) + M_PI;
            
            DNN_NUMERIC xx = rr * sin( t0 );
            DNN_NUMERIC yy = rr * cos( t0 );
            data.push_back( xx );       // x,y pair
            data.push_back( yy );
            label.push_back( 0.0 );     // label
            
            xx = rr * sin( t1 );
            yy = rr * cos( t1 );
            data.push_back( xx );       // x,y pair
            data.push_back( yy );
            label.push_back( 1.0 );     // label
        }
        return;
    }
    
    static bool
    setupDnn( Dnn &dnn ) {
        if( !dnn.addLayerInput( 1, 1, 2 )) {             // Input layer a single x, y data point.
            return log_error( "Cannot add Input layer" );
        }
        if( !dnn.addLayerFullyConnected( 8 )) {          // 8 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                       // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerFullyConnected( 6 )) {          // 6 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                       // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerFullyConnected( 2 )) {          // 2 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !dnn.addLayerTanh()) {                       // Activation function for fully connected layer.
            return log_error( "Cannot add Tanh Activation layer" );
        }
        if( !dnn.addLayerSoftmax( 2 )) {                  // Output classifier, 2 classes.
            return log_error( "Cannot add Softmax layer" );
        }
        const unsigned long count = dnn.count();
        log_info( "We have set up %lu layers", count );
        return true;
    }
    
    static bool
    localTestSpiral( void ) {
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_NUMERIC > label;   // binary labels.
        
        spiralSetUpData( data, label, 100 );
        
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerSGD trainer( &dnn );
        
        
        
        

        
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