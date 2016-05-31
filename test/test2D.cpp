//
//  test2D.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/31/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "DnnBuilder.h"
#include "DnnTrainerSGD.h"
#include "Error.h"
#include "test2D.hpp"
#include "Utility.h"

namespace tfs {
    
    static void
    setData( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, DNN_NUMERIC xx, DNN_NUMERIC yy, DNN_INTEGER value ) {
        data.push_back( xx );       // x,y pair
        data.push_back( yy );
        label.push_back( value );   // label
        return;
    }

    static void
    spiralSetUpDataSimple( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label ) {
        setData( data, label, -0.4326, 1.1909,   1 );
        setData( data, label,  3.0,      4.0,    1 );
        setData( data, label,  0.1253 , -0.0376, 1 );
        setData( data, label,  0.2877 ,  0.3273, 1 );
        setData( data, label, -1.1465 ,  0.1746, 1 );
        setData( data, label,  1.8133 ,  1.0139, 0 );
        setData( data, label,  2.7258 ,  1.0668, 0 );
        setData( data, label,  1.4117 ,  0.5593, 0 );
        setData( data, label,  4.1832 ,  0.3044, 0 );
        setData( data, label,  1.8636 ,  0.1677, 0 );
        setData( data, label,  0.5 ,     3.2,    1 );
        setData( data, label,  0.8 ,     3.2,    1 );
        setData( data, label,  1.0 ,    -2.2,    1 );
        return;
    }
    
    static void
    spiralSetUpDataCircle( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, const int count ) {
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = random( 0.0, 2.0 );
            const DNN_NUMERIC tt = random( 0.0, 2.0 * M_PI );
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 0 );
        }
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = random( 3.0, 5.0 );
            const DNN_NUMERIC tt = 2.0 * M_PI * ii / count;
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 1 );
        }
        return;
    }
    
    static void
    spiralSetUpDataSpiral( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, const int count ) {
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = ii / count * 5.0 + random( -0.1, 0.1 );
            const DNN_NUMERIC tt = 1.25 * ii / count * 2.0 * M_PI + random( -0.1, 0.1 );
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 0 );
        }
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = ii / count * 5.0 + random( -0.1, 0.1 );
            const DNN_NUMERIC tt = 1.25 * ii / count * 2.0 * M_PI + random( -0.1, 0.1 ) + M_PI;
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 1 );
        }
        return;
    }
    
    static bool
    setupDnn( Dnn &dnn ) {                                      // Same DNN for all 2d tests.
        DnnBuilder builder( dnn, ACTIVATION_TANH );
        if( !builder.addLayerInput( 1, 1, 2 )) {                // Input layer a single x, y data point.
            return log_error( "Cannot add Input layer" );
        }
        if( !builder.addLayerFullyConnected( 6 )) {             // 6 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !builder.addLayerFullyConnected( 2 )) {             // 2 Neurons
            return log_error( "Cannot add Fully Connected layer" );
        }
        if( !builder.addLayerSoftmax( 2 )) {                    // Output classifier, 2 classes.
            return log_error( "Cannot add Softmax layer" );
        }
        dnn.initialize();                                       // Randomize the weights.
        const unsigned long count = dnn.count();
        log_info( "We have set up %lu layers", count );
        return true;
    }
    
    static bool
    localTest2d( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label ) {
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerSGD trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.momentum(     0.1   );
        trainer.batchSize(   10     );
        trainer.l2Decay(      0.001 );
        
        Matrix *input = trainer.getMatrixInput();       // get the input matrix: x,y pair Matrix( 1, 1, 2 )
        if( input == 0 ) {
            return log_error( "Unable to obtain input matrix." );
        }
        const DNN_NUMERIC *dPtr = data.data();
        if( dPtr == 0 ) {
            return log_error( "Unable to obtain input data." );
        }
        const DNN_INTEGER *lPtr = label.data();
        if( lPtr == 0 ) {
            return log_error( "Unable to obtain input labels." );
        }
        const unsigned long DATA_COUNT = label.size();
        if( DATA_COUNT < 1 ) {
            return log_error( "Label data size < 1" );
        }
        const DNN_INTEGER *ePtr = lPtr + DATA_COUNT;
        
        const unsigned long MAX_ITERATION = 200;
        const DNN_NUMERIC   TARGET_LOSS   = 0.005;
        unsigned long count = 0;
        DNN_NUMERIC average_loss = 0.0;
        do {
            count++;
            average_loss = 0.0;
            for( unsigned long ii = 0; ii < MAX_ITERATION; ii++ ) {
                dPtr = data.data();
                lPtr = label.data();
                
                while( lPtr < ePtr ) {
                    DNN_NUMERIC *inPtr = input->data();
                    *inPtr++ = *dPtr++;     // x
                    *inPtr   = *dPtr++;     // y
                    average_loss += trainer.train( *lPtr++ );
                }
            }
            average_loss /= DATA_COUNT * MAX_ITERATION;
            log_info( "%lu: Average loss = %f", count, average_loss );
        } while( average_loss > TARGET_LOSS );
        
        return true;
    }
    
    static bool
    localTestSpiral( void ) {
        log_info( "Test Spiral - Start" );
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        spiralSetUpDataSpiral( data, label, 100 );
        
        localTest2d( data, label );
        
        log_info( "Test Spiral - End" );
        return true;
    }
    
    static bool
    localTestCircle( void ) {
        log_info( "Test Circle - Start" );
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        spiralSetUpDataCircle( data, label, 50 );
        
        localTest2d( data, label );

        log_info( "Test Circle - End" );
        return true;
    }

    static bool
    localTestSimple( void ) {
        log_info( "Test Simple - Start" );
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        spiralSetUpDataSimple( data, label );
        
        localTest2d( data, label );
        
        log_info( "Test Simple - End" );
        return true;
    }

}  // tfs namespace

bool
testSimple( void ) {
    return tfs::localTestSimple();
}

bool
testCircle( void ) {
    return tfs::localTestCircle();
}

bool
testSpiral( void ) {
    return tfs::localTestSpiral();
}
