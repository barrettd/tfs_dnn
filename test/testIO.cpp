//
//  testIO.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/31/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "DnnBuilder.h"
#include "DnnTrainerAdaDelta.h"
#include "DnnStream.h"
#include "Error.h"
#include "testIO.hpp"

namespace tfs {
    
    static bool
    setupDnn1( Dnn &dnn ) {                                      // Same DNN for all 2d tests.
        DnnBuilder builder( dnn, ACTIVATION_TANH );
        if( !builder.addLayerInput( 1, 1, 2 )) {                // Input layer a single x, y data point.
            return log_error( "Cannot add Input layer" );
        }
        if( !builder.addLayerFullyConnected( 8 )) {             // 8 Neurons
            return log_error( "Cannot add Fully Connected layer" );
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
        const unsigned long expected_count = 9;
        const unsigned long count = dnn.count();
        if( count != expected_count ) {
            return log_error( "We have set up %lu layers, expected %lu", count, expected_count );
        }
        return true;
    }
    
    static bool
    setupDnn2( Dnn &dnn ) {
        DnnBuilder builder( dnn, ACTIVATION_RELU );
        if( !builder.addLayerInput( 159, 119, 3 )) {            // Input layer: 159x119x3
            return log_error( "Cannot add Input layer" );
        }
        if( !builder.addLayerConvolution( 5, 16, 1, 2 )) {      // sx:5, filters:16, stride:1, pad:2
            return log_error( "Cannot add Convolution layer" );
        }
        if( !builder.addLayerPool( 2, 2 )) {                    // sx:2, stride:2
            return log_error( "Cannot add Pool layer" );
        }
        if( !builder.addLayerConvolution( 5, 20, 1, 2 )) {      // sx:5, filters:20, stride:1, pad:2
            return log_error( "Cannot add Convolution layer" );
        }
        if( !builder.addLayerPool( 2, 2 )) {                    // sx:2, stride:2
            return log_error( "Cannot add Pool layer" );
        }
        if( !builder.addLayerConvolution( 5, 20, 1, 2 )) {      // sx:5, filters:20, stride:1, pad:2
            return log_error( "Cannot add Convolution layer" );
        }
        if( !builder.addLayerPool( 2, 2 )) {                    // sx:2, stride:2
            return log_error( "Cannot add Pool layer" );
        }
        if( !builder.addLayerSoftmax( 10 )) {                   // Output classifier, 10 classes (digits 0 to 9)
            return log_error( "Cannot add Softmax layer" );
        }
        dnn.initialize();                                       // Randomize the weights.
        const unsigned long count = dnn.count();
        log_info( "We have set up %lu layers", count );
        return true;
    }
    
    static void
    setData( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, DNN_NUMERIC xx, DNN_NUMERIC yy, DNN_INTEGER value ) {
        data.push_back( xx );       // x,y pair
        data.push_back( yy );
        label.push_back( value );   // label
        return;
    }

    static void
    setUpDataSpiral( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, const DNN_NUMERIC count ) {
        DNN_NUMERIC val = 0.0;
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = val / count * 5.0 + random( -0.1, 0.1 );
            const DNN_NUMERIC tt = 1.25 * val / count * 2.0 * M_PI + random( -0.1, 0.1 );
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 0 );
            val += 1.0;
        }
        val = 0.0;
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = val / count * 5.0 + random( -0.1, 0.1 );
            const DNN_NUMERIC tt = 1.25 * val / count * 2.0 * M_PI + random( -0.1, 0.1 ) + M_PI;
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 1 );
            val += 1.0;
        }
        return;
    }
    
    DNN_NUMERIC
    localTest2d( Dnn &dnn, std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label ) {
        DnnTrainerAdaDelta trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.momentum(     0.1   );
        trainer.batchSize(   10     );
        trainer.l2Decay(      0.001 );
        
        Matrix *input = trainer.getMatrixInput();       // get the input matrix: x,y pair Matrix( 1, 1, 2 )
        if( input == 0 ) {
            log_error( "Unable to obtain input matrix." );
            return 0.0;
        }
        const DNN_NUMERIC *dPtr = data.data();
        if( dPtr == 0 ) {
            log_error( "Unable to obtain input data." );
            return 0.0;
        }
        const DNN_INTEGER *lPtr = label.data();
        if( lPtr == 0 ) {
            log_error( "Unable to obtain input labels." );
            return 0.0;
        }
        const unsigned long DATA_COUNT = label.size();
        if( DATA_COUNT < 1 ) {
            log_error( "Label data size < 1" );
            return 0.0;
        }
        const DNN_INTEGER *ePtr = lPtr + DATA_COUNT;
        
        const unsigned int MAX_ITERATION = 200;
        const DNN_NUMERIC   TARGET_LOSS  = 0.05;
        unsigned long count = 0;
        DNN_NUMERIC average_loss = 0.0;
        do {
            count++;
            average_loss = 0.0;
            for( unsigned int ii = 0; ii < MAX_ITERATION; ii++ ) {
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
        } while( average_loss > TARGET_LOSS );
//        log_info( "Average loss = %f/%f. Count = %lu", average_loss, TARGET_LOSS, count );
        return average_loss;
    }

    static bool
    writeDnn( const Dnn &dnn, const char *fileName ) {
        OutDnnStream outStream( fileName );
        const bool rc = outStream.writeDnn( dnn );
        if( !rc ) {
            log_error( "DNN write failed" );
        }
        outStream.close();
        return rc;
    }

    static bool
    localTestIO1( void ) {
        log_info( "Test I/O (1) - Start" );
        Dnn dnn;
        if( !setupDnn1( dnn )) {
            return false;
        }
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        setUpDataSpiral( data, label, 100.0 );
        
        localTest2d( dnn, data, label );
        
        OutDnnStream outStream( "dnn_test1.dnn" );
        if( !outStream.writeDnn( dnn )) {
            log_error( "DNN write failed" );
        }
        outStream.close();
        
        InDnnStream inStream( "dnn_test1.dnn" );
        Dnn *otherDnn = inStream.readDnn( true );
        inStream.close();
        if( otherDnn == 0 ) {
            return log_error( "Unable to read a DNN" );
        }
        DNN_NUMERIC orginal_loss = localTest2d(       dnn, data, label );
        DNN_NUMERIC other_loss   = localTest2d( *otherDnn, data, label );
        delete otherDnn;
        
        if( orginal_loss != other_loss ) {
            log_error( "Training loss for new dnn does not match orginial: %f/%f", other_loss, orginal_loss );
        }
        log_info( "Test I/O (1) - End" );
        return true;
    }
    
    static bool
    localTestIO2( void ) {
        log_info( "Test I/O (2) - Start" );
        Dnn dnn;
        if( !setupDnn2( dnn )) {
            return false;
        }
        writeDnn( dnn, "dnn_test2.dnn" );
        log_info( "Test I/O (2) - End" );
        return true;
    }

    
}  // tfs namespace



bool
testIO( void ) {
    return tfs::localTestIO1() &&  tfs::localTestIO2();
}
