//
//  test2D.cpp
//
//  Created by Barrett Davis on 5/31/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include <cmath>
#include "DnnBuilder.h"
#include "DnnTrainerAdaDelta.h"
#include "DnnTrainerAdam.h"
#include "DnnTrainerNesterov.h"
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
    setUpDataSimple( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label ) {
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
    setUpDataRandom( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, const int count ) {
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC xx = random( -3.0, 3.0 );
            const DNN_NUMERIC yy = random( -3.0, 3.0 );
            DNN_INTEGER value = 0;
            if( random( 0.0, 1.0 ) > 0.5 ) {
                value = 1;
            }
            setData( data, label, xx, yy, value );
        }
        return;
    }
    
    static void
    setUpDataCircle( std::vector< DNN_NUMERIC > &data, std::vector< DNN_INTEGER > &label, const int count ) {
        for( int ii = 0; ii < count; ii++ ) {
            const DNN_NUMERIC rr = random( 0.0, 2.0 );
            const DNN_NUMERIC tt = random( 0.0, 2.0 * M_PI );
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 0 );
        }
        DNN_NUMERIC val = 0.0;
        for( int ii = 0; ii < count; ii ++ ) {
            const DNN_NUMERIC rr = random( 3.0, 5.0 );
            const DNN_NUMERIC tt = 2.0 * M_PI * val / count;
            const DNN_NUMERIC xx = rr * sin( tt );
            const DNN_NUMERIC yy = rr * cos( tt );
            setData( data, label, xx, yy, 1 );
            val += 1.0;
        }
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
    
    static bool
    setupDnn( Dnn &dnn ) {                                      // Same DNN for all 2d tests.
        if( dnn.count() > 0 ) {
            return log_error("Dnn already has layers");
        }
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
    localTest2d( DnnTrainer &trainer,
                const std::vector< DNN_NUMERIC > &data,
                const std::vector< DNN_INTEGER > &label,
                const char *trainingName,
                const unsigned long max_iteration = 200 ) {
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
        const DNN_INTEGER * const ePtr = lPtr + DATA_COUNT;

        const unsigned int      MAX_EPOCH = 200;
        const DNN_NUMERIC     TARGET_LOSS = 0.02;
        unsigned long      iterationCount = 0;
        DNN_NUMERIC          average_loss = 0.0;
        do {
            iterationCount++;
            average_loss = 0.0;
            for( unsigned int ii = 0; ii < MAX_EPOCH; ii++ ) {
                dPtr = data.data();
                lPtr = label.data();
                
                while( lPtr < ePtr ) {
                    DNN_NUMERIC *inPtr = input->data();
                    *inPtr++ = *dPtr++;     // x
                    *inPtr   = *dPtr++;     // y
                    average_loss += trainer.train( *lPtr++ );
                }
            }
            average_loss /= DATA_COUNT * MAX_EPOCH;
        } while( average_loss > TARGET_LOSS && iterationCount < max_iteration );
        log_info( "Average loss = %f/%f. Count = %lu/%lu (%s)", average_loss, TARGET_LOSS, iterationCount, max_iteration, trainingName );
        return true;
    }
    
        static bool
    localTest2dAdaDelta( const std::vector< DNN_NUMERIC > &data, const std::vector< DNN_INTEGER > &label ) {
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerAdaDelta trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.momentum(     0.1   );
        trainer.batchSize(   10     );
        trainer.l2Decay(      0.001 );
        return localTest2d( trainer, data, label, "AdaDelta" );
    }
    
    static bool
    localTest2dAdam( const std::vector< DNN_NUMERIC > &data, const std::vector< DNN_INTEGER > &label ) {
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerAdam trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.momentum(     0.1   );
        trainer.batchSize(   10     );
        trainer.l2Decay(      0.001 );
        return localTest2d( trainer, data, label, "Adam" );
    }
    
    static bool
    localTest2dNesterov( const std::vector< DNN_NUMERIC > &data, const std::vector< DNN_INTEGER > &label ) {
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerNesterov trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.momentum(     0.1   );
        trainer.batchSize(   10     );
        trainer.l2Decay(      0.001 );
        return localTest2d( trainer, data, label, "Nesterov" );
    }

    static bool
    localTest2dAdaSGD( const std::vector< DNN_NUMERIC > &data, const std::vector< DNN_INTEGER > &label ) {
        Dnn dnn;
        if( !setupDnn( dnn )) {
            return false;
        }
        DnnTrainerSGD trainer( &dnn );
        trainer.learningRate( 0.01  );
        trainer.momentum(     0.1   );
        trainer.batchSize(   10     );
        trainer.l2Decay(      0.001 );
        return localTest2d( trainer, data, label, "SGD" );
    }

    static bool
    localTest2d( const std::vector< DNN_NUMERIC > &data, const std::vector< DNN_INTEGER > &label ) {
        if( !localTest2dAdaDelta( data, label )) {
            return log_error( "Cannot train AdaDelta" );
        }
        if( !localTest2dAdam( data, label )) {
            return log_error( "Cannot train Adam" );
        }
        if( !localTest2dNesterov( data, label )) {
            return log_error( "Cannot train Nesterov" );
        }
        if( !localTest2dAdaSGD( data, label )) {   // with momentum
            return log_error( "Cannot train SGD" );
        }
        return true;
    }
    
    static bool
    localTestSimple( void ) {
        log_info( "Test Simple - Start" );
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        setUpDataSimple( data, label );
        
        localTest2d( data, label );
        
        log_info( "Test Simple - End" );
        return true;
    }
    
    static bool
    localTestRandom( void ) {
        log_info( "Test Random - Start" );
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        setUpDataRandom( data, label, 40 );
        
        localTest2d( data, label );
        
        log_info( "Test Random - End" );
        return true;
    }

    static bool
    localTestCircle( void ) {
        log_info( "Test Circle - Start" );
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        setUpDataCircle( data, label, 50 );
        
        localTest2d( data, label );

        log_info( "Test Circle - End" );
        return true;
    }

    static bool
    localTestSpiral( void ) {
        log_info( "Test Spiral - Start" );
        std::vector< DNN_NUMERIC > data;    // x,y pairs
        std::vector< DNN_INTEGER > label;   // binary labels.
        
        setUpDataSpiral( data, label, 100.0 );
        
        localTest2d( data, label );

        log_info( "Test Spiral - End" );
        return true;
    }

}  // tfs namespace

bool
testSimple( void ) {
    return tfs::localTestSimple();
}

bool
testRandom( void ) {
    return tfs::localTestRandom();
}

bool
testCircle( void ) {
    return tfs::localTestCircle();
}

bool
testSpiral( void ) {
    return tfs::localTestSpiral();
}
