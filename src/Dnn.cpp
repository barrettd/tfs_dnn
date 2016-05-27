// --------------------------------------------------------------------
//  Dnn.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "Dnn.h"
#include "DnnLayers.h"
#include "Error.h"

namespace tfs {
   
    Dnn::Dnn( void ) :
    m_layer_input(    0 ),
    m_layer_previous( 0 ),
    m_layer_output(   0 ),
    m_trainable(   true ) {
        // Constructor
    }
    
    Dnn::~Dnn( void ) {
        // Destructor
        clear();
    }
    
    bool
    Dnn::trainable( void ) const {
        return m_trainable;
    }
    bool
    Dnn::trainable( const bool value ) {
        return m_trainable = value;
    }

    void
    Dnn::clear( void ) {
        // Delete all of the layers.
        m_layer_input    = 0;
        m_layer_previous = 0;
        m_layer_output   = 0;
        std::vector< DnnLayer* >::const_iterator layer_end = m_layers.end();
        for( std::vector< DnnLayer* >::const_iterator it = m_layers.begin(); it != layer_end; it++ ) {
            DnnLayer *layer = *it;
            delete layer;
        }
        m_layers.clear();
        return;
    }
    
    unsigned long
    Dnn::count( void ) const {
        // Return the number of layers
        return m_layers.size();
    }
    
    bool
    Dnn::addLayer( DnnLayerInput *layer ) {
        // Add the input layer to our collection.  It should be the first layer.
        if( layer == 0 ) {
            return log_error( "null layer" );
        }
        if( !m_layers.empty()) {
            delete layer;
            return log_error( "Input layer should be the first layer" );
        }
        m_layers.push_back( layer );
        m_layer_previous = layer;           // Remember the previous layer.
        m_layer_input    = layer;           // Remember our input layer
        return true;
    }
    
    bool
    Dnn::addLayer( DnnLayer *layer ) {
        // Add subsequent layers to our collection.
        if( layer == 0 ) {
            return log_error( "null layer" );
        }
        if( m_layers.empty()) {
            delete layer;
            return log_error( "Input layer should be the first layer" );
        }
        m_layers.push_back( layer );
        m_layer_previous = layer;         // Remember the previous layer.
        m_layer_output   = layer;         // Remember the last layer seen as the output layer.
        return true;
    }
    
    bool
    Dnn::addLayerInput( unsigned long xx, unsigned long yy, unsigned long zz, const bool retain_dw ) {
        // Add an input layer
        if( xx < 1 || yy < 1 || zz < 1 ) {
            return log_error( "bad args" );
        }
        DnnLayerInput *layer = new DnnLayerInput( xx, yy, zz, m_trainable, retain_dw );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerConvolution( unsigned long side, unsigned long filters, unsigned long stride, unsigned long pad ) {
        // Add a Convolution Layer
        DnnLayerConvolution *layer = new DnnLayerConvolution( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerDropout( void ) {
        DnnLayerDropout *layer = new DnnLayerDropout( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerFullyConnected( unsigned long neuronCount ) {
        DnnLayerFullyConnected *layer = new DnnLayerFullyConnected( m_layer_previous, neuronCount, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerLocalResponseNormalization( void ) {
        DnnLayerLocalResponseNormalization *layer = new DnnLayerLocalResponseNormalization( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerMaxout( void ) {
        DnnLayerMaxout *layer = new DnnLayerMaxout( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerPool( unsigned long side, unsigned long stride ) {
        DnnLayerPool *layer = new DnnLayerPool( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerRectifiedLinearUnit( void ) {
        // Add a ReLu layer
        DnnLayerRectifiedLinearUnit *layer = new DnnLayerRectifiedLinearUnit( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerRegression( void ) {
        DnnLayerRegression *layer = new DnnLayerRegression( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerSigmoid( void ) {
        DnnLayerSigmoid *layer = new DnnLayerSigmoid( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerSoftmax( void ) {
        DnnLayerSoftmax *layer = new DnnLayerSoftmax( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerSupportVectorMachine( void ) {
        DnnLayerSupportVectorMachine *layer = new DnnLayerSupportVectorMachine( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    bool
    Dnn::addLayerTanh( void ) {
        DnnLayerTanh *layer = new DnnLayerTanh( m_layer_previous, m_trainable );
        return addLayer( layer );
    }
    
    Matrix*
    Dnn::getMatrixInput( void ) {
        Matrix *matrix = 0;
        if( m_layer_input != 0 ) {
            matrix = m_layer_input->outA();     // The output activation matrix is the input layer.
        }
        return matrix;
    }

    Matrix*
    Dnn::getMatrixOutput( void ) {
        Matrix *matrix = 0;
        if( m_layer_output != 0 ) {
            matrix = m_layer_output->outA();    // The output activation matrix of the output layer.
        }
        return matrix;
    }
    
    DnnLayerInput*
    Dnn::getLayerInput(  void ) {
        return m_layer_input;
    }
    
    DnnLayer*
    Dnn::getLayerOutput( void ) {
        return m_layer_output;
    }

    void
    Dnn::initialize( void ) {
        // Initialize for learning
        if( m_layer_input != 0 ) {
            m_layer_input->initialize();     // Calls each layer in the forward direction.
        } else {
            log_error( "No input layer" );
        }
        return;
    }

    void
    Dnn::randomize( void ) {
        // Randomize weights and bias.
        if( m_layer_input != 0 ) {
            m_layer_input->randomize();     // Calls each layer in the forward direction.
        } else {
            log_error( "No input layer" );
        }
        return;
    }
    
    bool
    Dnn::forward( void ) {
        // Forward propagate while training
        if( m_layer_input != 0 ) {
            return m_layer_input->forward();      // Calls each layer in the forward direction.
        }
        return log_error( "No input layer" );
    }
    
    DNN_NUMERIC
    Dnn::backprop( const Matrix &expectation ) {
        // Back propagate while training
        if( m_layer_output == 0 ) {
            log_error( "No output layer" );
            return 0.0;
        }
        return m_layer_output->backprop( expectation ); // Calls each layer in the backward direction.
    }
    
    DNN_NUMERIC
    Dnn::backprop( const DNN_INTEGER expectation ) {
        // Back propagate while training
        if( m_layer_output == 0 ) {
            log_error( "No output layer" );
            return 0.0;
        }
        return m_layer_output->backprop( expectation ); // Calls each layer in the backward direction.
    }
    
    bool
    Dnn::predict( void ) {
        // Forward progagate when predicting
        if( m_layer_input != 0 ) {
            return m_layer_input->predict();            // Calls each layer in the forward direction.
        }
        return log_error( "No input layer" );
    }

    
    // Binary file I/O
    bool
    Dnn::save( const char *file_path ) const  {
        if( file_path == 0 || *file_path == 0 ) {
            return log_error( "bad file path" );
        }
        return true;
    }
    
    bool
    Dnn::load( const char *file_path ) {
        if( file_path == 0 || *file_path == 0 ) {
            return log_error( "bad file path" );
        }
        return true;
    }
    
    // JSON file I/O, compatible with ConvNetJs
    bool
    Dnn::saveJson( const char *file_path ) const {
        if( file_path == 0 || *file_path == 0 ) {
            return log_error( "bad file path" );
        }
        return true;
    }
    
    bool
    Dnn::loadJson( const char *file_path ) {
        if( file_path == 0 || *file_path == 0 ) {
            return log_error( "bad file path" );
        }
        return true;
    }

    

    
}   // namespace tfs
